"""
Copyright (c) 2016 Gianluca Gerard

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Portions of the code are
Copyright (c) 2010--2015, Deep Learning Tutorials Development Team
All rights reserved.
"""

from __future__ import print_function, division

import timeit
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy
import theano
from theano import tensor
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from theano.compile.nanguardmode import NanGuardMode

from utils import zscore
from rbm import RBM
from rbm import GRBM

def get_minibatches_idx(n, batch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + batch_size])
        minibatch_start += batch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return range(len(minibatches)), minibatches

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=tensor.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        Originally from: http://deeplearning.net/tutorial/code/mlp.py

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = tensor.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer GRBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output.
    """

    def __init__(self, numpy_rng, theano_rng=None, input=None, n_ins=784,
                 gauss=True,
                 hidden_layers_sizes=[400], n_outs=40):
        """This class is made to support a variable number of layers.

        Originally from: http://deeplearning.net/tutorial/code/DBN.py

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type gauss: bool
        :param gauss: True if the first layer is Gaussian otherwise
                      the first layer is Binomial

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        stacked_layers_sizes = hidden_layers_sizes + [n_outs]
        self.n_layers = len(stacked_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data

        # the data is presented as rasterized images
        self.x = tensor.matrix('x')

        theano.printing.Print('this is a very important value')(self.x)

        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well).

        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = stacked_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                if input is None:
                    layer_input = self.x
                else:
                    layer_input = input
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=stacked_layers_sizes[i],
                                        activation=tensor.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            if i==0 and gauss:
                rbm_layer = GRBM(numpy_rng=numpy_rng,
                                theano_rng=theano_rng,
                                input=layer_input,
                                n_visible=input_size,
                                n_hidden=stacked_layers_sizes[i],
                                W=sigmoid_layer.W,
                                hbias=sigmoid_layer.b)
            else:
                rbm_layer = RBM(numpy_rng=numpy_rng,
                                theano_rng=theano_rng,
                                input=layer_input,
                                n_visible=input_size,
                                n_hidden=stacked_layers_sizes[i],
                                W=sigmoid_layer.W,
                                hbias=sigmoid_layer.b)

            self.rbm_layers.append(rbm_layer)

    def inspect_inputs(self, i, node, fn):
        print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs],
              end='\n')

    def inspect_outputs(self, i, node, fn):
        print(" output(s) value(s):", [output[0] for output in fn.outputs])


    def pretraining_functions(self, train_set_x, batch_size, k, monitor=False):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        indexes = tensor.lvector('indexes')  # index to a minibatch
        learning_rate = tensor.scalar('lr', dtype=theano.config.floatX)  # learning rate to use
        momentum = tensor.scalar('momentum', dtype=theano.config.floatX)
        test_sample = tensor.vector('test_smaple', dtype=theano.config.floatX)

        # TODO: deal with batch_size of 1

        assert batch_size > 1

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            if isinstance(rbm, GRBM):
                cost, updates = rbm.get_cost_updates(learning_rate,
                                                     lambda_2 = 0.1,
                                                     batch_size=batch_size-1,
                                                     persistent=None, k=k)
            else:
                cost, updates = rbm.get_cost_updates(learning_rate,
                                                     weightcost = 0.002,
                                                     batch_size=batch_size-1,
                                                     persistent=None, k=k)

            feg = rbm.free_energy_gap(test_sample)

            # compile the theano function
            if monitor:
                fn = theano.function(
                    inputs=[indexes, momentum, theano.In(learning_rate, value=0.1)],
                    outputs=[cost, feg],
                    updates=updates,
                    givens={
                        self.x: train_set_x[indexes[:-1]],  # leave one out
                        rbm.momentum: momentum,
                        test_sample: train_set_x[indexes[-1]]
                    }
                    , mode = theano.compile.MonitorMode(
                                 pre_func=self.inspect_inputs)
    #                ,mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                )
            else:
                fn = theano.function(
                    inputs=[indexes, momentum, theano.In(learning_rate, value=0.1)],
                    outputs=[cost, feg],
                    updates=updates,
                    givens={
                        self.x: train_set_x[indexes[:-1]],
                        rbm.momentum: momentum,
                        test_sample: train_set_x[indexes[-1]]
                    }
                )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def pretraining(self, train_set_x, n, batch_size, k,
                    pretraining_epochs, pretrain_lr,
                    monitor=False):
        #########################
        # PRETRAINING THE MODEL #
        #########################

        print('... getting the pretraining functions')
        pretraining_fns = self.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=batch_size,
                                                    k=k,
                                                    monitor=monitor)

        print('... pre-training the model')
        start_time = timeit.default_timer()
        # Pre-train layer-wise

        for i in range(self.n_layers):
            if isinstance(self.rbm_layers[i], GRBM):
                momentum = 0.0
            else:
                momentum = 0.6
            # go through pretraining epochs
            mean_cost = []
            free_energy = []
            for epoch in range(pretraining_epochs[i]):
                _, minibatches = get_minibatches_idx(n,
                                                    batch_size,
                                                    shuffle=False)
                # go through the training set
                c = []
                frequency = pretraining_epochs[i]/40
                if not isinstance(self.rbm_layers[i], GRBM) and epoch == 6:
                    momentum = 0.9
                for mb, minibatch in enumerate(minibatches):
                    c.append(pretraining_fns[i](indexes=minibatch,
                                                momentum=momentum,
                                                lr=pretrain_lr[i]))
                    if epoch % frequency == 0 and mb == 0:
                        print('Free energy gap layer %i, epoch %d ' % (i, epoch), end=' ')
                        print(c[-1][1])
                        free_energy.append(c[-1][1])

                if epoch % frequency == 0:
                    print('Pre-training layer %i, epoch %d, cost ' % (i, epoch), end=' ')
                    mean_cost.append(numpy.mean(c[0]))
                    print(mean_cost[-1])

            plt.plot(mean_cost)
            plt.plot(free_energy)

        end_time = timeit.default_timer()

        print('The pretraining code for file ' + os.path.split(__file__)[1] +
              ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)


    def output(self, dataset):
        fn = theano.function(inputs=[],
                             outputs=self.sigmoid_layers[-1].output,
                             givens={
                                 self.x: dataset
                             })
        return fn()

def importdata(file):
    with open(file) as f:
        ncols = len(f.readline().split('\t'))

    return (ncols-1,
            numpy.loadtxt(file,
                       dtype=theano.config.floatX,
                       delimiter='\t',
                       skiprows=1,
                       usecols=range(1,ncols)))

# batch_size changed from 1 as in M.Liang to 20

def test(batch_size=20,
             n_chains=20, n_samples=10, output_folder='MDBN_plots'):
    """

    :param datafile: path to the dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain

    """

    # Load the data, each column is a single person
    gecols, GE = importdata("../data/3.GE1_0.5.out")
    mecols, ME  = importdata("../data/3.Methylation_0.5.out")
    rnacols, mRNA  = importdata("../data/3.miRNA_0.5.out")
    # Pass to a row representation, i.e. the data for each person is now on a
    # single row.
    # Normalize the data so that each measurement on our population has zero
    # mean and zero variance
    normGE = zscore(GE.T)
    normME = zscore(ME.T)
    normRNA = zscore(mRNA.T)
    datage = theano.shared(normGE,borrow=True)
    datame = theano.shared(normME,borrow=True)
    datarna = theano.shared(normRNA,borrow=True)

    x = tensor.matrix('x')

    # compute number of minibatches for training, validation and testing
    n_data = datarna.get_value(borrow=True).shape[0]

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    print('*** Training on RNA ***')
    rna_DBN = DBN(numpy_rng=rng, n_ins=datarna.get_value().shape[1],
              hidden_layers_sizes=[],
              n_outs=40)
    rna_DBN.pretraining(datarna, n_data, batch_size, k=10,
                        pretraining_epochs=[2000],
                        pretrain_lr=[0.0005])

    output_RNA = rna_DBN.output(datarna)

    print('*** Training on GE ***')
    ge_DBN = DBN(numpy_rng=rng, n_ins=datage.get_value().shape[1],
              hidden_layers_sizes=[400],
              n_outs=40)
    ge_DBN.pretraining(datage, n_data, batch_size, k=1,
                       pretraining_epochs=[8000, 800],
                       pretrain_lr=[0.0005, 0.1])

    output_GE = ge_DBN.output(datage)

    print('*** Training on ME ***')
    me_DBN = DBN(numpy_rng=rng, n_ins=datame.get_value().shape[1],
              hidden_layers_sizes=[400],
              n_outs=40)
    me_DBN.pretraining(datame, n_data, batch_size, k=1,
                       pretraining_epochs=[8000, 800],
                       pretrain_lr=[0.0005, 0.1])

    output_ME = me_DBN.output(datame)

    print('*** Training on joint layer ***')

    joint_data = theano.shared(numpy.concatenate([
                    output_RNA, output_GE, output_ME],axis=1))

    top_DBN = DBN(numpy_rng=rng, n_ins=120,
                  gauss=False,
                  hidden_layers_sizes=[24],
                  n_outs=8)
    top_DBN.pretraining(joint_data, n_data, batch_size, k=1,
                        pretraining_epochs=[800, 800],
                        pretrain_lr=[0.1, 0.1])

    classes = top_DBN.output(joint_data)

    numpy.savez('parameters_at_gaussian_layer_RNA.npz',
             k=20,
             epoch=8000,
             batch_size=10,
             learning_rate=0.0005,
             stocastic_steps=False,
             momentum=False,
             weight_cost=False,
             classes=classes,
             rna_params=[{p.name: p.get_value()} for p in rna_DBN.params],
             ge_params=[{p.name: p.get_value()} for p in ge_DBN.params],
             me_params=[{p.name: p.get_value()} for p in me_DBN.params],
             top_params=[{p.name: p.get_value()} for p in top_DBN.params]
             )

    os.chdir('..')

if __name__ == '__main__':
    test()
