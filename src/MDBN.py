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
import zipfile
import urllib

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy
import theano
from theano import tensor
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from theano.compile.nanguardmode import NanGuardMode

from utils import zscore
from utils import get_minibatches_idx
from rbm import RBM
from rbm import GRBM

from MNIST import MNIST

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

        self.activation = activation

        # parameters of the model
        self.params = [self.W, self.b]

    def output(self, input=None):
        if input is None:
            input = self.input
        lin_output = tensor.dot(input, self.W) + self.b
        result = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        return result


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
                layer_input = self.sigmoid_layers[-1].output()

            print('Adding a layer with %i input and %i outputs' %
                  (input_size, stacked_layers_sizes[i]))
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


    def pretraining_functions(self, train_set_x, validation_set_x,
                              batch_size, k, monitor=False):
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

        # TODO: deal with batch_size of 1
        assert batch_size > 1

        pretrain_fns = []
        overfit_monitor_fns = []
        for i, rbm in enumerate(self.rbm_layers):
            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            if isinstance(rbm, GRBM):
                cost, updates = rbm.get_cost_updates(learning_rate,
                                                     lambda_2 = 0.1,
                                                     batch_size=batch_size,
                                                     persistent=None, k=k)
            else:
                cost, updates = rbm.get_cost_updates(learning_rate,
                                                     weightcost = 0.002,
                                                     batch_size=batch_size,
                                                     persistent=None, k=k)

            # compile the theano function
            if monitor:
                mode = theano.compile.MonitorMode(pre_func=self.inspect_inputs)
            else:
                mode = theano.config.mode

            fn = theano.function(
                inputs=[indexes, momentum, theano.In(learning_rate)],
                outputs=cost,
                updates=updates,
                givens={
                        self.x: train_set_x[indexes],
                        rbm.momentum: momentum
                },
                mode = mode
    #           mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
            )

            # append `fn` to the list of functions
            pretrain_fns.append(fn)

            train_sample = tensor.matrix('train_smaple', dtype=theano.config.floatX)
            validation_sample = tensor.matrix('validation_smaple', dtype=theano.config.floatX)

            feg = rbm.free_energy_gap(train_sample, validation_sample)

            if i == 0:
                fn = theano.function(
                    inputs=[indexes],
                    outputs=feg,
                    givens={
                        train_sample: train_set_x[indexes],
                        validation_sample: validation_set_x
                    },
                    mode=mode
                )
            else:
                t_sample = self.sigmoid_layers[i-1].output(train_set_x)
                v_sample = self.sigmoid_layers[i-1].output(validation_set_x)
                fn = theano.function(
                    inputs=[indexes],
                    outputs=feg,
                    givens={
                        train_sample: t_sample[indexes],
                        validation_sample: v_sample
                    },
                    mode=mode
                )

            overfit_monitor_fns.append(fn)

        return pretrain_fns, overfit_monitor_fns

    def pretraining(self, train_set_x, validation_set_x,
                    batch_size, k,
                    pretraining_epochs, pretrain_lr,
                    monitor=False):
        #########################
        # PRETRAINING THE MODEL #
        #########################

        print('... getting the pretraining functions')
        print('Training set sample size %i' % train_set_x.get_value().shape[0])
        print('Validation set sample size %i' % validation_set_x.get_value().shape[0])

        pretraining_fns, overfitting_monitor_fns = self.pretraining_functions(train_set_x=train_set_x,
                                                     validation_set_x=validation_set_x,
                                                     batch_size=batch_size,
                                                     k=k,
                                                     monitor=monitor)

        print('... pre-training the model')
        start_time = timeit.default_timer()
        # Pre-train layer-wise

        n_data = train_set_x.get_value().shape[0]

        for i in range(self.n_layers):
            if isinstance(self.rbm_layers[i], GRBM):
                momentum = 0.0
            else:
                momentum = 0.6

            print_frequency = int(pretraining_epochs[i] / 40)
            if print_frequency == 0:
                print_frequency = 1

            print('Printing every %i epochs' % print_frequency)
            # go through pretraining epochs
            mean_cost = []
            free_energy = []
            for epoch in range(pretraining_epochs[i]):
                _, minibatches = get_minibatches_idx(n_data,
                                                    batch_size,
                                                    shuffle=True)
                # go through the training set
                c = []
                f = []
                if not isinstance(self.rbm_layers[i], GRBM) and epoch == 6:
                    momentum = 0.9

                for mb, minibatch in enumerate(minibatches):
                    c.append(pretraining_fns[i](indexes=range(39),
                                                momentum=momentum,
                                                lr=pretrain_lr[i]))

                    if mb == 0:
                        f.append(overfitting_monitor_fns[i](indexes=minibatch))

                if epoch % print_frequency == 0:
                    print('Free energy gap (layer %i, epoch %i): ' % (i, epoch), end=' ')
                    free_energy.append(numpy.mean(f))
                    print(free_energy[-1])
                    print('Pre-training cost (layer %i, epoch %d): ' % (i, epoch), end=' ')
                    mean_cost.append(numpy.mean(c))
                    print(mean_cost[-1])

#            plt.plot(mean_cost)
#            plt.plot(free_energy)

        end_time = timeit.default_timer()

        print('The pretraining code for file ' + os.path.split(__file__)[1] +
              ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)


    def output(self, dataset):
        fn = theano.function(inputs=[],
                             outputs=self.sigmoid_layers[-1].output(),
                             givens={
                                 self.x: dataset
                             })
        return fn

# batch_size changed from 1 as in M.Liang to 20

def test(datafiles,
         datadir='data',
         batch_size=20,
         output_folder='MDBN_run',
         rng=None):
    """
    :param datafile: path to the dataset

    :param batch_size: size of a batch used to train the RBM
    """

    if rng is None:
        rng = numpy.random.RandomState(123)

    #################################
    #     Training the RBM          #
    #################################


    rna_DBN, output_RNA = train_RNA(datafiles['mRNA'], datadir)

    ge_DBN, output_GE = train_GE(datafiles['GE'], datadir)

    me_DBN, output_ME = train_ME(datafiles['ME'], datadir)

    print('*** Training on joint layer ***')

    joint_data = theano.shared(numpy.concatenate([
                    output_RNA, output_GE, output_ME],axis=1))

    top_DBN = DBN(numpy_rng=rng, n_ins=120,
                  gauss=False,
                  hidden_layers_sizes=[24],
                  n_outs=8)
    top_DBN.pretraining(joint_data, joint_data.get_value().shape[0],
                        batch_size, k=1,
                        pretraining_epochs=[800, 800],
                        pretrain_lr=[0.1, 0.1])

    classes = top_DBN.output(joint_data, range(joint_data.get_value().shape[0]))

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

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

def importdata(file, datadir):
    root_dir = os.getcwd()
    os.chdir(datadir)
    with open(file) as f:
        ncols = len(f.readline().split('\t'))

    data = numpy.loadtxt(file,
                       dtype=theano.config.floatX,
                       delimiter='\t',
                       skiprows=1,
                       usecols=range(1,ncols))

    os.chdir(root_dir)
    return (data.shape[1], ncols-1, data)

def load_n_preprocess_data(datafile, datadir='data'):
    # Load the data, each column is a single person
    # Pass to a row representation, i.e. the data for each person is now on a
    # single row.
    # Normalize the data so that each measurement on our population has zero
    # mean and zero variance
    _, _, data = importdata(datafile, datadir)
    data = zscore(data.T)
    train_set = theano.shared(data[:int(data.shape[0] * 0.9)], borrow=True)
    validation_set = theano.shared(data[int(data.shape[0] * 0.9):], borrow=True)
    return train_set, validation_set

def train_bottom_layer(train_set, validation_set,
                       batch_size=20,
                       k=1, layers_sizes=[40],
                       pretraining_epochs=[800],
                       pretrain_lr=[0.0005],
                       rng=None
                    ):

    if rng is None:
        rng = numpy.random.RandomState(123)

    print('Visible nodes: %i' % train_set.get_value().shape[1])
    print('Output nodes: %i' % layers_sizes[-1])
    dbn = DBN(numpy_rng=rng, n_ins=train_set.get_value().shape[1],
                  hidden_layers_sizes=layers_sizes[:-1],
                  n_outs=layers_sizes[-1])

    dbn.pretraining(train_set,
                        validation_set,
                        batch_size, k=k,
                        pretraining_epochs=pretraining_epochs,
                        pretrain_lr=pretrain_lr)
    output = dbn.output(train_set)
    return dbn, output

def train_ME(datafile, datadir='data'):
    print('*** Training on ME ***')

    train_set, validation_set = load_n_preprocess_data(datafile, datadir)

    return train_bottom_layer(train_set, validation_set,
                              batch_size=20,
                              k=1,
                              layers_sizes=[400, 40],
                              pretraining_epochs=[8000, 800],
                              pretrain_lr=[0.0005, 0.1])

def train_GE(datafile, datadir='data'):
    print('*** Training on GE ***')

    train_set, validation_set = load_n_preprocess_data(datafile, datadir)

    return train_bottom_layer(train_set, validation_set,
                              batch_size=20,
                              k=1,
                              layers_sizes=[400, 40],
                              pretraining_epochs=[8000, 800],
                              pretrain_lr=[0.0005, 0.1])

def train_RNA(datafile, datadir='data'):
    print('*** Training on RNA ***')

    train_set, validation_set = load_n_preprocess_data(datafile, datadir)

    return train_bottom_layer(train_set, validation_set,
              batch_size=10,
              k=10,
              layers_sizes=[40],
              pretraining_epochs=[1600],
              pretrain_lr=[0.0005])

def train_MNIST_Gaussian():
    # Load the data
    mnist = MNIST()
    raw_dataset = mnist.images
    n_data = raw_dataset.shape[0]

    dataset = mnist.normalize(raw_dataset)

    train_set = theano.shared(dataset[0:int(n_data*5/6)], borrow=True)
    validation_set = theano.shared(dataset[-39:], borrow=True)

    print('*** Training on MNIST ***')

    return train_bottom_layer(train_set, validation_set,
                          batch_size=20,
                          k=1,
                          layers_sizes=[500],
                          pretraining_epochs=[5],
                          pretrain_lr=[0.01])


def prepare_datafiles(datadir='data'):
    base_url = 'http://nar.oxfordjournals.org/content/suppl/2012/07/25/gks725.DC1/'
    archive = 'nar-00961-n-2012-File005.zip'
    datafiles = {
        'GE': 'TCGA_Data/3.GE1_0.5.out',
        'ME': 'TCGA_Data/3.Methylation_0.5.out',
        'mRNA': 'TCGA_Data/3.miRNA_0.5.out'
    }
    if not os.path.isdir(datadir):
        os.mkdir(datadir)
    root_dir = os.getcwd()
    os.chdir(datadir)
    for name, datafile in datafiles.iteritems():
        if not os.path.isfile(datafile):
            if not os.path.isfile(archive):
                print('Downloading TCGA_Data from ' + base_url)
                testfile = urllib.URLopener()
                testfile.retrieve(base_url + archive, archive)
            zipfile.ZipFile(archive, 'r').extract(datafile)
    os.chdir(root_dir)
    return datafiles

if __name__ == '__main__':
    datafiles = prepare_datafiles()
#    test(datafiles)

#    train_RNA(datafiles['mRNA'])
    train_GE(datafiles['GE'])

#    train_MNIST_Gaussian()
