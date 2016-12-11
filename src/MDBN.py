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

import numpy
import theano
from theano import tensor
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from theano.compile.nanguardmode import NanGuardMode

from utils import get_minibatches_idx
from utils import load_n_preprocess_data

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

        lin_output = tensor.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
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

    def __init__(self, numpy_rng=None, theano_rng=None, input=None, n_ins=784,
                 gauss=True,
                 hidden_layers_sizes=[400], n_outs=40,
                 W_list=None, b_list=None):
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

        self.n_ins = n_ins
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.stacked_layers_sizes = hidden_layers_sizes + [n_outs]
        self.n_layers = len(self.stacked_layers_sizes)

        assert self.n_layers > 0

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(123)

        if theano_rng is None:
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
                input_size = self.stacked_layers_sizes[i - 1]

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

            n_in = input_size
            n_out= self.stacked_layers_sizes[i]

            print('Adding a layer with %i input and %i outputs' %
                  (n_in, n_out))

            if W_list is None:
                W = numpy.asarray(numpy_rng.uniform(
                                low=-4.*numpy.sqrt(6. / (n_in + n_out)),
                                high=4.*numpy.sqrt(6. / (n_in + n_out)),
                                size=(n_in, n_out)
                             ),dtype=theano.config.floatX)
            else:
                W = W_list[i]

            if b_list is None:
                b = numpy.zeros((n_out,), dtype=theano.config.floatX)
            else:
                b = b_list[i]

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=n_in,
                                        n_out=n_out,
                                        W=theano.shared(W,name='W',borrow=True),
                                        b=theano.shared(b,name='b',borrow=True),
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
                                n_hidden=self.stacked_layers_sizes[i],
                                W=sigmoid_layer.W,
                                hbias=sigmoid_layer.b)
            else:
                rbm_layer = RBM(numpy_rng=numpy_rng,
                                theano_rng=theano_rng,
                                input=layer_input,
                                n_visible=input_size,
                                n_hidden=self.stacked_layers_sizes[i],
                                W=sigmoid_layer.W,
                                hbias=sigmoid_layer.b)

            self.rbm_layers.append(rbm_layer)

    def number_of_nodes(self):
        return [self.n_ins] + self.stacked_layers_sizes

    def inspect_inputs(self, i, node, fn):
        print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs],
              end='\n')

    def inspect_outputs(self, i, node, fn):
        print(" output(s) value(s):", [output[0] for output in fn.outputs])

    def get_output(self, input, layer=-1):
        if input is not None:
            fn = theano.function(inputs=[],
                                 outputs=self.sigmoid_layers[layer].output,
                                 givens={
                                     self.x: input
                                 })
            return fn()
        else:
            return None

    def pretraining_functions(self, train_set_x, validation_set_x,
                              batch_size, k, lambda_1 = 0.0, lambda_2 = 0.1, monitor=False):
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
        free_energy_gap_fns = []
        for i, rbm in enumerate(self.rbm_layers):
            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            if isinstance(rbm, GRBM):
                cost, updates = rbm.get_cost_updates(learning_rate,
                                                     lambda_1=lambda_1,
                                                     lambda_2 = lambda_2,
                                                     batch_size=batch_size,
                                                     persistent=None, k=k)
            else:
                cost, updates = rbm.get_cost_updates(learning_rate,
                                                     weightcost = 0.0002,
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
            test_sample = tensor.matrix('validation_smaple', dtype=theano.config.floatX)

            feg = rbm.free_energies(train_sample, test_sample)

            # Obtain the input of layer i as the output of the previous
            # layer
            fn = theano.function(
                inputs=[train_sample, test_sample],
                outputs=feg,
                mode=mode
            )

            free_energy_gap_fns.append(fn)

        return pretrain_fns, free_energy_gap_fns

    def pretraining(self, train_set_x, validation_set_x,
                    batch_size, k,
                    pretraining_epochs, pretrain_lr,
                    lambda_1 = 0.0,
                    lambda_2 = 0.1,
                    monitor=False, graph_output=False):
        #########################
        # PRETRAINING THE MODEL #
        #########################

        print('... getting the pretraining functions')
        print('Training set sample size %i' % train_set_x.get_value().shape[0])
        if validation_set_x is not None:
            print('Validation set sample size %i' % validation_set_x.get_value().shape[0])

        pretraining_fns, free_energy_gap_fns = self.pretraining_functions(train_set_x=train_set_x,
                                                     validation_set_x=validation_set_x,
                                                     batch_size=batch_size,
                                                     k=k,
                                                     lambda_1=lambda_1,
                                                     lambda_2=lambda_2,
                                                     monitor=monitor)

        print('... pre-training the model')
        start_time = timeit.default_timer()
        # Pre-train layer-wise

        if graph_output:
            plt.ion()

        n_data = train_set_x.get_value().shape[0]

        if validation_set_x is not None:
            t_set = train_set_x.get_value(borrow=True)
            v_set = validation_set_x.get_value(borrow=True)

        # early-stopping parameters

        patience_increase = 2  # wait this much longer when a new best is
        # found
        improvement_threshold = 0.995  # a relative improvement of this much is
        # considered significant

        # go through this many
        # minibatches before checking the network
        # on the validation set; in this case we
        # check every epoch

        idx_minibatches, minibatches = get_minibatches_idx(n_data,
                                                           batch_size,
                                                           shuffle=True)

        n_train_batches = idx_minibatches[-1] + 1

        for i in range(self.n_layers):
            if graph_output:
                plt.figure(i+1)

            if isinstance(self.rbm_layers[i], GRBM):
                momentum = 0.0
            else:
                momentum = 0.6

            # go through pretraining epochs
            best_cost = numpy.inf
            epoch = 0
            done_looping = False

            patience = pretraining_epochs[i]  # look as this many examples regardless
            validation_frequency = min(20 * n_train_batches, patience // 2)
            print('Validation frequency: %d' % validation_frequency)

            while (epoch < pretraining_epochs[i]) and (not done_looping):
                epoch = epoch + 1

                idx_minibatches, minibatches = get_minibatches_idx(n_data,
                                                                   batch_size,
                                                                   shuffle=True)

                # go through the training set
                if not isinstance(self.rbm_layers[i], GRBM) and epoch == 6:
                    momentum = 0.9

                for mb, minibatch in enumerate(minibatches):
                    current_cost = pretraining_fns[i](indexes=minibatch,
                                                momentum=momentum,
                                                lr=pretrain_lr[i])
                    # iteration number
                    iter = (epoch - 1) * n_train_batches + mb

                    if (iter + 1) % validation_frequency == 0:
                        print('Pre-training cost (layer %i, epoch %d): ' % (i, epoch), end=' ')
                        print(current_cost)

                        # Plot the output
                        if graph_output:
                            plt.clf()
                            training_output = self.get_output(train_set_x, i)
                            plt.imshow(training_output, cmap='gray')
                            plt.axis('tight')
                            plt.title('epoch %d' % (epoch))
                            plt.draw()
                            plt.pause(1.0)

                        # if we got the best validation score until now
                        if current_cost < best_cost:
                            # improve patience if loss improvement is good enough
                            if (
                                    current_cost < best_cost *
                                    improvement_threshold
                            ):
                                patience = max(patience, iter * patience_increase)

                            best_cost = current_cost
                            best_iter = iter

                            if validation_set_x is not None:
                                # Compute the free energy gap
                                if i == 0:
                                    input_t_set = t_set
                                    input_v_set = v_set
                                else:
                                    input_t_set = self.get_output(
                                                    t_set[range(v_set.shape[0])], i-1)
                                    input_v_set = self.get_output(v_set, i-1)

                                free_energy_train, free_energy_test = free_energy_gap_fns[i](
                                                    input_t_set,
                                                    input_v_set)
                                free_energy_gap = free_energy_test.mean() - free_energy_train.mean()

                                print('Free energy gap (layer %i, epoch %i): ' % (i, epoch), end=' ')
                                print(free_energy_gap)

                    if patience <= iter:
                        done_looping = True
                        break

            if graph_output:
                plt.close()

        end_time = timeit.default_timer()


        print('The pretraining code for file ' + os.path.split(__file__)[1] +
              ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)

    def MLP_output_from_datafile(self,
                                 datafile,
                                 holdout=0.0,
                                 repeats=1,
                                 clip=None,
                                 transform_fn=None,
                                 exponent=1.0,
                                 datadir='data'):
        train_set, validation_set = load_n_preprocess_data(datafile,
                                                           holdout=holdout,
                                                           clip=clip,
                                                           transform_fn=transform_fn,
                                                           exponent=exponent,
                                                           repeats=repeats,
                                                           shuffle=False,
                                                           datadir=datadir)

        return (self.get_output(train_set), self.get_output(validation_set))

def train_top(batch_size, graph_output, joint_train_set, joint_val_set, rng):
    top_DBN = DBN(numpy_rng=rng, n_ins=joint_train_set.get_value().shape[1],
                  gauss=False,
                  hidden_layers_sizes=[24],
                  n_outs=3)
    top_DBN.pretraining(joint_train_set, joint_val_set,
                        batch_size, k=1,
                        pretraining_epochs=[800, 800],
                        pretrain_lr=[0.1, 0.1],
                        graph_output=graph_output)
    return top_DBN


def train_bottom_layer(train_set, validation_set,
                       batch_size=20,
                       k=1, layers_sizes=[40],
                       pretraining_epochs=[800],
                       pretrain_lr=[0.005],
                       lambda_1 = 0.0,
                       lambda_2 = 0.1,
                       rng=None,
                       graph_output=False
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
                        pretrain_lr=pretrain_lr,
                        lambda_1=lambda_1,
                        lambda_2=lambda_2,
                        graph_output=graph_output)

    output_train_set = dbn.get_output(train_set)
    if validation_set is not None:
        output_val_set = dbn.get_output(validation_set)
    else:
        output_val_set = None

    return dbn, output_train_set, output_val_set

def train_MNIST_Gaussian(graph_output=False):
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
                                pretraining_epochs=[100],
                                pretrain_lr=[0.01],
                                graph_output=graph_output)

if __name__ == '__main__':
    train_MNIST_Gaussian(graph_output=True)