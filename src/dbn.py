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
from mlp import HiddenLayer

from MNIST import MNIST

class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.

    Originally from: http://deeplearning.net/tutorial/code/DBN.py
    """

    def __init__(self, numpy_rng=None, theano_rng=None, n_ins=784,
                 gauss=True,
                 hidden_layers_sizes=[400], n_outs=40,
                 W_list=None, b_list=None, c_list=None):
        """This class is made to support a variable number of layers.

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
                      the first layer is Bernoullian

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type W_list: list of numpy.ndarray
        :param W_list: the list of weigths matrixes for each layer of the MLP; if
                       None each matrix is randomly initialized

        :type b_list: list of numpy.ndarray
        :param b_list: the list of biases vectors for each hidden layer of the MLP; if
                       None each vector is randomly initialized

        :type c_list: list of numpy.ndarray
        :param c_list: the list of visual nodes biases vectors for the RBMs; if
                       None it is randomly initialized
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

        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng

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
                layer_input = self.x
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

            if c_list is None:
                c = numpy.zeros((n_in,), dtype=theano.config.floatX)
            else:
                c = c_list[i]

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=n_in,
                                        n_out=n_out,
                                        W=theano.shared(W,name='W',borrow=True),
                                        b=theano.shared(b,name='hbias',borrow=True),
                                        activation=tensor.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # Construct an RBM that shared weights with this layer
            if (i==0) and gauss:
                rbm_layer = GRBM(numpy_rng=numpy_rng,
                                 theano_rng=theano_rng,
                                 input=layer_input,
                                 n_visible=input_size,
                                 n_hidden=self.stacked_layers_sizes[i],
                                 W=sigmoid_layer.W,
                                 hbias=sigmoid_layer.b,
                                 vbias=theano.shared(c,name='vbias',borrow=True))
            else:
                rbm_layer = RBM(numpy_rng=numpy_rng,
                                theano_rng=theano_rng,
                                input=layer_input,
                                n_visible=input_size,
                                n_hidden=self.stacked_layers_sizes[i],
                                W=sigmoid_layer.W,
                                hbias=sigmoid_layer.b,
                                vbias=theano.shared(c,name='vbias',borrow=True))

            self.params.extend(rbm_layer.params)

            self.rbm_layers.append(rbm_layer)

    def number_of_nodes(self):
        '''
        Generate a list with the number of nodes in each layer

        :return: list of int representing the nodes at each layer
        '''
        return [self.n_ins] + self.stacked_layers_sizes

    def get_output(self, input, layer=-1):
        '''
        Return the output of the MLP layer of index layer when the network
        is presented a set of samples input.

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type layer: int
        :param layer: the index of the layer; if None it defaults to the
                      last layer of the network

        :return: a theano.function object or None if the input is None
        '''
        if input is not None:
            fn = theano.function(inputs=[],
                                 outputs=self.sigmoid_layers[layer].output,
                                 givens={
                                     self.x: input
                                 })
            return fn()
        else:
            return None

    def training_functions(self, train_set_x, k,
                           max_batch_size=20,
                           lambdas = [0.0, 0.1],
                           persistent=False,
                           monitor=False):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the DBN

        :type k: int
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        :type lambdas: list of float
        :param lambdas: parameters for tuning weigths updates in CD-k/PCD-k
                         of Bernoullian RBM

        :type monitor: bool
        :param monitor: set to true to enable theano debugging Monitoring Mode;
                        default is false
        '''

        # index to a [mini]batch
        indexes = tensor.lvector('indexes')  # index to a minibatch
        learning_rate = tensor.scalar('lr', dtype=theano.config.floatX)  # learning rate to use
        batch_size = tensor.iscalar('batch_size')
        momentum = tensor.scalar('momentum', dtype=theano.config.floatX)

        train_fns = []
        free_energy_gap_fns = []
        persistent_chain = []

        for i, rbm in enumerate(self.rbm_layers):
            # initialize storage for the persistent chain (state = hidden
            # layer of chain) if persisten is True
            if persistent:
                persistent_chain.append(theano.shared(numpy.zeros((max_batch_size, rbm.n_hidden),
                                                         dtype=theano.config.floatX),
                                             borrow=True))
            else:
                persistent_chain.append(None)

            # get the cost and the updates list
            if isinstance(rbm, GRBM):
                cost, updates = rbm.get_cost_updates(learning_rate,
                                                     lambdas=lambdas,
                                                     batch_size=batch_size,
                                                     persistent=persistent_chain[i],
                                                     k=k)
            else:
                cost, updates = rbm.get_cost_updates(learning_rate,
                                                     weightcost = 0.0002,
                                                     batch_size=batch_size,
                                                     persistent=persistent_chain[i],
                                                     k=k)

            # compile the theano function
            if monitor:
                mode = theano.compile.MonitorMode(pre_func=self.inspect_inputs)
            else:
                mode = theano.config.mode

            fn = theano.function(
                inputs=[indexes, momentum, theano.In(learning_rate), theano.In(batch_size)],
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
            train_fns.append(fn)

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

        return train_fns, free_energy_gap_fns

    def training(self, train_set_x,
                 batch_size, k,
                 pretraining_epochs, pretrain_lr,
                 lambdas = [0.0, 0.1],
                 persistent=False,
                 validation_set_x=None,
                 run=0,
                 verbose=False,
                 monitor=False,
                 graph_output=False):
        '''
        Run the DBN pretraining.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the DBN

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type k: int
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        :type pretraining_epochs: int
        :param pretraining_epochs: number of epochs used for pretraining

        :type pretrain_lr: float
        :param pretrain_lr: learning rate

        :type lambdas: list of float
        :param lambdas: parameters for tuning weigths updates in CD-k/PCD-k
                         of Bernoullian RBM

        :type validation_set_x: theano.tensor.TensorType
        :param validation_set_x: Shared var. that contains all datapoints used
                            for validating the DBN

        :type monitor: bool
        :param monitor: set to true to enable theano debugging Monitoring Mode;
                        default is false

        :type graph_output: bool
        :param graph_output: set to true to enable graphical output;
                        default is false

        :return:
        '''

        print('... getting the pretraining functions')
        print('Training set sample size %i' % train_set_x.get_value().shape[0])
        if validation_set_x is not None:
            print('Validation set sample size %i' % validation_set_x.get_value().shape[0])

        training_fns, free_energy_gap_fns = self.training_functions(train_set_x=train_set_x,
                                                                    k=k,
                                                                    max_batch_size=batch_size,
                                                                    lambdas=lambdas,
                                                                    persistent=persistent,
                                                                    monitor=monitor)

        print('... pre-training the model')
        start_time = timeit.default_timer()
        # train layer-wise

        if graph_output:
            plt.ion()

        n_data = train_set_x.get_value().shape[0]

        if validation_set_x is not None:
            t_set = train_set_x.get_value(borrow=True)
            v_set = validation_set_x.get_value(borrow=True)

        # go through this many
        # minibatches before checking the network
        # on the validation set; in this case we
        # check every epoch

        idx_minibatches, minibatches = get_minibatches_idx(n_data,
                                                           batch_size,
                                                           self.numpy_rng)

        n_train_batches = idx_minibatches[-1] + 1

        print('Number of training batches: %d' % n_train_batches)

        for i in range(self.n_layers):
            if graph_output:
                plt.figure(i+1)

            if isinstance(self.rbm_layers[i], GRBM):
                momentum = 0.0
            else:
                momentum = 0.6

            # go through training epochs
            epoch = 0

            minCost = numpy.inf

            while epoch < pretraining_epochs[i]:
                epoch = epoch + 1

                idx_minibatches, minibatches = get_minibatches_idx(n_data,
                                                                   batch_size,
                                                                   self.numpy_rng)

                # go through the training set
                if not isinstance(self.rbm_layers[i], GRBM) and epoch == 6:
                    momentum = 0.9

                costs=[]
                for mb, minibatch in enumerate(minibatches):
                    costs.append(training_fns[i](indexes=minibatch,
                                                   momentum=momentum,
                                                   lr=pretrain_lr[i],
                                                   batch_size=len(minibatch)))

                meanCost = -numpy.mean(costs)
                if verbose:
                    print('Pre-training cost (layer %i, epoch %d): ' % (i, epoch), end=' ')
                    print(meanCost)
                else:
                    print('.',end='')

                if meanCost < minCost:
                    print('\nMinimum cost (layer %i, epoch %d): ' % (i, epoch), end=' ')
                    print(meanCost)
                    minCost = meanCost
                    bestParams = dict()
                    for p in self.rbm_layers[i].params:
                        bestParams[p.name] = p.get_value()

                # Plot the output
                if graph_output:
                    plt.clf()
                    training_output = self.get_output(train_set_x, i)
                    plt.imshow(training_output, cmap='gray')
                    plt.axis('tight')
                    plt.title('epoch %d' % (epoch))
                    plt.draw()
                    plt.pause(1.0)

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

                    print('\nFree energy gap (layer %i, epoch %i): ' % (i, epoch), end=' ')
                    print(free_energy_gap)

            print('\n*** Run %i - Minimum cost %f' % (run, minCost))
            self.rbm_layers[i].W = theano.shared(numpy.asarray(bestParams['W'],dtype=theano.config.floatX))
            self.rbm_layers[i].hbias = theano.shared(numpy.asarray(bestParams['hbias'], dtype=theano.config.floatX))
            self.rbm_layers[i].vbias = theano.shared(numpy.asarray(bestParams['vbias'], dtype=theano.config.floatX))

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
                                                           datadir=datadir)

        return (self.get_output(train_set), self.get_output(validation_set))

    def inspect_inputs(self, i, node, fn):
        '''
        Helper function to inspect inputs of each node. For details see
        http://deeplearning.net/software/theano/tutorial/debug_faq.html
        :param i:
        :param node:
        :param fn:
        :return: None
        '''
        print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs],
              end='\n')

    def inspect_outputs(self, i, node, fn):
        '''
        Helper function to inspect outputs of each node. For details see
        http://deeplearning.net/software/theano/tutorial/debug_faq.html
        :param i:
        :param node:
        :param fn:
        :return: None
        '''
        print(" output(s) value(s):", [output[0] for output in fn.outputs])

def train_MNIST_Gaussian(graph_output=False):
    # Load the data
    mnist = MNIST()
    raw_dataset = mnist.images
    n_data = raw_dataset.shape[0]

    dataset = mnist.normalize(raw_dataset)

    train_set = theano.shared(dataset[0:int(n_data*5/6)], borrow=True)
    validation_set = theano.shared(dataset[-39:], borrow=True)

    batch_size = 20
    k = 1
    layers_sizes = [1000, 500]
    pretraining_epochs = [100, 100]
    pretrain_lr = [0.01, 0.01]
    lambda_1 = 0.0,
    lambda_2 = 0.1

    print('*** Training on MNIST ***')

    print('Visible nodes: %i' % train_set.get_value().shape[1])
    print('Output nodes: %i' % layers_sizes[-1])

    dbn = DBN(n_ins=dataset.shape[1],
                hidden_layers_sizes=layers_sizes[:-1],
                n_outs=layers_sizes[-1])

    dbn.training(train_set,
                 batch_size, k=k,
                 pretraining_epochs=pretraining_epochs,
                 pretrain_lr=pretrain_lr,
                 lambdas=[lambda_1,lambda_2],
                 validation_set_x=validation_set,
                 graph_output=graph_output)

    output_train_set = dbn.get_output(train_set)
    output_val_set = dbn.get_output(validation_set)

    return dbn, output_train_set, output_val_set

if __name__ == '__main__':
    train_MNIST_Gaussian(graph_output=True)