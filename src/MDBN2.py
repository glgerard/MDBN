from __future__ import print_function, division

import timeit
import sys
import os

import numpy
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import zscore
from rbm2 import RBM
from rbm2 import GRBM

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
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

        lin_output = T.dot(input, self.W) + self.b
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

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
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
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

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
                layer_input = self.x
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
            if i==0:
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

        self.output = self.sigmoid_layers[-1].output

    def pretraining_functions(self, train_set_x, batch_size, k):
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
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.In(learning_rate, value=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def pretraining(self, train_set_x, batch_size, k, pretraining_epochs, pretrain_lr):
        #########################
        # PRETRAINING THE MODEL #
        #########################

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

        print('... getting the pretraining functions')
        pretraining_fns = self.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=batch_size,
                                                    k=k)

        print('... pre-training the model')
        start_time = timeit.default_timer()
        # Pre-train layer-wise
        for i in range(self.n_layers):
            # go through pretraining epochs
            for epoch in range(pretraining_epochs[i]):
                # go through the training set
                c = []
                for batch_index in range(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                                                lr=pretrain_lr[i]))
                print('Pre-training layer %i, epoch %d, cost ' % (i, epoch), end=' ')
                print(numpy.mean(c))

        end_time = timeit.default_timer()

        print('The pretraining code for file ' + os.path.split(__file__)[1] +
              ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)

def importdata(file):
    with open(file) as f:
        ncols = len(f.readline().split('\t'))

    return (ncols-1,
            np.loadtxt(file,
                       dtype=theano.config.floatX,
                       delimiter='\t',
                       skiprows=1,
                       usecols=range(1,ncols)))

def test_MDBN(batch_size=1,
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
    datage = zscore(GE.T)
    datame = zscore(ME.T)
    datarna = zscore(mRNA.T)

    x = tensor.matrix('x')

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    rna_DBN = DBN(numpy_rng=rng, n_ins=datarna.shape[1],
              hidden_layers_sizes=[],
              n_outs=40)
    rna_DBN.pretraining(datarna, batch_size, k=1,
                        pretraining_epochs=[8000],
                        pretrain_lr=[0.0005])

    ge_DBN = DBN(numpy_rng=rng, n_ins=datage.shape[1],
              hidden_layers_sizes=[400],
              n_outs=40)
    ge_DBN.pretraining(datage, batch_size, k=1,
                       pretraining_epochs=[8000, 800],
                       pretrain_lr=[0.0005, 0.1])

    me_DBN = DBN(numpy_rng=rng, n_ins=datame.shape[1],
              hidden_layers_sizes=[400],
              n_outs=40)
    me_DBN.pretraining(datame, batch_size, k=1,
                       pretraining_epochs=[8000, 800],
                       pretrain_lr=[0.0005, 0.1])

    joint_layer = []
    joint_layer.append(rna_DBN.output)
    joint_layer.append(ge_DBN.output())
    joint_layer.append(me_DBN.output())

    np.savez('parameters_at_gaussian_layer_RNA.npz',
             k=20,
             epoch=8000,
             batch_size=10,
             learning_rate=0.0005,
             stocastic_steps=False,
             momentum=False,
             weight_cost=False,
             W=rnaGRBM.W.get_value(borrow=True),
             a=rnaGRBM.a.get_value(borrow=True),
             b=rnaGRBM.b.get_value(borrow=True))

if __name__ == '__main__':
    test_MDBN()
