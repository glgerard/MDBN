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
import os

import numpy
import theano
from theano import tensor
from theano.tensor import nnet
from theano.compile.nanguardmode import NanGuardMode

#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import scipy.misc
from MNIST import MNIST
from utils import get_minibatches_idx

class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    """Initial version from http://deeplearning.net/tutorial/code/rbm.py """
    def __init__(
        self,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        hbias=None,
        vbias=None,
        numpy_rng=None,
        theano_rng=None
    ):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = tensor.matrix('input')

        self.W = W
        self.Wt = W.T
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]

        # Parameters to implement momentum
        # See: Hinton, "A Practical Guide to Training Restricted Boltzmann Machines",
        # UTML TR 2010-003, 2010. Section 9

        self.momentum = tensor.cast(0, dtype=theano.config.floatX)

        self.W_speed = theano.shared(
            numpy.zeros((n_visible, n_hidden), dtype=theano.config.floatX),
            name='W_speed',
            borrow=True)
        self.hbias_speed = theano.shared(numpy.zeros(n_hidden, dtype=theano.config.floatX),
                                        name='hbias_speed',
                                        borrow=True)
        self.vbias_speed = theano.shared(numpy.zeros(n_visible, dtype=theano.config.floatX),
                                        name='vbias_speed',
                                        borrow=True)

        self.params_speed = [self.W_speed, self.hbias_speed, self.vbias_speed]

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = tensor.dot(v_sample, self.W) + self.hbias
        vbias_term = tensor.dot(v_sample, self.vbias)
        hidden_term = tensor.sum(nnet.softplus(wx_b), axis=1)
        return -hidden_term - vbias_term

    def free_energy_gap(self, train, validation):
        """ Computes the free energy gap between train and test set, F(x_test) - F(x_train).

        See: Hinton, "A Practical Guide to Training Restricted Boltzmann Machines", UTML TR 2010-003, 2010, section 6.

        Originally from: https://github.com/wuaalb/keras_extensions/blob/master/keras_extensions/rbm.py
        """
        return tensor.mean(self.free_energy(train)) - tensor.mean(self.free_energy(validation))

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = tensor.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = tensor.dot(hid, self.Wt) + self.vbias
        return [pre_sigmoid_activation, nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, k=1,
                         lambda_1=0.0, lambda_2=0.0,
                         weightcost = 0.0,
                         batch_size=None, persistent=None):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing archived state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        :param lambda_1: gradual shrinkage (?)

        :param lambda_2: gradual shrinkage (?)

        :param weightcost: L2 weight-decay (see Hinton 2010
            "A Practical Guide to Training Restricted Boltzmann
            Machines" section 10

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        self.Wt = self.W.T
        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the archived state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k,
            name="gibbs_hvh"
        )
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        if batch_size is not None:
            W_grad =  (tensor.dot(self.input.T, ph_mean) -
                        tensor.dot(nv_means[-1].T, nh_means[-1]))/\
                        tensor.cast(batch_size,dtype=theano.config.floatX) - \
                        tensor.cast(weightcost, dtype=theano.config.floatX) * \
                            self.W.get_value(borrow=True)

            hbias_grad = tensor.mean(ph_mean - nh_means[-1], axis=0)

            vbias_grad = tensor.mean(self.input - nv_means[-1], axis=0)

            gradients = [W_grad, hbias_grad, vbias_grad ]
        else:
            cost = tensor.mean(self.free_energy(chain_end)) - \
                   tensor.mean(self.free_energy(self.input))

            # We must not compute the gradient through the gibbs sampling
            gradients = tensor.grad(cost, self.params, consider_constant=[chain_end])

# ISSUE: it returns Inf when Wij is small
#        gparams[0] = gparams[0] / (1 + 2 * tensor.cast(lr * lambda_1, dtype=theano.config.floatX) / \
#                                    tensor.abs_(self.W))

        # constructs the update dictionary
        multipliers = [
            (1 - 2 * lr * lambda_2),
            # Issue: it returns Inf when Wij is small
            #           (1 - 2 * tensor.cast(lr * lambda_2, dtype=theano.config.floatX)) / \
            #           (1 + 2 * tensor.cast(lr * lambda_1, dtype=theano.config.floatX) / \
            #            tensor.abs_(self.W)),
            1,1]

        for gradient, param, multiplier, param_speed in zip(gradients, self.params,
                                                        multipliers, self.params_speed):
            # make sure that the learning rate is of the right dtype
            # update rules as in https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/models/rbm.py

            updates[param_speed] = gradient + (param_speed - gradient) * \
                                   tensor.cast(self.momentum, dtype=theano.config.floatX)

            updates[param] = param * tensor.cast(multiplier, dtype=theano.config.floatX) + \
                             param_speed * tensor.cast(lr, dtype=theano.config.floatX)

        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = tensor.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = tensor.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = - tensor.mean(self.n_visible * nnet.softplus(fe_xi - fe_xi_flip))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """

        cross_entropy = nnet.binary_crossentropy(
                            nnet.sigmoid(pre_sigmoid_nv),self.input).sum(axis=1).mean()

        return cross_entropy

    def training(self, train_set_x, training_epochs, batch_size=10,
                 learning_rate=0.1, k=1,
                 initial_momentum = 0.0, final_momentum = 0.0,
                 weightcost = 0.0, display_fn=None,
                 persistent=True):

        if persistent:
            # initialize storage for the persistent chain (state = hidden
            # layer of chain)
            persistent_chain = theano.shared(numpy.zeros((batch_size, self.n_hidden),
                                                         dtype=theano.config.floatX),
                                             borrow=True)
        else:
            persistent_chain = None

        # get the cost and the gradient corresponding to one step of CD-15

        cost, updates = self.get_cost_updates(lr=learning_rate,
                                              k=k,
                                              weightcost=weightcost,
                                              batch_size=batch_size,
                                              persistent=persistent_chain
                                              )

        self.learn_model(train_set_x, training_epochs, batch_size,
                   initial_momentum, final_momentum,
                   cost, updates,
                   display_fn)

    def learn_model(self, train_set_x, training_epochs, batch_size,
                    initial_momentum, final_momentum,
                    cost, updates,
                    display_fn):
        # allocate symbolic variables for the data
        indexes = tensor.lvector('indexes')  # index to a [mini]batch
        momentum = tensor.scalar('momentum', dtype=theano.config.floatX)

        # it is ok for a theano function to have no output
        # the purpose of train_rbm is solely to update the RBM parameters
        train_rbm = theano.function(
            [indexes, momentum],
            cost,
            updates=updates,
            givens={
                self.input: train_set_x[indexes],
                self.momentum: momentum
            },
            name='train_rbm'
# TODO: NanGuardMode should be selected with a flag
#            ,mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
        )

        # compute number of minibatches for training, validation and testing
        n_train_data = train_set_x.get_value(borrow=True).shape[0]

        plotting_time = 0.
        start_time = timeit.default_timer()

        # go through training epochs
        momentum = initial_momentum
        for epoch in range(training_epochs):

            if epoch == 6:
                momentum = final_momentum

            _, minibatches = get_minibatches_idx(n_train_data,
                                                 batch_size,
                                                 shuffle=True)

            # go through the training set
            mean_cost = []

            for batch_indexes in minibatches:
                mean_cost += [train_rbm(batch_indexes, momentum)]

            print('Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost))

            # Plot filters after each training epoch
            plotting_start = timeit.default_timer()
            if display_fn is not None:
                # Construct image from the weight matrix
                Wimg = display_fn(self.W.get_value(borrow=True), self.n_hidden)
                scipy.misc.imsave('filters_at_epoch_%i.png' % epoch, Wimg)
            plotting_stop = timeit.default_timer()
            plotting_time += (plotting_stop - plotting_start)

        end_time = timeit.default_timer()

        pretraining_time = (end_time - start_time) - plotting_time

        print ('Training took %f minutes' % (pretraining_time / 60.))

class GRBM(RBM):
    # Implement a Gaussian-Bernoulli Restricted Boltzmann Machine
    def __init__(self,
                 input=None,
                 n_visible=784,
                 n_hidden=500,
                 W=None,
                 hbias=None,
                 vbias=None,
                 numpy_rng=None,
                 theano_rng=None,
                 error_free=True):
        super(GRBM, self).__init__(input, n_visible, n_hidden,
                                   W, hbias, vbias, numpy_rng, theano_rng)
        self.error_free = error_free

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        v1_mean = tensor.dot(h0_sample, self.Wt) + self.vbias

        if self.error_free:
            v1_sample = v1_mean
        else:
            # get a sample of the visible given their activation
            v1_sample = v1_mean + self.theano_rng.normal(size=v1_mean.shape,
                                               avg=0, std=1.0,
                                               dtype=theano.config.floatX)

        return [v1_mean, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state.
            For Gaussian Bernoulli we uses a mean field approximation
            of the intermediate visible state.
        '''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_mean)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state.
            For Gaussian Bernoulli we uses a mean field approximation
            of the intermediate hidden state.
        '''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_mean)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def free_energy(self, v_sample):
        wx_b = tensor.dot(v_sample, self.W) + self.hbias
        vbias_term = 0.5*tensor.sqr(v_sample - self.vbias).sum(axis=1)
        hidden_term = nnet.softplus(wx_b).sum(axis=1)
        return -hidden_term + vbias_term

    def get_reconstruction_cost(self, pre_sigmoid_nv):
        """ Compute mean squared error between reconstructed data and input data.

            Mean over the samples and features.

        """

        error = tensor.sqr(nnet.sigmoid(pre_sigmoid_nv) - self.input).mean()

        return error

    def training(self, train_set_x, training_epochs, batch_size=10,
                 learning_rate=0.01, k=1,
                 initial_momentum = 0.0, final_momentum = 0.0,
                 weightcost = 0.0, display_fn=None,
                 lambda_2 = 0.1):

        cost, updates = self.get_cost_updates(lr=learning_rate,
                                              k=k,
                                              lambda_2=lambda_2,
                                              weightcost=weightcost,
                                              batch_size=batch_size
                                              )

        self.learn_model(train_set_x, training_epochs, batch_size,
                   initial_momentum, final_momentum,
                   cost, updates,
                   display_fn)

def test(class_to_test=RBM,
         learning_rate=0.1,
         training_epochs=15,
         batch_size=20,
         n_chains=20,
         n_samples=10,
         output_folder='rbm_plots',
         n_hidden=500):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param datafile: path to the dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain

    """
    # Load the data
    mnist = MNIST()
    raw_dataset = mnist.images
    n_data = raw_dataset.shape[0]

    if class_to_test == GRBM:
        dataset = mnist.normalize(raw_dataset)
        # Gaussian RBM needs a lower learning rate. See Hinton'10
        learning_rate = learning_rate / 10
    else:
        dataset = raw_dataset/255

    train_set_x = theano.shared(dataset[0:n_data*5/6], borrow=True)
    test_set_x = theano.shared(dataset[n_data*5/6:n_data], borrow=True)

    # find out the number of test samples
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    print('Number of test samples %d' % number_of_test_samples)

    x = tensor.matrix('x')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # construct the RBM class
    rbm = class_to_test(input=x, n_visible=mnist.sizeX * mnist.sizeY,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    rbm.training(train_set_x,
                 training_epochs,
                 batch_size,
                 learning_rate,
                 initial_momentum=0.6, final_momentum=0.9,
                 weightcost=0.0002,
                 display_fn=mnist.display_weigths)

    #################################
    #     Sampling from the RBM     #
    #################################

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        numpy.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )
    plot_every = 500
    # define one step of Gibbs sampling define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every,
        name="gibbs_vhv"
    )

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    samples = []
    for idx in range(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        print(' ... computing sample %d' % idx)
        vis_mf, vis_sample = sample_fn()
        samples.append(vis_mf)

    # construct image
    Y = mnist.display_samples(samples)
    scipy.misc.imsave('samples.png', Y)

    os.chdir('../')

if __name__ == '__main__':
    test(class_to_test=GRBM, training_epochs=5)