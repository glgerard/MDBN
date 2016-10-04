import numpy as np
import theano
from theano import tensor
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import scipy.misc
from MNIST import MNIST
from utils import zscore

class RBM(object):
    # Implement a Bernoulli Restricted Boltzmann Machine

    def __init__(self, input, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.input = input

        # Rescale terms for visible units
        self.a = theano.shared(value=np.zeros(n_visible, dtype=theano.config.floatX),
                               borrow=True,
                               name='a')
        # Bias terms for hidden units
        self.b = theano.shared(np.zeros(n_hidden,dtype=theano.config.floatX),
                               borrow=True,
                               name='b')

        # Weights
        rng = np.random.RandomState(2468)
        self.W = theano.shared(np.asarray(
                    rng.uniform(
                            -4 * np.sqrt(6. / (n_hidden + n_visible)),
                            4 * np.sqrt(6. / (n_hidden + n_visible)),
                            (n_visible, n_hidden)
                        ),dtype=theano.config.floatX),
                    borrow=True,
                    name='W')
        self.Wt = self.W.T
        self.srng = RandomStreams(rng.randint(2 ** 30))

    def v_sample(self, h):
        # Derive a sample of visible units from the hidden units h
        act = self.a + tensor.tensordot(h,self.Wt,axes=[1,0])
        prob = tensor.nnet.sigmoid(act)
        return [prob, self.srng.binomial(size=act.shape,n=1,p=prob,dtype=theano.config.floatX)]

    def h_sample(self, v):
        # Derive a sample of hidden units from the visible units v
        act = self.b + tensor.dot(v,self.W)
        prob = tensor.nnet.sigmoid(act)
        return [prob, self.srng.binomial(size=act.shape,n=1,p=prob,dtype=theano.config.floatX)]

    def output(self):
        prob, hS = self.h_sample(self.input)
        return prob

    def gibbs_step_hvh(self, h):
        # A Gibbs step
        nv_prob, nv_sample = self.v_sample(h)
        nh_prob, nh_sample = self.h_sample(nv_sample)
        return [nv_prob, nv_sample, nh_prob, nh_sample]

    def gibbs_step_hvhp(self, hp):
        # A Gibbs step
        nv_prob, nv_sample = self.v_sample(hp)
        nh_prob, nh_sample = self.h_sample(nv_prob)
        return [nv_prob, nv_sample, nh_prob, nh_sample]

    def gibbs_step_vhv(self, v):
        # A Gibbs step
        nh_prob, nh_sample = self.h_sample(v)
        nv_prob, nv_sample = self.v_sample(nh_sample)
        return [nv_prob, nv_sample, nh_prob, nh_sample]

    def contrastive_divergence(self, k=1, lr=0.01, lambda1=0, lambda2=0,
                               persistent=None, stocastic_steps=True):
        # Contrastive divergence
        # Positive phase
        h0_prob, h0_sample = self.h_sample(self.input)

        if persistent is None:
            h_sample = h0_sample
        else:
            h_sample = persistent

        self.Wt = self.W.T

        # Negative phase
        if stocastic_steps:
            ( [
                nv_probs,
                nv_samples,
                nh_probs,
                nh_samples],
              updates) = theano.scan(self.gibbs_step_hvh,
                                     outputs_info=[None, None, None, h_sample],
                                     n_steps=k,
                                     name="gibbs_update")
        else:
            ([
                 nv_probs,
                 nv_samples,
                 nh_probs,
                 nh_samples],
             updates) = theano.scan(self.gibbs_step_hvhp,
                                    outputs_info=[None, None, h0_prob, None],
                                    n_steps=k,
                                    name="gibbs_update")

        vK_prob = nv_probs[-1]
        vK_sample = nv_samples[-1]
        hK_prob = nh_probs[-1]
        hK_sample = nh_samples[-1]

        if persistent:
            updates[persistent] = hK_sample

        # See https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
        # I keep sigma unit as reported in https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf 13.2

        W_grad = (tensor.dot(self.input.T, h0_prob) - tensor.dot(vK_prob.T, hK_prob))/\
                 tensor.cast(self.input.shape[0],dtype=theano.config.floatX)

        a_grad = tensor.mean(self.input - vK_prob, axis=0)

        b_grad = tensor.mean(h0_prob - hK_prob, axis=0)

        params = [self.a, self.b]
        gparams = [a_grad, b_grad]
        eps = tensor.cast(lr, dtype=theano.config.floatX)

        for param, gparam in zip(params, gparams):
            updates[param] = param + gparam * eps
        if lambda1+lambda2 == 0:
            updates[self.W] = self.W + W_grad * eps
        else:
            # Used in M. Liang et al. 2015
            l1 = tensor.cast(1-2*lambda1*lr, dtype=theano.config.floatX)
            l2 = tensor.cast(2*lambda2*lr, dtype=theano.config.floatX)
            updates[self.W] = (l1 * self.W + W_grad * eps ) /\
                              (1 + l2/tensor.abs_(self.W))

        if stocastic_steps:
            sme = tensor.mean(tensor.sum((self.input - vK_sample)**2,axis=1))
        else:
            sme = tensor.mean(tensor.sum((self.input - vK_prob)**2,axis=1))

        return sme, updates

    def training(self, dataset, batch_size, training_epochs, k, lr,
                 lambda1=0, lambda2=0,
                 stocastic_steps=True,
                 display_fn=None):
        index = tensor.lscalar('index')
        train_set = theano.shared(dataset, borrow=True)

        sme, updates = self.contrastive_divergence(k=k, lr=lr,
                                                   lambda1=lambda1,
                                                   lambda2=lambda2,
                                                   stocastic_steps=stocastic_steps)

        train = theano.function(
            [index],
            sme,
            updates=updates,
            givens={
                self.input: train_set[index * batch_size: (index + 1) * batch_size]
            },
            name="train"
        )

        n_data = dataset.shape[0]
        for epoch in xrange(training_epochs):
            sme_list = []
            for n_batch in xrange(n_data // batch_size):
                sme_list.append(train(n_batch))

            print("Training epoch %d, reconstruction error %f" % (epoch, sme_list[-1]))

            if display_fn is not None:
                # Construct image from the weight matrix
                Wimg = display_fn(self.W.get_value(borrow=True), self.n_hidden)
                scipy.misc.imsave('filters_at_epoch_%i.png' % epoch, Wimg)

class GRBM(RBM):
    # Implement a Gaussian-Bernoulli Restricted Boltzmann Machine
    def __init__(self, input, n_visible, n_hidden):
        super(GRBM, self).__init__(input, n_visible, n_hidden)

    def v_sample(self, h):
    # Derive a sample of visible units from the hidden units h
        mu = self.a + tensor.tensordot(h, self.Wt, axes=[1,0])
#       v_sample = mu + self.srng.normal(size=mu.shape, avg=0, std=1.0, dtype=theano.config.floatX)
        v_sample = mu  # error-free reconstruction
        return [mu, v_sample]

def test_rbm(batch_size = 20, training_epochs = 15, k=1, n_hidden=200, binary=True):
    # Load the data
    mnist = MNIST('../data/train-images-idx3-ubyte')
    raw_dataset = mnist.images
    n_visible = mnist.sizeX * mnist.sizeY

    print("Building an RBM with %i visible inputs and %i hidden units" % (n_visible, n_hidden))

#    init_chain = theano.shared(np.zeros((batch_size,n_hidden),dtype=theano.config.floatX))

    # Build the RBM
    x = tensor.matrix('x')

    if binary == True:
        rbm = RBM(x, n_visible, n_hidden)
        dataset = np.ones(raw_dataset.shape,dtype=theano.config.floatX)
        zscore(raw_dataset)
        dataset = dataset * (raw_dataset > 0)
        lr = 0.01
    else:
        rbm = GRBM(x, n_visible, n_hidden)
        dataset = mnist.normalize(raw_dataset)
        lr = 0.01

    # Train the RBM
    rbm.training(dataset, batch_size, training_epochs, k, lr, mnist.display_weigths)

    # Test the model we have learned
    samples = []
    vis_sample = theano.shared(np.asarray(dataset[1000:1010], dtype=theano.config.floatX))
    samples.append(vis_sample.get_value(borrow=True))

    rbm.Wt = rbm.W.T
    for i in xrange(10):
        ( [
            nv_probs,
            nv_samples,
            nh_probs,
            nh_samples],
            updates) = theano.scan(rbm.gibbs_step_vhv,
                                   outputs_info=[None, vis_sample, None, None],
                                   n_steps=1000,
                                   name="alt_gibbs_update")

        run_gibbs = theano.function(
                [],
                [   nv_probs[-1],
                    nv_samples[-1]],
                updates=updates,
                name="run_gibbs"
            )

        nv_prob, nv_sample = run_gibbs()

        samples.append(nv_prob)

    Y = mnist.display_samples(samples)
    scipy.misc.imsave('mix.png',Y)

if __name__ == '__main__':
    test_rbm(training_epochs=15, k=1, binary=False)