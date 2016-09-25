import numpy as np
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import struct
import scipy.misc
from utils import load_MNIST
from utils import display_weigths
from utils import display_samples

class RBM(object):
    # Implement a Bernoulli Restricted Boltzmann Machine

    def __init__(self, input, n_input, n_hidden):
        self.input = input
        self.n_visible = n_input
        self.n_input = n_hidden

        # Rescale terms for visible units
        self.b = theano.shared(value=np.zeros(n_input, dtype=theano.config.floatX),
                               borrow=True,
                               name='b')
        # Bias terms for hidden units
        self.a = theano.shared(np.zeros(n_hidden,dtype=theano.config.floatX),
                               borrow=True,
                               name='a')

        # Weights
        rng = np.random.RandomState(2468)
        self.W = theano.shared(np.asarray(
                    rng.uniform(
                            -4 * np.sqrt(6. / (n_hidden + n_input)),
                            4 * np.sqrt(6. / (n_hidden + n_input)),
                            (n_input, n_hidden)
                        ),dtype=theano.config.floatX),
                    name='W')
        self.srng = RandomStreams(rng.randint(2 ** 30))

    def v_sample(self, h):
        # Derive a sample of visible units from the hidden units h
        act = self.b + T.dot(h,self.W.T)
        prob = T.nnet.sigmoid(act)
        return [act, prob, self.srng.binomial(size=act.shape,n=1,p=prob,dtype=theano.config.floatX)]

    def h_sample(self, v):
        # Derive a sample of hidden units from the visible units v
        act = self.a + T.dot(v,self.W)
        prob = T.nnet.sigmoid(act)
        return [act, prob, self.srng.binomial(size=act.shape,n=1,p=prob,dtype=theano.config.floatX)]

    def output(self):
        prob, hS = self.h_sample(self.input)
        return prob

    def gibbs_update(self, h):
        # A Gibbs step
        v_act, nv_prob, nv_sample = self.v_sample(h)
        h_act, nh_prob, nh_sample = self.h_sample(nv_sample)
        return [v_act, nv_prob, nv_sample, h_act, nh_prob, nh_sample]

    def alt_gibbs_update(self, v):
        # A Gibbs step
        h_act, nh_prob, nh_sample = self.h_sample(v)
        v_act, nv_prob, nv_sample = self.v_sample(nh_sample)
        return [v_act, nv_prob, nv_sample, h_act, nh_prob, nh_sample]

    def CD(self, persistent=None, k=1, eps=0.1):
        # Contrastive divergence
        # Positive phase
        h_act, h0_prob, h0_sample = self.h_sample(self.input)

        if persistent is None:
            h_sample = h0_sample
        else:
            h_sample = persistent

        # Negative phase
        ( [ v_acts,
            nv_probs,
            nv_samples,
            h_acts,
            nh_probs,
            nh_samples],
          updates) = theano.scan(self.gibbs_update,
                                 outputs_info=[None, None, None, None, None, h_sample],
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

        w_grad = (T.dot(self.input.T, h0_prob) - T.dot(vK_prob.T, hK_prob))/\
                 T.cast(self.input.shape[0],dtype=theano.config.floatX)

        b_grad = T.mean(self.input - vK_prob, axis=0)

        a_grad = T.mean(h0_prob - hK_prob, axis=0)

        params = [self.a, self.b, self.W]
        gparams = [a_grad, b_grad, w_grad]

        for param, gparam in zip(params, gparams):
            updates[param] = param + gparam * T.cast(eps,dtype=theano.config.floatX)

        dist = T.mean(T.sqr(self.input - vK_prob))
        return dist, updates

def test_rbm(batch_size = 20, training_epochs = 15, k=1, n_hidden=200):

    n_data, n_row, n_col, dataset, levels, targets = load_MNIST()
    n_visible = n_row * n_col

    index = T.lscalar('index')
    x = T.matrix('x')
    print("Building an RBM with %i visible inputs and %i hidden units" % (n_visible, n_hidden))
    rbm = RBM(x, n_visible, n_hidden)

    init_chain = theano.shared(np.zeros((batch_size,n_hidden),dtype=theano.config.floatX))
    dist, updates = rbm.CD(k=k)

    bin_dataset = np.ones(dataset.shape,dtype=theano.config.floatX)
    bin_dataset = bin_dataset * (dataset > 128)

    train_set = theano.shared(bin_dataset, borrow=True)

    train = theano.function(
        [index],
        dist,
        updates=updates,
        givens={
            x: train_set[index*batch_size : (index+1)*batch_size]
        },
        name="train"
    )

    for epoch in xrange(training_epochs):
        dist = []
        for n_batch in xrange(n_data//batch_size):
            dist.append(train(n_batch))

        print("Training epoch %d, mean batch reconstructed distance %f" % (epoch, np.mean(dist)))

        # Construct image from the weight matrix
        Wimg = display_weigths(rbm.W.get_value(borrow=True), n_row, n_col, n_hidden)
        scipy.misc.imsave('filters_at_epoch_%i.png' % epoch,Wimg)

    samples = []
    vis_sample = theano.shared(np.asarray(bin_dataset[1000:1010], dtype=theano.config.floatX))
    samples.append(vis_sample.get_value(borrow=True))

    for i in xrange(10):
        ( [ v_acts,
            nv_probs,
            nv_samples,
            h_acts,
            nh_probs,
            nh_samples],
            updates) = theano.scan(rbm.alt_gibbs_update,
                                   outputs_info=[None, None, vis_sample, None, None, None],
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

    Y = display_samples(samples, n_row, n_col)
    scipy.misc.imsave('mix.png',Y)

if __name__ == '__main__':
    test_rbm(training_epochs=15, k=15)