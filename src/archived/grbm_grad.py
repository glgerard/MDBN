import numpy as np
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import scipy.misc
from src.utils import load_MNIST
from src.utils import display_weigths
from src.utils import display_samples
from src.utils import normalize_img

class GRBM(object):
    # Implement a Gaussian-Bernoulli Restricted Boltzmann Machine

    def __init__(self, input, n_input, n_hidden):
        self.input = input
        self.n_input = n_input
        self.n_hidden = n_hidden

        # Rescale terms for visible units
        self.a = theano.shared(value=np.zeros(self.n_input, dtype=theano.config.floatX),
                               borrow=True,
                               name='b')
        # Bias terms for hidden units
        self.b = theano.shared(np.zeros(self.n_hidden,dtype=theano.config.floatX),
                               borrow=True,
                               name='a')

        # Weights
        rng = np.random.RandomState(2468)
        self.W = theano.shared(np.asarray(
                    rng.uniform(
                            -4 * np.sqrt(6. / (self.n_hidden + self.n_input)),
                            4 * np.sqrt(6. / (self.n_hidden + self.n_input)),
                            (self.n_input, self.n_hidden)
                        ),dtype=theano.config.floatX),
                    name='W')
        self.srng = RandomStreams(rng.randint(2 ** 30))

    def v_sample(self, h):
        # Derive a sample of visible units from the hidden units h
        mu = self.a + T.dot(h,self.W.T)
#        v_sample = mu + self.srng.normal(size=mu.shape, avg=0, std=1.0, dtype=theano.config.floatX)
        v_sample = mu # error-free reconstruction
        return [mu, v_sample]

    def h_sample(self, v):
        # Derive a sample of hidden units from the visible units v
        act = self.b + T.dot(v,self.W)
        prob = T.nnet.sigmoid(act)
        return [prob, self.srng.binomial(size=act.shape,n=1,p=prob,dtype=theano.config.floatX)]

    def output(self):
        prob, hS = self.h_sample(self.input)
        return prob

    def gibbs_update(self, h):
        # A Gibbs step
        nv_prob, nv_sample = self.v_sample(h)
        nh_prob, nh_sample = self.h_sample(nv_prob)
        return [nv_prob, nv_sample, nh_prob, nh_sample]

    def alt_gibbs_update(self, v):
        # A Gibbs step
        nh_prob, nh_sample = self.h_sample(v)
        nv_prob, nv_sample = self.v_sample(nh_prob)
        return [nv_prob, nv_sample, nh_prob, nh_sample]

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.b
        vbias_term = 0.5 * T.dot((v_sample - self.a), (v_sample - self.a).T)
        hidden_term = T.sum(T.nnet.softplus(wx_b), axis=1)
        return -hidden_term - vbias_term

    def CD(self, k=1, eps=0.01):
        # Contrastive divergence
        # Positive phase
        h0_prob, h0_sample = self.h_sample(self.input)

        # Negative phase
        ( [ nv_probs,
            nv_samples,
            nh_probs,
            nh_samples],
          updates) = theano.scan(self.gibbs_update,
                                 outputs_info=[None, None, None, h0_sample],
                                 n_steps=k,
                                 name="gibbs_update")

        vK_sample = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(vK_sample))
        params = [self.a, self.b, self.W]

        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, params, consider_constant=[self.input, vK_sample])

        for param, gparam in zip(params, gparams):
            updates[param] = param - gparam * T.cast(eps,dtype=theano.config.floatX)

        dist = T.mean(T.sqr(self.input - vK_sample))
        return dist, updates

def test_grbm(batch_size = 20, training_epochs = 15, k=1, n_hidden=200):

    n_data, n_row, n_col, r_dataset, levels, targets = load_MNIST()
    n_visible = n_row * n_col

    # See https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    # 13.2: "it is [...] easier to first normalise each component of the data to have zero
    #        mean and unit variance and then to use noise free reconstructions, with the variance
    #        in equation 17 set to 1"
    dataset = normalize_img(r_dataset)

    index = T.lscalar('index')
    x = T.matrix('x')
    print("Building an RBM with %i visible inputs and %i hidden units" % (n_visible, n_hidden))
    rbm = GRBM(x, n_visible, n_hidden)

    dist, updates = rbm.CD(k)

    train_set = theano.shared(dataset, borrow=True)

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
    vis_sample = theano.shared(np.asarray(dataset[1000:1010], dtype=theano.config.floatX))
    samples.append(vis_sample.get_value(borrow=True))

    for i in xrange(10):
        ( [ nv_probs,
            nv_samples,
            nh_probs,
            nh_samples],
            updates) = theano.scan(rbm.alt_gibbs_update,
                                   outputs_info=[None, vis_sample, None, None],
                                    n_steps=1000,
                                    name="alt_gibbs_update")

        run_gibbs = theano.function(
                [],
                [   nv_probs[-1],
                    nv_samples[-1]],
                updates=updates,
                mode='NanGuardMode',
                name="run_gibbs"
            )

        nv_prob, nv_sample = run_gibbs()

        samples.append(nv_prob)

    Y = display_samples(samples, n_row, n_col)
    scipy.misc.imsave('mix.png',Y)

if __name__ == '__main__':
    test_grbm(training_epochs=15, k=1)