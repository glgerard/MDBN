import numpy as np
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import scipy.misc
from utils import load_MNIST
from utils import display_weigths
from utils import display_samples
from utils import normalize

class RBM(object):
    # Implement a Bernoulli Restricted Boltzmann Machine

    def __init__(self, input, n_input, n_hidden):
        self.input = input
        self.n_visible = n_input
        self.n_input = n_hidden

        # Rescale terms for visible units
        self.a = theano.shared(value=np.zeros(n_input, dtype=theano.config.floatX),
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
                            -4 * np.sqrt(6. / (n_hidden + n_input)),
                            4 * np.sqrt(6. / (n_hidden + n_input)),
                            (n_input, n_hidden)
                        ),dtype=theano.config.floatX),
                    name='W')
        self.srng = RandomStreams(rng.randint(2 ** 30))

    def v_sample(self, h):
        # Derive a sample of visible units from the hidden units h
        act = self.a + T.dot(h,self.W.T)
        prob = T.nnet.sigmoid(act)
        return [prob, self.srng.binomial(size=act.shape,n=1,p=prob,dtype=theano.config.floatX)]

    def h_sample(self, v):
        # Derive a sample of hidden units from the visible units v
        act = self.b + T.dot(v,self.W)
        prob = T.nnet.sigmoid(act)
        return [prob, self.srng.binomial(size=act.shape,n=1,p=prob,dtype=theano.config.floatX)]

    def output(self):
        prob, hS = self.h_sample(self.input)
        return prob

    def gibbs_step_hvh(self, h):
        # A Gibbs step
        nv_prob, nv_sample = self.v_sample(h)
        nh_prob, nh_sample = self.h_sample(nv_sample)
        return [nv_prob, nv_sample, nh_prob, nh_sample]

    def gibbs_step_vhv(self, v):
        # A Gibbs step
        nh_prob, nh_sample = self.h_sample(v)
        nv_prob, nv_sample = self.v_sample(nh_sample)
        return [nv_prob, nv_sample, nh_prob, nh_sample]

    def CD(self, persistent=None, k=1, eps=0.01):
        # Contrastive divergence
        # Positive phase
        h0_prob, h0_sample = self.h_sample(self.input)

        if persistent is None:
            h_sample = h0_sample
        else:
            h_sample = persistent

        # Negative phase
        ( [
            nv_probs,
            nv_samples,
            nh_probs,
            nh_samples],
          updates) = theano.scan(self.gibbs_step_hvh,
                                 outputs_info=[None, None, None, h_sample],
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

        a_grad = T.mean(self.input - vK_prob, axis=0)

        b_grad = T.mean(h0_prob - hK_prob, axis=0)

        params = [self.a, self.b, self.W]
        gparams = [a_grad, b_grad, w_grad]

        for param, gparam in zip(params, gparams):
            updates[param] = param + gparam * T.cast(eps,dtype=theano.config.floatX)

        dist = T.mean(T.sqr(self.input - vK_sample))
        return dist, updates

class GRBM(RBM):
    # Implement a Gaussian-Bernoulli Restricted Boltzmann Machine
    def __init__(self,input, n_input, n_hidden):
        super(GRBM, self).__init__(input, n_input, n_hidden)

    def v_sample(self, h):
    # Derive a sample of visible units from the hidden units h
        mu = self.a + T.dot(h, self.W.T)
#       v_sample = mu + self.srng.normal(size=mu.shape, avg=0, std=1.0, dtype=theano.config.floatX)
        v_sample = mu  # error-free reconstruction
        return [mu, v_sample]

def test_rbm(batch_size = 20, training_epochs = 15, k=1, n_hidden=200, binary=True):

    n_data, n_row, n_col, r_dataset, levels, targets = load_MNIST()
    n_visible = n_row * n_col

    index = T.lscalar('index')
    x = T.matrix('x')
    print("Building an RBM with %i visible inputs and %i hidden units" % (n_visible, n_hidden))

    init_chain = theano.shared(np.zeros((batch_size,n_hidden),dtype=theano.config.floatX))

    n_dataset = normalize(r_dataset)

    if binary == True:
        rbm = RBM(x, n_visible, n_hidden)
        dataset = np.ones(r_dataset.shape,dtype=theano.config.floatX)
        r_dataset = r_dataset-np.mean(r_dataset,axis=1,keepdims=True)
        r_dataset = r_dataset / np.std(r_dataset,axis=1,keepdims=True)
        dataset = dataset * (r_dataset > 0)
        lr = 0.01
    else:
        rbm = GRBM(x, n_visible, n_hidden)
        dataset = normalize(r_dataset)
        lr = 0.01

    dist, updates = rbm.CD(k=k, eps=lr)

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

    Y = display_samples(samples, n_row, n_col)
    scipy.misc.imsave('mix.png',Y)

if __name__ == '__main__':
    test_rbm(training_epochs=15, k=1, binary=False)