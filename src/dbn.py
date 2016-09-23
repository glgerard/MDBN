import numpy as np
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from grbm import GRBM
from grbm import load_MNIST

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
        activation = self.b + T.dot(h,self.W.T)
        prob = T.nnet.sigmoid(activation)
        return [prob, self.srng.binomial(size=activation.shape,n=1,p=prob,dtype=theano.config.floatX)]

    def h_sample(self, v):
        # Derive a sample of hidden units from the visible units v
        activation = self.a + T.dot(v,self.W)
        prob = T.nnet.sigmoid(activation)
        return [prob, self.srng.binomial(size=activation.shape,n=1,p=prob,dtype=theano.config.floatX)]

    def output(self):
        prob, hS = self.h_sample(self.input)
        return prob

    def gibbs_update(self, h):
        # A Gibbs step
        nv_prob, nv_sample = self.v_sample(h)
        nh_prob, nh_sample = self.h_sample(nv_sample)
        return [nv_prob, nv_sample, nh_prob, nh_sample]

    def alt_gibbs_update(self, v):
        # A Gibbs step
        nh_prob, nh_sample = self.h_sample(v)
        nv_prob, nv_sample = self.v_sample(nh_sample)
        return [nv_prob, nv_sample, nh_prob, nh_sample]

    def CD(self, k=1, eps=0.1):
        # Contrastive divergence
        # Positive phase
        h0_prob, h0_sample = self.h_sample(self.input)

        # Negative phase
        ( [ nv_probs,
            nv_samples,
            nh_probs,
            nh_samples],
          updates) = theano.scan(self.gibbs_update,outputs_info=[None, h0_sample],n_steps=k,name="gibbs_update")

        vK_prob = nv_probs[-1]
        vK_sample = nv_samples[-1]
        hK_prob = nh_probs[-1]
        hK_sample = nh_samples[-1]

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

        dist = T.sum(T.sqr(self.input - vK_prob))
        return dist, updates

class DBN(object):
    def __init__(self, n_input, hidden_layers_sizes, n_output):
        self.n_input = n_input
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_output = n_output

        self.input = T.matrix('input')

        self.rbm_layers = []

        self.rbm_layers.append(GRBM(self.input, self.n_input, self.hidden_layers_sizes[0]))

        for (l, size) in enumerate(self.hidden_layers_sizes):
            self.rbm_layers.append(RBM(self.rbm_layers[-1].output(),
                                       size,
                                       self.hidden_layers_sizes[l+1]))

        self.rbm_layers.append(RBM(self.rbm_layers[-1].output(),
                                   self.hidden_layers_sizes[-1],
                                   self.n_output))

    def training_functions(self, training_set, batch_size, k):
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # beginning of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        fns = []
        for rbm in self.rbm_layers:
            # using CD-k for training each RBM.
            cost, updates = rbm.CD(k, learning_rate)

            # compile the theano function
            fn = theano.function(
                inputs=[index],
                outputs=cost,
                updates=updates,
                givens={
                    self.input: training_set[batch_begin:batch_end]
                }
            )
            # append `fn` to the list of functions
            fns.append(fn)

        return fns


def test(batch_size=20, training_epochs=15, k=1):
    n_data, input_size, dataset, levels, targets = load_MNIST()

    print("Building a DBN with %i visible inputs and %i output units" % (input_size, levels))
    print("This DBN has 2 hidden layers of size 200 and 50 each")

    train_set = theano.shared(dataset, borrow=True)

    # construct the Deep Belief Network
    dbn = DBN(n_input=input_size,
              hidden_layers_sizes=[200, 50],
              n_output=levels)

    training_fns = dbn.training_functions(training_set=train_set,
                                          batch_size=batch_size,
                                          k=k)

    for i in xrange(len(dbn.rbm_layers)):
        # go through training epochs
        for epoch in xrange(training_epochs):
            # go through the training set
            c = []
            for n_batch in xrange(n_data // batch_size):
                c.append(training_fns[i](index=n_batch))
            print("Training epoch %d, mean batch reconstructed distance %f" % (epoch, np.mean(c)))

if __name__ == '__main__':
    test(training_epochs=10)