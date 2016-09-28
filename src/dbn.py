import numpy as np
import theano
from theano import tensor as T
import scipy.misc
from rbm import RBM
from rbm import GRBM
from utils import load_MNIST
from utils import display_weigths
from utils import display_samples
from utils import normalize

class DBN(object):
    def __init__(self, n_input, hidden_layers_sizes, n_output):
        self.n_input = n_input
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_output = n_output

        self.input = T.matrix('x')

        self.rbm_layers = []

        self.rbm_layers.append(GRBM(self.input, self.n_input, self.hidden_layers_sizes[0]))

        for (l, size) in enumerate(self.hidden_layers_sizes[:-1]):
            self.rbm_layers.append(RBM(self.rbm_layers[-1].output(),
                                       size,
                                       self.hidden_layers_sizes[l+1]))

        self.rbm_layers.append(RBM(self.rbm_layers[-1].output(),
                                   self.hidden_layers_sizes[-1],
                                   self.n_output))

    def training_functions(self, training_set, batch_size, k):
        # index to a mini-batch
        index = T.lscalar('index')

        fns = []
        for rbm in self.rbm_layers:
            # using CD-k for training each RBM.
            dist, updates = rbm.CD(k)

            # compile the theano function
            fn = theano.function(
                [index],
                dist,
                updates=updates,
                givens={
                    self.input: training_set[index * batch_size:(index + 1)* batch_size]
                }
            )
            # append `fn` to the list of functions
            fns.append(fn)

        return fns


def test_dbn(batch_size=20, training_epochs=15, k=1):
    n_data, n_row, n_col, r_dataset, levels, targets = load_MNIST()
    n_visible = n_row * n_col

    print("Building a DBN with %i visible inputs and %i output units" % (n_visible, levels))
    print("This DBN has 2 hidden layers of size 200 and 50 each")

    dataset = normalize(r_dataset)
    train_set = theano.shared(dataset, borrow=True)

    # construct the Deep Belief Network
    dbn = DBN(n_input=n_visible,
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
    test_dbn(training_epochs=15,k=1)