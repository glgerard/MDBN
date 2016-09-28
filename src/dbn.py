import numpy as np
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from rbm import RBM
from rbm import GRBM
from utils import load_MNIST

class DBN(object):
    def __init__(self, n_input, hidden_layers_sizes, n_output):
        self.n_input = n_input
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_output = n_output

        self.input = T.matrix('input')

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
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        # learning_rate = T.scalar('lr')  # learning rate to use

        # beginning of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        fns = []
        for rbm in self.rbm_layers:
            # using CD-k for training each RBM.
            cost, updates = rbm.CD(k)

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


def test_dbn(batch_size=20, training_epochs=15, k=1):
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
    test_dbn(training_epochs=10)