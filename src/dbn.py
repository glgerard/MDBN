import numpy as np
import theano
from theano import pp, tensor
import scipy.misc
from rbm import RBM
from rbm import GRBM
from utils import load_MNIST
from utils import display_weigths
from utils import display_samples
from utils import normalize_img

class DBN(object):
    def __init__(self, n_input, hidden_layers_sizes, n_output):
        self.n_input = n_input
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_output = n_output
        self.x = tensor.matrix('x')

        self.rbm_layers = []

        self.rbm_layers.append(GRBM(self.x, n_input,self.hidden_layers_sizes[0]))

        for (l, size) in enumerate(self.hidden_layers_sizes[:-1]):
            input = self.rbm_layers[-1].output()
            self.rbm_layers.append(RBM(
                input,
                size,
                self.hidden_layers_sizes[l+1]))

        input = self.rbm_layers[-1].output()
        self.rbm_layers.append(RBM(
                input,
                self.hidden_layers_sizes[-1],
                self.n_output))

    def training_functions(self, training_set, batch_size, k):
        # index to a mini-batch
        index = tensor.lscalar('index')

        fns = []
        for rbm in self.rbm_layers:
            # using CD-k for training each RBM.
            dist, updates = rbm.CD(k=k)

            # compile the theano function
            fn = theano.function(
                [index],
                dist,
                updates=updates,
                givens={
                    self.x: training_set[index * batch_size:(index + 1)* batch_size]
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

    dataset = normalize_img(r_dataset)
    train_set = theano.shared(dataset, borrow=True)

    # construct the Deep Belief Network
    dbn = DBN(n_input=n_visible,
              hidden_layers_sizes=[196, 64],
              n_output=16)

    training_fns = dbn.training_functions(training_set=train_set,
                                          batch_size=batch_size,
                                          k=k)

    for layer in xrange(len(dbn.rbm_layers)):
        # go through training epochs
        for epoch in xrange(training_epochs):
            # go through the training set
            c = []
            for n_batch in xrange(n_data // batch_size):
                c.append(training_fns[layer](index=n_batch))
            print("Training epoch %d, mean batch reconstructed distance %f" % (epoch, np.mean(c)))
        # Construct image from the weight matrix
        current_layer = dbn.rbm_layers[layer];
        n_row = int(np.round(np.sqrt(current_layer.n_input)))
        n_output = current_layer.n_output
        Wimg = display_weigths(current_layer.W.get_value(borrow=True), n_row, n_row, n_output)
        scipy.misc.imsave('final_filter_at_layer_%i.png' % layer, Wimg)
        np.savez('parameters_at_layer_%i.npz' % layer, state=(layer),
                                                       W=current_layer.W.get_value(borrow=True),
                                                       a=current_layer.b.get_value(borrow=True),
                                                       b=current_layer.a.get_value(borrow=True))

if __name__ == '__main__':
    test_dbn(training_epochs=10,k=1)