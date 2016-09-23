import numpy as np
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import struct
import scipy.misc

class GRBM(object):
    # Implement a Gaussian-Bernoulli Restricted Boltzmann Machine

    def __init__(self, input, n_input, n_hidden):
        self.input = input
        self.n_input = n_input
        self.n_hidden = n_hidden

        # Rescale terms for visible units
        self.b = theano.shared(value=np.zeros(self.n_input, dtype=theano.config.floatX),
                               borrow=True,
                               name='b')
        # Bias terms for hidden units
        self.a = theano.shared(np.zeros(self.n_hidden,dtype=theano.config.floatX),
                               borrow=True,
                               name='a')
        # Rescale terms for visible units
        self.sigma = theano.shared(np.ones(self.n_input, dtype=theano.config.floatX),
                                   borrow=True,
                                   name='sigma')
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
        mu = self.b + self.sigma * T.dot(h,self.W.T)
        return [mu, self.srng.normal(size=mu.shape, avg=mu, std=self.sigma*self.sigma, dtype=theano.config.floatX)]

    def h_sample(self, v):
        # Derive a sample of hidden units from the visible units v
        activation = self.a + T.dot(v/self.sigma,self.W)
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

def load_MNIST():
    #  http://yann.lecun.com/exdb/mnist/

    # Read the images
    with open('../data/train-images-idx3-ubyte', 'rb') as file:
        data = file.read()
        header = struct.unpack(">IIII", data[:16])
        n_images = header[1]
        print("Reading %i images" % n_images)
        n_row = header[2]
        n_col = header[3]
        print("Images dimension %i x %i" % (n_row, n_col))
        img_size = n_row * n_col
        images = np.zeros((n_images,img_size),dtype=theano.config.floatX)
        for i in xrange(n_images):
            image = struct.unpack("B"*img_size, data[16+i*img_size:16+(i+1)*img_size])
            images[i] = list(image)

        # Normalize the images to have zero mean and unit standar
        # deviation (see https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf 13.2)
        images = images-np.mean(images,axis=1,keepdims=True)
        images = images / np.std(images,axis=1,keepdims=True)

    # Read the labels
    with open('../data/train-labels-idx1-ubyte') as file:
        data = file.read()
        header = struct.unpack(">II", data[:8])
        n_labels = header[1]
        print("Reading %i labels" % n_images)

        labels = np.zeros((n_labels,1),dtype=theano.config.floatX)
        for i in xrange(n_labels):
            label = struct.unpack("B", data[8+i:8+i+1])
            labels[i] = list(label)

        n_levels = np.int(np.max(labels)-np.min(labels)+1)

    return [n_images, img_size, images, n_levels, labels]

def test(batch_size = 20, training_epochs = 15, k=1):

    n_data, input_size, dataset, levels, targets = load_MNIST()

    index = T.lscalar('index')
    x = T.matrix('x')
    print("Building an RPM with %i visible inputs and %i hidden units" % (input_size, levels))
    rbm = GRBM(x, input_size, levels)

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
        scipy.misc.imsave('filters_at_epoch_%i.png' % epoch,rbm.W.get_value(borrow=True).T)

    vis_sample = theano.shared(np.asarray(dataset[1000:1010],dtype=theano.config.floatX))
    ( [ nv_probs,
        nv_samples,
        nh_probs,
        nh_samples],
        updates) = theano.scan(rbm.alt_gibbs_update,outputs_info=[None, vis_sample, None, None],
                                 n_steps=1000,
                                 name="alt_gibbs_update")

    run_gibbs = theano.function(
            [],
            [nv_samples[-1]],
            updates=updates,
            name="run_gibbs"
        )

    run_gibbs()

    # scipy.misc.imsave('input.png',vis_sample.get_value(borrow=True)[0])
    # scipy.misc.imsave('output.png',nv_samples.get_value(borrow=True)[-1,0])

if __name__ == '__main__':
    test(training_epochs=10)