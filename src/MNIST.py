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
"""

from __future__ import print_function

import os
import urllib
import gzip

import numpy
import theano
import struct

class MNIST(object):
    def __init__(self,
                 datafile='train-images-idx3-ubyte.gz',
                 targetfile='train-labels-idx1-ubyte.gz',
                 datadir='data'):
        self.n_images = self.load(datafile, targetfile, datadir)

    def load(self, datafile, targetfile, datadir):
        #  http://yann.lecun.com/exdb/mnist/
        if not os.path.isdir(datadir):
            os.mkdir(datadir)

        root_dir = os.getcwd()
        os.chdir(datadir)

        # Read the images
        if not os.path.isfile(datafile):
            print('Downloading the data file from http://yann.lecun.com/exdb/mnist')
            testfile = urllib.URLopener()
            testfile.retrieve("http://yann.lecun.com/exdb/mnist/"+datafile, datafile)

        with gzip.open(datafile, 'rb') as file:
            data = file.read()
            header = struct.unpack(">IIII", data[:16])
            n_images = header[1]
            print("Reading %i images" % n_images)
            self.sizeY = header[2]
            self.sizeX = header[3]
            print("Images dimension %i x %i" % (self.sizeY, self.sizeX))
            img_size = self.sizeY * self.sizeX
            self.images = numpy.zeros((n_images,img_size),dtype=theano.config.floatX)
            for i in xrange(n_images):
                image = struct.unpack("B"*img_size, data[16+i*img_size:16+(i+1)*img_size])
                self.images[i] = list(image)

        # Read the labels
        if not os.path.isfile(targetfile):
            print('Downloading the target file from http://yann.lecun.com/exdb/mnist')
            testfile = urllib.URLopener()
            testfile.retrieve("http://yann.lecun.com/exdb/mnist/"+targetfile, targetfile)

        with gzip.open(targetfile, 'rb') as file:
            data = file.read()
            header = struct.unpack(">II", data[:8])
            n_labels = header[1]
            print("Reading %i labels" % n_images)

            self.labels = numpy.zeros((n_labels,1),dtype=theano.config.floatX)
            for i in xrange(n_labels):
                label = struct.unpack("B", data[8+i:8+i+1])
                self.labels[i] = list(label)

            self.n_levels = numpy.int(numpy.max(self.labels)-numpy.min(self.labels)+1)

        os.chdir(root_dir)
        return n_images

    def normalize(self, X):
        # Normalize the images features to have zero mean and approximately unit standard
        # deviation (see https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf 13.2)
        X = (X - 128.0) / 128.0
        # take the global standard deviation as a normalization constant for all features
        gs = numpy.std(X)
        return X / gs

    def display_weigths(self, X, n_hidden):
        X=X.T
        n_tiles = int(numpy.round(numpy.sqrt(n_hidden)))
        img_gap_row = self.sizeY + 1
        img_gap_col = self.sizeX + 1
        Y = numpy.zeros((n_tiles * img_gap_row, n_tiles * img_gap_col), dtype=theano.config.floatX)
        for r in xrange(n_tiles):
            for c in xrange(n_tiles):
                if (r*n_tiles + c) < n_hidden:
                    Y[r*img_gap_row:(r + 1) * img_gap_row - 1, c * img_gap_col:(c + 1) * img_gap_col - 1] =\
                        X[r * n_tiles + c].reshape(self.sizeX, self.sizeY)
                else:
                    break
        return Y

    def display_samples(self, samples):
        img_gap_row = self.sizeY + 1
        img_gap_col = self.sizeX + 1
        Y = numpy.zeros((len(samples) * img_gap_row, len(samples[0]) * img_gap_col), dtype=theano.config.floatX)

        for (r, sample) in enumerate(samples):
            for (c, img) in enumerate(sample):
                Y[r*img_gap_row:(r + 1) * img_gap_row - 1, c * img_gap_col:(c + 1) * img_gap_col - 1] =\
                    img.reshape(self.sizeX, self.sizeY)
        return Y