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

Portions of the code are
Copyright (c) 2010--2015, Deep Learning Tutorials Development Team
All rights reserved.
"""

import numpy
from scipy.spatial import distance
import os
import zipfile
import urllib

def prepare_TCGA_datafiles(datadir='data'):
    base_url = 'http://nar.oxfordjournals.org/content/suppl/2012/07/25/gks725.DC1/'
    archive = 'nar-00961-n-2012-File005.zip'
    datafiles = {
        'GE': 'TCGA_Data/3.GE1_0.5.out',
        'ME': 'TCGA_Data/3.Methylation_0.5.out',
        'mRNA': 'TCGA_Data/3.miRNA_0.5.out'
    }
    if not os.path.isdir(datadir):
        os.mkdir(datadir)
    root_dir = os.getcwd()
    os.chdir(datadir)
    for name, datafile in datafiles.iteritems():
        if not os.path.isfile(datafile):
            if not os.path.isfile(archive):
                print('Downloading TCGA_Data from ' + base_url)
                testfile = urllib.URLopener()
                testfile.retrieve(base_url + archive, archive)
            zipfile.ZipFile(archive, 'r').extract(datafile)
    os.chdir(root_dir)
    return datafiles

def import_TCGA_data(file, datadir, dtype):
    root_dir = os.getcwd()
    os.chdir(datadir)
    with open(file) as f:
        ncols = len(f.readline().split('\t'))

    data = numpy.loadtxt(file,
                       dtype=dtype,
                       delimiter='\t',
                       skiprows=1,
                       usecols=range(1,ncols))

    os.chdir(root_dir)
    return (data.shape[1], ncols-1, data)

def get_minibatches_idx(n, batch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + batch_size])
        minibatch_start += batch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return range(len(minibatches)), minibatches

# The following function help reduce the number of classes based on highest
# frequency and lowest Hamming distance

def remap_class(classified_samples, distance_matrix, n_classes):
    def class_by_frequency(a):
        classes = range(int(numpy.max(a)) + 1)
        frequency = [numpy.sum(a == idx) for idx in classes]
        idx_sort0 = reversed(numpy.argsort(frequency).tolist())
        idx_sort1 = reversed(numpy.argsort(frequency).tolist())

        return [{classes[i]: (r, frequency[i]) for r, i in enumerate(idx_sort0)},
                {r: (classes[i], frequency[i]) for r, i in enumerate(idx_sort1)}]

    def merge_classes(map, D, n_classes):
        new_map = {}
        n_initial_classes = D.shape[0]

        for i in range(n_classes):
            new_map[map[1][i][0]] = i

        to_be_merged_classes = [map[1][i][0] for i in range(n_classes, n_initial_classes)]
        for c in to_be_merged_classes:
            for i in numpy.argsort(D[c]):
                r = map[0][i][0]
                if r < n_classes and r != c:
                    new_map[c] = r

        return new_map

    # Order the classes by frequency and create an ordered list of those
    map = class_by_frequency(classified_samples)

    # Reduce the number of classes to retain only the fist n_classes by frequency
    # Samples belonging to lower frequency classes will be reassigned to the nearest
    # permitted class by hamming distance
    # TODO: evaluate if hierarchical clustering leads to better results

    new_classification = merge_classes(map=map,D=distance_matrix,n_classes=n_classes)
    return numpy.array([new_classification[i] for i in classified_samples])


def find_unique_classes(dbn_output):
    # Find the unique node patterns present in the output
    tmp = numpy.ascontiguousarray(dbn_output).view(numpy.dtype(
        (numpy.void, dbn_output.dtype.itemsize * dbn_output.shape[1])))
    _, idx = numpy.unique(tmp, return_index=True)
    class_representation = dbn_output[idx]
    # Find the Hamming distances among all the classes
    distance_matrix = distance.cdist(class_representation, class_representation, metric='hamming')
    # Assign each sample to a class
    classified_samples = numpy.zeros((dbn_output.shape[0]))
    output_nodes = dbn_output.shape[1]
    for idx, pattern in enumerate(class_representation):
        classified_samples = classified_samples + \
                             (numpy.sum(dbn_output == pattern, axis=1) == output_nodes) * idx

    return classified_samples, distance_matrix