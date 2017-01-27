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

from __future__ import print_function, division

import os
import numpy
import theano

from utils import load_n_preprocess_data
from dbn import DBN

def train_dbn(train_set, validation_set,
              gauss=True,
              batch_size=20,
              k=1, layers_sizes=[40],
              pretraining_epochs=[800],
              pretrain_lr=[0.005],
              lambdas = [0.01, 0.1],
              rng=None,
              run=0,
              persistent=False,
              verbose=False,
              graph_output=False
              ):
    print('Visible nodes: %i' % train_set.get_value().shape[1])
    print('Output nodes: %i' % layers_sizes[-1])
    dbn = DBN(numpy_rng=rng, n_ins=train_set.get_value().shape[1],
                gauss=gauss,
                hidden_layers_sizes=layers_sizes[:-1],
                n_outs=layers_sizes[-1])

    dbn.training(train_set,
                 batch_size, k,
                 pretraining_epochs,
                 pretrain_lr,
                 lambdas,
                 persistent=persistent,
                 run=run,
                 verbose=verbose,
                 validation_set_x=validation_set,
                 graph_output=graph_output)

    output_train_set = dbn.get_output(train_set)
    if validation_set is not None:
        output_val_set = dbn.get_output(validation_set)
    else:
        output_val_set = None

    return dbn, output_train_set, output_val_set

def train_MDBN(datafiles,
               config,
               datadir='data',
               holdout=0,
               repeats=1,
               run=0,
               verbose=False,
               graph_output=False,
               output_folder='MDBN_run',
               output_file='parameters_and_classes.npz',
               rng=None):
    """
    :param datafiles: a dictionary with the path to the unimodal datasets

    :param datadir: directory where the datasets are located

    :param holdout: percentage of samples used for validation. By default there
                    is no validation set

    :param repeats: repeat each sample repeats time to artifically increase the size
                    of each dataset. By default data is not repeated

    :param graph_output: if True it will output graphical representation of the
                        network parameters

    :param output_folder: directory where the results are stored

    :param output_file: name of the file where the parameters are saved at the end
                        of the training

    :param rng: random number generator, by default is None and it is initialized
                by the function
    """

    if rng is None:
        rng = numpy.random.RandomState(123)

    #################################
    #     Training the RBM          #
    #################################

    dbn_dict = dict()
    output_t_list = []
    output_v_list = []

    for key in config["pathways"]:
        print('*** Run %i - Training on %s ***' % (run, key))

        train_set, validation_set = load_n_preprocess_data(datafiles[key],
                                                       holdout=holdout,
                                                       repeats=repeats,
                                                       datadir=datadir,
                                                       rng=rng)

        netConfig = config[key]
        netConfig['inputNodes'] = train_set.get_value().shape[1]

        dbn_dict[key], _, _ = train_dbn(train_set, validation_set,
                                  gauss=True,
                                  batch_size=netConfig["batchSize"],
                                  k=netConfig["k"],
                                  layers_sizes=netConfig["layersNodes"],
                                  pretraining_epochs=netConfig["epochs"],
                                  pretrain_lr=netConfig["lr"],
                                  lambdas=netConfig["lambdas"],
                                  rng=rng,
                                  persistent=netConfig["persistent"],
                                  run=run,
                                  verbose=verbose,
                                  graph_output=graph_output)

        output_t, output_v = dbn_dict[key].MLP_output_from_datafile(datafiles[key],
                                                                    holdout=holdout,
                                                                    repeats=repeats)
        output_t_list.append(output_t)
        output_v_list.append(output_v)

    print('*** Run %i - Training on joint layer ***' % run)

    joint_train_set = theano.shared(numpy.hstack(output_t_list), borrow=True)

    if holdout > 0:
        joint_val_set = theano.shared(numpy.hstack(output_v_list), borrow=True)
    else:
        joint_val_set = None

    netConfig = config['top']
    netConfig['inputNodes'] = joint_train_set.get_value().shape[1]

    dbn_dict['top'], _, _ = train_dbn(joint_train_set, joint_val_set,
                                      gauss=False,
                                      batch_size=netConfig["batchSize"],
                                      k=netConfig["k"],
                                      layers_sizes=netConfig["layersNodes"],
                                      pretraining_epochs=netConfig["epochs"],
                                      pretrain_lr=netConfig["lr"],
                                      rng=rng,
                                      persistent=netConfig["persistent"],
                                      run=run,
                                      verbose=verbose,
                                      graph_output=graph_output)

    # Identifying the classes

    dbn_output_list = []
    for key in config["pathways"]:
        dbn_output, _ = dbn_dict[key].MLP_output_from_datafile(datafiles[key])
        dbn_output_list.append(dbn_output)

    joint_output = theano.shared(numpy.hstack(dbn_output_list), borrow=True)

    classes = dbn_dict['top'].get_output(joint_output)

    save_network(classes, config, dbn_dict,
                 holdout, output_file, output_folder, repeats)

    return classes

def save_network(classes, config, dbn_dict, holdout, output_file, output_folder, repeats):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    root_dir = os.getcwd()
    os.chdir(output_folder)
    dbn_params = {}
    for n in config['pathways']+['top']:
        dbn = dbn_dict[n]
        params = {}
        for p in dbn.params:
            if p.name in params:
                params[p.name].append(p.get_value())
            else:
                params[p.name] = [p.get_value()]
        dbn_params[n] = params

    numpy.savez(output_file,
                holdout=holdout,
                repeats=repeats,
                config=config,
                classes=classes,
                dbn_params=dbn_params
                )
    os.chdir(root_dir)

def load_network(input_file, input_folder):
    root_dir = os.getcwd()
    # TODO: check if the input_folder exists
    os.chdir(input_folder)
    npz = numpy.load(input_file)

    config = npz['config'].tolist()
    dbn_params = npz['dbn_params'].tolist()

    dbn_dict = {}
    for key in config['pathways']+["top"]:
        params=dbn_params[key]
        netConfig = config[key]
        layer_sizes = netConfig['layersNodes']
        dbn_dict[key] = DBN(n_ins=netConfig['inputNodes'],
                            hidden_layers_sizes=layer_sizes[:-1],
                            gauss=key!='top',
                            n_outs=layer_sizes[-1],
                            W_list=params['W'],b_list=params['hbias'],c_list=params['vbias'])

    os.chdir(root_dir)

    return config, dbn_dict
