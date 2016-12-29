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

import numpy
from dbn import DBN

def train_top(batch_size, graph_output, joint_train_set, joint_val_set, rng):
    top_DBN = DBN(numpy_rng=rng, n_ins=joint_train_set.get_value().shape[1],
                  gauss=False,
                  hidden_layers_sizes=[24],
                  n_outs=3)
    top_DBN.pretraining(joint_train_set, joint_val_set,
                        batch_size, k=1,
                        pretraining_epochs=[800, 800],
                        pretrain_lr=[0.1, 0.1],
                        graph_output=graph_output)
    return top_DBN


def train_bottom_layer(train_set, validation_set,
                       batch_size=20,
                       k=1, layers_sizes=[40],
                       pretraining_epochs=[800],
                       pretrain_lr=[0.005],
                       lambda_1 = 0.0,
                       lambda_2 = 0.1,
                       rng=None,
                       graph_output=False
                    ):
    print('Visible nodes: %i' % train_set.get_value().shape[1])
    print('Output nodes: %i' % layers_sizes[-1])
    dbn = DBN(numpy_rng=rng, n_ins=train_set.get_value().shape[1],
                  hidden_layers_sizes=layers_sizes[:-1],
                  n_outs=layers_sizes[-1])

    dbn.pretraining(train_set,
                        validation_set,
                        batch_size, k=k,
                        pretraining_epochs=pretraining_epochs,
                        pretrain_lr=pretrain_lr,
                        lambda_1=lambda_1,
                        lambda_2=lambda_2,
                        graph_output=graph_output)

    output_train_set = dbn.get_output(train_set)
    if validation_set is not None:
        output_val_set = dbn.get_output(validation_set)
    else:
        output_val_set = None

    return dbn, output_train_set, output_val_set