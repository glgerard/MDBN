import os
import datetime

import numpy
import theano

from MDBN import train_bottom_layer
from MDBN import train_top
from MDBN import DBN

from utils import find_unique_classes
from utils import load_n_preprocess_data

# batch_size changed from 1 as in M.Liang to 20

def train_AML_MDBN(datafiles,
                   datadir='data',
                   batch_size=20,
                   holdout=0.1,
                   repeats=10,
                   graph_output=False,
                   output_folder='MDBN_run',
                   output_file='parameters_and_classes.npz',
                   rng=None):
    """
    :param datafile: path to the dataset

    :param batch_size: size of a batch used to train the RBM
    """

    if rng is None:
        rng = numpy.random.RandomState(123)

    #################################
    #     Training the DBM          #
    #################################

    me_DBN, output_ME_t_set, output_ME_v_set = train_ME(datafiles['ME'],
                                                        holdout=holdout,
                                                        repeats=repeats,
                                                        lambda_1=0.01,
                                                        lambda_2=0.01,
                                                        graph_output=graph_output,
                                                        datadir=datadir)

    ge_DBN, output_GE_t_set, output_GE_v_set = train_GE(datafiles['GE'],
                                                        holdout=holdout,
                                                        repeats=repeats,
                                                        lambda_1=0.01,
                                                        lambda_2=0.1,
                                                        graph_output=graph_output,
                                                        datadir=datadir)
    sm_DBN, output_SM_t_set, output_SM_v_set = train_SM(datafiles['SM'],
                                                        holdout=holdout,
                                                        repeats=repeats,
                                                        lambda_1=0.01,
                                                        lambda_2=0.01,
                                                        graph_output=graph_output,
                                                        datadir=datadir)

#    dm_DBN, output_DM_t_set, output_DM_v_set = train_DM(datafiles['DM'],
#                                                        holdout=holdout,
#                                                        repeats=repeats,
#                                                        lambda_1=0.01,
#                                                        lambda_2=0.1,
#                                                        graph_output=graph_output,
#                                                        datadir=datadir)

    print('*** Training on joint layer ***')

    output_ME_t_set, output_ME_v_set = me_DBN.MLP_output_from_datafile(datafiles['ME'], holdout=holdout, repeats=repeats)
    output_GE_t_set, output_GE_v_set = ge_DBN.MLP_output_from_datafile(datafiles['GE'], holdout=holdout, repeats=repeats)
    output_SM_t_set, output_SM_v_set = sm_DBN.MLP_output_from_datafile(datafiles['SM'], holdout=holdout, repeats=repeats)

    # output_DM_t_set, output_DM_v_set = dm_DBN.MLP_output_from_datafile(datafiles['DM'], holdout=holdout, repeats=repeats)

    joint_train_set = theano.shared(numpy.concatenate([
    #               output_ME_t_set, output_GE_t_set, output_DM_t_set],axis=1), borrow=True)
                    output_ME_t_set, output_GE_t_set, output_SM_t_set], axis = 1), borrow = True)

    if holdout > 0:
        joint_val_set = theano.shared(numpy.concatenate([
    #                        output_ME_v_set, output_GE_v_set, output_DM_v_set],axis=1), borrow=True)
                            output_ME_v_set, output_GE_v_set, output_SM_v_set],axis=1), borrow=True)
    else:
        joint_val_set = None

    top_DBN = train_top(batch_size, graph_output, joint_train_set, joint_val_set, rng)

    # Identifying the classes

    ME_output, _ = me_DBN.MLP_output_from_datafile(datafiles['ME'])
    GE_output, _ = ge_DBN.MLP_output_from_datafile(datafiles['GE'])
    SM_output, _ = sm_DBN.MLP_output_from_datafile(datafiles['SM'])

#    DM_output, _ = dm_DBN.MLP_output_from_datafile(datafiles['DM'])

#    joint_output = theano.shared(numpy.concatenate([ME_output, GE_output, DM_output],axis=1), borrow=True)
    joint_output = theano.shared(numpy.concatenate([ME_output, GE_output, SM_output],axis=1), borrow=True)

    classes = top_DBN.get_output(joint_output)

#    save_network(classes, ge_DBN, me_DBN, dm_DBN, top_DBN, holdout, output_file, output_folder, repeats)
    save_network(classes, ge_DBN, me_DBN, sm_DBN, None, top_DBN, holdout, output_file, output_folder, repeats)

    return classes

def save_network(classes, ge_DBN, me_DBN, sm_DBN, dm_DBN, top_DBN, holdout, output_file, output_folder, repeats):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    root_dir = os.getcwd()
    os.chdir(output_folder)
    numpy.savez(output_file,
                holdout=holdout,
                repeats=repeats,

                me_config={
                    'number_of_nodes': me_DBN.number_of_nodes(),
                    'epochs': [80000],
                    'learning_rate': [0.005],
                    'batch_size': 20,
                    'k': 10
                },
                ge_config={
                    'number_of_nodes': ge_DBN.number_of_nodes(),
                    'epochs': [8000, 800],
                    'learning_rate': [0.005, 0.1],
                    'batch_size': 20,
                    'k': 1
                },
                sm_config={
                    'number_of_nodes': sm_DBN.number_of_nodes(),
                    'epochs': [8000, 800],
                    'learning_rate': [0.005, 0.1],
                    'batch_size': 20,
                    'k': 1
                },
#                dm_config={
#                    'number_of_nodes': dm_DBN.number_of_nodes(),
#                    'epochs': [8000, 800],
#                    'learning_rate': [0.005, 0.1],
#                    'batch_size': 20,
#                    'k': 1
#                },
                top_config={
                    'number_of_nodes': top_DBN.number_of_nodes(),
                    'epochs': [800, 800],
                    'learning_rate': [0.1, 0.1],
                    'batch_size': 20,
                    'k': 1
                },
                classes=classes,
                me_params=[{p.name: p.get_value()} for p in me_DBN.params],
                ge_params=[{p.name: p.get_value()} for p in ge_DBN.params],
                sm_params=[{p.name: p.get_value()} for p in sm_DBN.params],
                #                dm_params=[{p.name: p.get_value()} for p in dm_DBN.params],
                top_params=[{p.name: p.get_value()} for p in top_DBN.params]
                )
    os.chdir(root_dir)

def load_network(input_file, input_folder):
    root_dir = os.getcwd()
    # TODO: check if the input_folder exists
    os.chdir(input_folder)
    npz = numpy.load(input_file)

    config = npz['me_config'].tolist()
    params = npz['me_params']
    layer_sizes = config['number_of_nodes']
    me_DBN = DBN(n_ins=layer_sizes[0], hidden_layers_sizes=layer_sizes[1:-1], n_outs=layer_sizes[-1],
                  W_list=[params[0]['W']],b_list=[params[1]['b']])

    config = npz['ge_config'].tolist()
    params = npz['ge_params']
    layer_sizes = config['number_of_nodes']
    ge_DBN = DBN(n_ins=layer_sizes[0], hidden_layers_sizes=layer_sizes[1:-1], n_outs=layer_sizes[-1],
                  W_list=[params[0]['W'],params[2]['W']],b_list=[params[1]['b'],params[3]['b']])

    config = npz['sm_config'].tolist()
    params = npz['sm_params']
    layer_sizes = config['number_of_nodes']
    sm_DBN = DBN(n_ins=layer_sizes[0], hidden_layers_sizes=layer_sizes[1:-1], n_outs=layer_sizes[-1],
                  W_list=[params[0]['W'],params[2]['W']],b_list=[params[1]['b'],params[3]['b']])

#    config = npz['dm_config'].tolist()
#    params = npz['dm_params']
#    layer_sizes = config['number_of_nodes']
#    dm_DBN = DBN(n_ins=layer_sizes[0], hidden_layers_sizes=layer_sizes[1:-1], n_outs=layer_sizes[-1],
#                  W_list=[params[0]['W'],params[2]['W']],b_list=[params[1]['b'],params[3]['b']])

    config = npz['top_config'].tolist()
    params = npz['top_params']
    layer_sizes = config['number_of_nodes']
    top_DBN = DBN(n_ins=layer_sizes[0], hidden_layers_sizes=layer_sizes[1:-1], n_outs=layer_sizes[-1],
                  gauss=False,
                  W_list=[params[0]['W'],params[2]['W']],b_list=[params[1]['b'],params[3]['b']])

    os.chdir(root_dir)

#    return (me_DBN, ge_DBN, dm_DBN, top_DBN)
    return (me_DBN, ge_DBN, sm_DBN, None, top_DBN)

def train_DM(datafile,
             clip=None,
             batch_size=20,
             k=1,
             lambda_1=0,
             lambda_2=1,
             layers_sizes=[400, 40],
             pretraining_epochs=[8000, 800],
             pretrain_lr=[0.005, 0.1],
             holdout=0.1,
             repeats=10,
             graph_output=False,
             datadir='data'):
    print('*** Training on DM ***')

    train_set, validation_set = load_n_preprocess_data(datafile,
                                                       clip=clip,
                                                       holdout=holdout,
                                                       repeats=repeats,
#                                                       transform_fn=numpy.power,
#                                                       exponent=1.0/6.0,
                                                       datadir=datadir)

    return train_bottom_layer(train_set, validation_set,
                              batch_size=batch_size,
                              k=k,
                              layers_sizes=layers_sizes,
                              pretraining_epochs=pretraining_epochs,
                              pretrain_lr=pretrain_lr,
                              lambda_1=lambda_1,
                              lambda_2=lambda_2,
                              graph_output=graph_output)

def train_GE(datafile,
             clip=None,
             batch_size=20,
             k=1,
             lambda_1=0,
             lambda_2=1,
             layers_sizes=[400, 40],
             pretraining_epochs=[8000, 800],
             pretrain_lr=[0.005, 0.1],
             holdout=0.1,
             repeats=10,
             graph_output=False,
             datadir='data'):
    print('*** Training on GE ***')

    train_set, validation_set = load_n_preprocess_data(datafile,
                                                       clip=clip,
                                                       holdout=holdout,
                                                       repeats=repeats,
                                                       datadir=datadir)

    return train_bottom_layer(train_set, validation_set,
                              batch_size=batch_size,
                              k=k,
                              layers_sizes=layers_sizes,
                              pretraining_epochs=pretraining_epochs,
                              pretrain_lr=pretrain_lr,
                              lambda_1=lambda_1,
                              lambda_2=lambda_2,
                              graph_output=graph_output)

def train_ME(datafile,
             clip=None,
             batch_size=20,
             k=10,
             lambda_1=0.0,
             lambda_2=0.1,
             layers_sizes=[40],
             pretraining_epochs=[80000],
             pretrain_lr=[0.005],
             holdout=0.1,
             repeats=10,
             graph_output=False,
             datadir='data'):
    print('*** Training on ME ***')

    train_set, validation_set = load_n_preprocess_data(datafile,
                                                       clip=clip,
                                                       holdout=holdout,
                                                       repeats=repeats,
                                                       datadir=datadir)

    return train_bottom_layer(train_set, validation_set,
                                batch_size=batch_size,
                                k=k,
                                layers_sizes=layers_sizes,
                                pretraining_epochs=pretraining_epochs,
                                pretrain_lr=pretrain_lr,
                                lambda_1=lambda_1,
                                lambda_2=lambda_2,
                                graph_output=graph_output)

def train_SM(datafile,
             clip=None,
             batch_size=20,
             k=1,
             lambda_1=0.0,
             lambda_2=0.1,
             layers_sizes=[200, 20],
             pretraining_epochs=[8000, 800],
             pretrain_lr=[0.005, 0.1],
             holdout=0.1,
             repeats=10,
             graph_output=False,
             datadir='data'):
    print('*** Training on SM ***')

    train_set, validation_set = load_n_preprocess_data(datafile,
                                                       clip=clip,
                                                       holdout=holdout,
                                                       repeats=repeats,
                                                       datadir=datadir)

    return train_bottom_layer(train_set, validation_set,
                                batch_size=batch_size,
                                k=k,
                                layers_sizes=layers_sizes,
                                pretraining_epochs=pretraining_epochs,
                                pretrain_lr=pretrain_lr,
                                lambda_1=lambda_1,
                                lambda_2=lambda_2,
                                graph_output=graph_output)

def prepare_AML_TCGA_datafiles(datadir='data'):
    datafiles = {
        'GE': 'AML/AML_gene_expression_table2.csv',
#        'DM': 'AML/3.Methylation_0.5.out',
        'ME': 'AML/AML_miRNA_Seq_table2.csv',
        'SM': 'AML/AML_somatic_mutations_table2.csv'
    }

    return datafiles

if __name__ == '__main__':
    datafiles = prepare_AML_TCGA_datafiles()

    output_dir = 'MDBN_run'
    run_start_date = datetime.datetime.now()
    run_start_date_str = run_start_date.strftime("%Y-%m-%d_%H%M")
    results = []
    for i in range(1):
        dbn_output = train_AML_MDBN(datafiles,
                                    output_folder=output_dir,
                                    output_file='Exp_%s_run_%d.npz' %
                                                               (run_start_date_str, i),
                                    holdout=0.0, repeats=1)
        results.append(find_unique_classes((dbn_output > 0.5) * numpy.ones_like(dbn_output)))

    current_date_time = datetime.datetime.now()
    print('*** Run started at %s' % run_start_date.strftime("%H:%M:%S on %B %d, %Y"))
    print('*** Run completed at %s' % current_date_time.strftime("%H:%M:%S on %B %d, %Y"))

    root_dir = os.getcwd()
    os.chdir(output_dir)
    numpy.savez('Results_%s.npz' % run_start_date_str,
                results=results)
    os.chdir(root_dir)

#    train_ME(datafiles['ME'],graph_output=True)
#    train_GE(datafiles['GE'],graph_output=True)