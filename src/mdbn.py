import numpy as np
import theano
from theano import tensor
from utils import zscore
from rbm import RBM
from rbm import GRBM

def importdata(file):
    with open(file) as f:
        ncols = len(f.readline().split('\t'))

    return (ncols-1,
            np.loadtxt(file,
                       dtype=theano.config.floatX,
                       delimiter='\t',
                       skiprows=1,
                       usecols=range(1,ncols)))

def test_mdbn():
    # upload the data, each column is a single person
    gecols, GE = importdata("../data/3.GE1_0.5.out")
    mecols, ME  = importdata("../data/3.Methylation_0.5.out")
    rnacols, mRNA  = importdata("../data/3.miRNA_0.5.out")
    # Pass to a row representation, i.e. the data for each person is now on a
    # single row.
    # Normalize the data so that each measurement on our population has zero
    # mean and zero variance
    datage = zscore(GE.T)
    datame = zscore(ME.T)
    datarna = zscore(mRNA.T)

    x = tensor.matrix('x')

#    rbm.training(self, dataset, batch_size, training_epochs, k, lr,
#                 lam1=0, lam2=0,
#                 stocastic_steps=True,
#                 data_shuffle=False,
#                 display_fn=None)

    # Create the 1st layer of Gaussian Bernoulli RBM
    rna_GRBM_l1 = GRBM(x, datarna.shape[1], 40)
    rna_GRBM_l1.training(datarna, 1, 8000, 10, 0.0005, 0.1, 0.1, data_shuffle=True, stocastic_steps=False)

    ge_GRBM_l1 = GRBM(x, datage.shape[1], 400)
    ge_GRBM_l1.training(datage, 1, 8000, 1, 0.0005, 1, 1, data_shuffle=True, stocastic_steps=False)
    ge_RBM_l2 = RBM(x, 400, 40)
    ge_RBM_l2 = ge_RBM_l2.training(ge_GRBM_l1.output(),1,800, 1, 0.1)

    me_GRBM_l1 = GRBM(x, datame.shape[1], 400)
    me_GRBM_l1.training(datame, 1, 8000, 1, 0.0005, 1, 1, data_shuffle=True, stocastic_steps=False)
    me_RBM_l2 = RBM(x, 400, 40)
    me_RBM_l2 = me_RBM_l2.training(me_GRBM_l1.output(),1,800, 1, 0.1)

    joint_layer = []
    joint_layer.append(rna_GRBM_l1.output())
    joint_layer.append(ge_RBM_l2.output())
    joint_layer.append(me_RBM_l2.output())

    np.savez('parameters_at_gaussian_layer_RNA.npz',
             k=20,
             epoch=8000,
             batch_size=10,
             learning_rate=0.0005,
             stocastic_steps=False,
             momentum=False,
             weight_cost=False,
             W=rnaGRBM.W.get_value(borrow=True),
             a=rnaGRBM.a.get_value(borrow=True),
             b=rnaGRBM.b.get_value(borrow=True))

if __name__ == '__main__':
    test_mdbn()
