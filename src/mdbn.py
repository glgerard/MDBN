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
    # upload the data
    gecols, GE = importdata("../data/3.GE1_0.5.out")
    mecols, ME  = importdata("../data/3.Methylation_0.5.out")
    rnacols, mRNA  = importdata("../data/3.miRNA_0.5.out")
    # Normalize the data so that each sample has zero mean and zero variance
    datage = zscore(GE.T)
    datame = zscore(ME.T)
    datarna = zscore(mRNA.T)

    x = tensor.matrix('x')
    x.tag.test_value = np.asarray([datarna[0]])
    rnaGRBM = GRBM(x, datarna.shape[1], 40)
    rnaGRBM.training(datarna, 1, 50, 10, 0.0005, 0.1, 0.1, data_shuffle=True, stocastic_steps=False)
    geGRBM = GRBM(x, datage.shape[1], 400)
    geGRBM.training(datage, 1, 8000, 20, 0.0005, 1, 1, data_shuffle=True, stocastic_steps=False)

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
