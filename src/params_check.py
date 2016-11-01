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

import matplotlib.pyplot as plt
import numpy as np
import MDBN

# Initialize background to dark gray
def display_weights(W, nRows=5, nCols=8, dimX = 20, dimY = 40 ):
   X = np.vstack((W,np.zeros((dimX*dimY-W.shape[0],W.shape[1]))))
   tiled = np.ones(((dimY+1)*nRows, (dimX+1)*nCols), dtype='uint8') * 51
   for row in xrange(nRows):
      for col in xrange(nCols):
         patch = X[:,row*nCols + col].reshape((dimY,dimX))
         normPatch = ((patch - patch.min()) /
                (patch.max()-patch.min()+1e-6))
         tiled[row*(dimY+1):row*(dimY+1)+dimY, col*(dimX+1):col*(dimX+1)+dimX] = \
                                  normPatch * 255
   plt.imshow(tiled)

def display_sample(X, dimX=20, dimY=40, cmap='gray'):
    y = np.zeros(dimX * dimY)
    y[:X.shape[0] - dimX*dimY] = X
    plt.imshow(y.reshape(dimX,dimY),cmap=cmap)

def plotit(values):
    plt.hist(values);
    plt.title('mm = %g' % np.mean(np.fabs(values)))

def run(datafile, training_fn):
    dbn, _, _ = training_fn(datafile,graph_output=True, layers_sizes=[290, 40],
                            pretrain_lr=[0.01, 0.01], pretraining_epochs=[8000, 800])

    hbias = dbn.rbm_layers[0].hbias.get_value(borrow=True)
    vbias = dbn.rbm_layers[0].vbias.get_value(borrow=True)
    W = dbn.rbm_layers[0].W.get_value(borrow=True)

    return hbias, vbias, W

"""
 Utils to support visual tuning of the learning parameters
 See http://yosinski.com/media/papers/Yosinski2012VisuallyDebuggingRestrictedBoltzmannMachine.pdf
 "Visually Debugging Restricted Boltzmann Machine Training
 with a 3D Example" Yosinski 2010
"""
if __name__ == '__main__':
    datafiles = MDBN.prepare_TCGA_datafiles()
    hbias, vbias, W = run(datafiles['GE'],MDBN.train_GE)

    plt.close(1)
    plt.close(2)

    plt.figure(3)
    plt.subplot(221); plotit(W)
    plt.subplot(222); plotit(hbias)
    plt.subplot(223); plotit(vbias)

    plt.figure(4)
    display_weights(W, nRows = 20, nCols = 14, dimX = 126, dimY = 128 ) # GE