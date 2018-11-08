import numpy as np
import matplotlib.pyplot as plt
import DimensionalityReduction.Hessian_LLE as Hessian_LLE
import DimensionalityReduction.Isomap as Isomap
import DimensionalityReduction.Kernel_PCA as Kernel_PCA
import DimensionalityReduction.Laplacian_Eigenmaps as Laplacian_Eigenmaps
import DimensionalityReduction.LLE as LLE
import DimensionalityReduction.LTSA as LTSA
import DimensionalityReduction.MDS as MDS
import DimensionalityReduction.Modified_LLE as Modified_LLE
import DimensionalityReduction.PCA as PCA
import DimensionalityReduction.TSNE as TSNE

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from Datasets import SwissRollLoader

# LOAD DATA
X, colors = SwissRollLoader.load_swissroll(n_datapoints=1000)

# VISUALISATION
def plot_original_swissroll():
    '''
        Plots the original Swiss roll dataset.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1] ,X[:,2], c=colors, cmap=plt.cm.Spectral)
    ax.set_title("Original Swiss Roll dataset")

# VISUALISATION
def plot_low_dimensional_embedding(X_low, name="", time=0, error=0):
    '''
        Plots the low dimensional embedding given by the matrix X_low (n_samples, n_features).

        X_low: The low-dimensional representation.
        name: The (string) name of the technique used for the low-dimensional embedding.
        time: Time the execution took.
        error: The error of the representation.
    '''
    plt.figure()
    ax = plt.subplot(111) 
    ax.scatter(X_low[:, 0], X_low[:, 1], c=colors, s=20, cmap=plt.cm.Spectral)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')
    if error != 0:
        plt.title("{}, time: {:.3f}s, error: {:.3e}".format(name, time, error))
    else:
        plt.title("{}, time: {:.3f}s".format(name, time))

num_neighbors = 12
# PCA
X_pca, tpca = PCA.fit_transform(X)
plot_low_dimensional_embedding(X_pca, "PCA", tpca)
# LLE
X_lle, tlle, err_lle = LLE.fit_transform(X, num_neighbors)
plot_low_dimensional_embedding(X_lle, "LLE", tlle, err_lle)
# Hessian LLE
X_hlle, thlle, err_hlle = Hessian_LLE.fit_transform(X, num_neighbors)
plot_low_dimensional_embedding(X_hlle, "Hessian LLE", thlle, err_hlle)
# Modified LLE
X_mlle, tmlle, err_mlle = Modified_LLE.fit_transform(X, num_neighbors)
plot_low_dimensional_embedding(X_mlle, "Modified LLE", tmlle, err_mlle)
# LTSA
X_ltsa, tltsa, err_ltsa = LTSA.fit_transform(X, num_neighbors)
plot_low_dimensional_embedding(X_ltsa, "LTSA", tltsa, err_ltsa)
# Isomap
X_isomap, tisomap, err_isomap = Isomap.fit_transform(X, n_neighbors=12)
plot_low_dimensional_embedding(X_isomap, "Isomap", tisomap, err_isomap)
# Kernel PCA
X_kpca, tkpca = Kernel_PCA.fit_transform(X, t_kernel="sigmoid")
plot_low_dimensional_embedding(X_kpca, "Kernel PCA", tkpca)
# Laplacian Eigenmaps
X_laplacian, tlaplacian = Laplacian_Eigenmaps.fit_transform(X, n_neighbors=12)
plot_low_dimensional_embedding(X_laplacian, "Laplacian Eigenmaps", tlaplacian)
# Metric MDS
X_mmds, tmmds, stress_m = MDS.fit_transform(X, is_metric=True)
plot_low_dimensional_embedding(X_mmds, "Metric MDS", tmmds, stress_m)
# Non-metric MDS
X_nmmds, tnmmds, stress_nm = MDS.fit_transform(X, is_metric=False)
plot_low_dimensional_embedding(X_nmmds, "Non-metric MDS", tnmmds, stress_nm)
# t-SNE
X_tsne, ttsne = TSNE.fit_transform(X)
plot_low_dimensional_embedding(X_tsne, "t-SNE", ttsne)

plot_original_swissroll()
plt.show();