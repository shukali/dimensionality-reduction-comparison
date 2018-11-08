# Comparison of Dimensionalty Reduction techniques

Here you find an implementation of a **comparison of various dimensionalty reduction techniques from scikit-learn**. The comparison comprises the following techniques:  **PCA, t-SNE, LLE, Hessian LLE, Modified LLE, Isomap, Kernel PCA, Laplacian Eigenmaps, LTSA and (Non-)Metric MDS**.

These techniques will be compared against each other in their performance on four different datasets. Those datasets are **[Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg), [Iris](https://archive.ics.uci.edu/ml/datasets/iris), [Swiss Roll](http://people.cs.uchicago.edu/~dinoj/manifold/swissroll.html)** and **[MNIST](http://yann.lecun.com/exdb/mnist/)**. Except for MNIST, no additional dataset needs to be downloaded.

The results of the comparison will be nicely visualized. The objective is to reach a 2-dimensional visualization of the different datasets with the given dimensionality reduction algorithms. All of the algorithms are chosen from scikit-learn.

## How to start it?

Just choose a dataset, e.g. Iris, and start the Python-script: `python IrisComparison.py`. The projections of the various algorithms will be shown in a 2D plot. To see the performance of the algorithms on the other datasets, start the other `...comparison.py` file, respectively.

## Prerequisites

You need to have the following installed:
- **[numpy](http://www.numpy.org/)**
- **[matplotlib](https://matplotlib.org/)**
- **[scikit-learn](https://scikit-learn.org/stable/)** 
- **[pandas](https://pandas.pydata.org/)**

Just go for `pip install xy`. The code was tested with Python 3.6.6. To see the performance on the MNIST dataset, you need to download it first: 
1. Go to http://yann.lecun.com/exdb/mnist/
2. Download at least the two training .gz files, don't rename them
3. Unpack the files to Datasets/MNIST/

## Visualization
The visualizations at the end of each comparison show the results of a 2D mapping for that specific dataset. The coloring is chosen to be reasonable: In case of classification datasets (e.g. Iris), its according to the *ground-truth* class labels. In case of regression, its the attribute that is tried to be predicted, (e.g. the MPG value for Auto MPG). This makes it easier to compare specific points across the different visualizations and to see, if that specific algorithm is good at recognizing clusters.

## Please Note:
Although carefully chosen, the parameters of the different dimensionality reduction algorithms (e.g. which kernel to choose, the number of neighbours to consider, etc.) may not be the 100% perfect choice for the given dataset. The parameters as they are archieve an adequate, although barely perfect projection of the datasets. If you think the projection for a specific technique can be improved with some specific parameters, feel free to notify me. I would be thankful!


## Authors
* **Marcus Rottschäfer** - [GitHub profile](https://github.com/shukali)
* **Valentino Sabbatino**
* **Steven Söhnel**

If you find any errors, have any ideas or questions regarding the code, feel free to contact us!


## License
This project is licensed under the MIT License. See the [LICENSE.md](LICENSE) file for details.
