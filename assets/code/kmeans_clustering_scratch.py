import numpy as np
import scipy.io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

mat = scipy.io.loadmat('clustering_data.mat')
mat.keys()
X = mat['X']

initial_centroids = np.array([[3,3],[6,2],[8,5]])
K = initial_centroids.shape[0]
X.shape[0]

def find_closest_centroid(X, centroids):

    """
    Returns the closest centroids in idx for a dataset X
    where each row is a single example. idx = m x 1 vector
    (i.e. each entry in range [1..K])
    Args:
        X        : array(# examples, n)
        centroids: array(K, n)
    Returns:
        idx      : array(# examples, 1)

    Iterate over every example, find its closest centroid, and store
    the index inside idx at the appropriate location. Concretely,
    idx[i] should contain the index of the centroid closest to
    example i. Hence, it should be a value in the range 1..K.
    """

    idx = np.zeros((X.shape[0],1))
    for i in range(X.shape[0]):
        idx[i] = np.argmin(np.sqrt(np.sum((X[i,:]-centroids)**2, axis=1)))+1

    return idx

idx = find_closest_centroid(X, initial_centroids)

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means
    of the data points assigned to each centroid. It is
    given a dataset X where each row is a single data point,
    a vector idx of centroid assignments (i.e. each entry
    in range [1..K]) for each example, and K, the number of
    centroids. A matrix centroids is returned, where each row
    of centroids is the mean of the data points assigned to it.
    Args:
        X        : array(# training examples, 2)
        idx      : array(# training examples, 1)
        K        : int, # of centroids
    Returns:
        centroids: array(# of centroids, 2)
    """

    centroids = np.zeros((K, X.shape[1]))
    for i in range(K):
        centroids[i,:] = np.mean(X[(idx==i+1).T[0], :], axis=0)
    return centroids

centroids = compute_centroids(X, idx, K)

def plot_kmeans(X, centroids, previous, idx, K, axes):
    for i, ax in enumerate(axes):
        sns.scatterplot(x=X[:,0], y=X[:,1], hue=idx.ravel(), legend=False, palette=['r', 'g', 'b'], ax=ax)
        sns.scatterplot(x=centroids[:,0], y=centroids[:,1], marker='X', color='k', legend=False, s=100, ax=ax)
        for i in range(centroids.shape[0]):
            ax.plot([centroids[i,0], previous[i,0]], [centroids[i,1], previous[i,1]], '--k');


def run_kmeans(X, initial_centroids, max_iters, plot_progress):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example. It uses initial_centroids used as the initial
    centroids. max_iters specifies the total number of interactions
    of K-Means to execute. plot_progress is a true/false flag that
    indicates if the function should also plot its progress as the
    learning happens.

    run_kmeans returns
    - centroids,
    - K x n matrix of the computed centroids and idx,
    - m x 1 vector of centroid assignments (i.e. each entry in range [1..K])

    Args:
        X                : array(# training examples, n)
        initial_centroids: array(# of centroids, n)
        max_iters        : int, # of iterations
        plot_progress    : boolean, default set to False
    Returns:
        centroids        : array(# of centroids, n)
        idx              : array(# training examples, 1)
    """


    if plot_progress:
        ncols = 3
        nrows = int(max_iters/ncols)
        if max_iters % ncols > 0:
            nrows = nrows + 1
        fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(20,nrows*8))
        ax_tuple = list(np.ndindex(nrows,ncols))
        for ax in ax_tuple[max_iters:]:
            axes[ax].set_axis_off()
        ax_tuple = ax_tuple[:max_iters]

    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids

    for i in range(max_iters):
        idx = find_closest_centroid(X, centroids)

        if plot_progress:
            plot_axes = [axes[axi] for axi in ax_tuple[i:]]
            axes[ax_tuple[i]].set_title('K-Means iteration {0}/{1}'.format(i+1, max_iters))
            plot_kmeans(X, centroids, previous_centroids, idx, K, plot_axes)
            previous_centroids = centroids
            previous_ax = plt.gca()
        else:
            print('K-Means iteration {0}/{1}'.format(i+1, max_iters))

        centroids = compute_centroids(X, idx, K)

    if plot_progress:
        plt.show()
    return centroids, idx

max_iters = 10
initial_centroids = np.array([[3,3],[6,2],[8,5]])

centroids, idx = run_kmeans(X, initial_centroids, max_iters, plot_progress=True)

# random initialization

def init_random_centroids(X, K):
    """
    Initializes K centroids that are to be
    used in K-Means on the dataset X.

    Args:
        X                : array(# training examples, n)
        K                : int, # of centroids
    Returns:
        initial_centroids: array(# of centroids, n)

    """
    centroids = X[(np.random.choice(X.shape[0], K)), :]
    return centroids


# Set K-Means variables.
K = 3
max_iters = 10

initial_centroids = init_random_centroids(X, K)

# Run K-Means algorithm.
centroids, idx = run_kmeans(X, initial_centroids, max_iters, plot_progress=True)
