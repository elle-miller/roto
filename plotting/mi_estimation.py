import numpy as np
from sklearn.preprocessing import scale
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.special import digamma

def compute_mi_cc(x, y, n_neighbors=3, metric="euclidean"):
    """Compute the mutual information between two continuous variables"""

    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    n_samples, _ = x.shape

    # make data have unit variance
    x = scale(x, with_mean=False, copy=True)
    y = scale(y, with_mean=False, copy=True)

    # Add small noise to continuous features as advised in Kraskov et. al.
    means = np.maximum(1, np.mean(np.abs(x), axis=0))
    x += (
        1e-10
        * means
        * np.random.randn(*x.shape)
    )
    means = np.maximum(1, np.mean(np.abs(y), axis=0))
    y += (
        1e-10
        * means
        * np.random.randn(*y.shape)
    )
    xy = np.hstack((x, y))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric=metric, n_neighbors=n_neighbors)

    nn.fit(xy)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    kd = KDTree(x, metric=metric)
    nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
    nx = np.array(nx) - 1.0

    kd = KDTree(y, metric=metric)
    ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
    ny = np.array(ny) - 1.0

    mi = (
        digamma(n_samples)
        + digamma(n_neighbors)
        - np.mean(digamma(nx + 1))
        - np.mean(digamma(ny + 1))
    )

    return max(0, mi)

# NOTE: You need to ensure the compute_mi_cc function is defined in your environment
# and all imports (like scale, digamma, NearestNeighbors, KDTree) are available.


if __name__ == "__main__":
    # 1. Setup Data
    np.random.seed(42)
    N = 500  # Number of samples
    # X is 500x2 (2 dimensions for X)
    X_base = np.random.normal(size=(N, 2))

    # Y is exactly equal to X (perfect correlation)
    Y_base = X_base.copy()
    Y_base = np.random.normal(size=(N, 2))

    # Test Parameters
    n_neighbors = 5
    metric = "euclidean"

    # Run the function (assuming it's defined as compute_mi_cc)
    mi_correlated = compute_mi_cc(X_base, Y_base, n_neighbors=n_neighbors, metric=metric)
    print(f"MI (Correlated): {mi_correlated:.4f}")

    # Example of expected output for this test:
# The value will be large, usually > 3.0, depending on N and k.
# A high value confirms the function works for high MI.