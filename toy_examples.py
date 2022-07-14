import numpy as np
import FastGromovWass
import utils

import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D  # noqa


### Some examples of toy data
def Mixture_of_Gaussians(num_samples, sigma, dimension1, dimension2, seed=49):
    nX1 = int(num_samples / 3)
    nX2 = nX1
    nX3 = num_samples - 2 * nX1

    cov1 = sigma * np.eye(dimension1)

    mean_X1 = np.zeros(dimension1)
    mean_X2 = np.zeros(dimension1)
    mean_X2[1] = 1
    mean_X3 = np.zeros(dimension1)
    mean_X3[0], mean_X3[1] = 1, 1

    X1 = np.random.multivariate_normal(mean_X1, cov1, nX1)
    X2 = np.random.multivariate_normal(mean_X2, cov1, nX2)
    X3 = np.random.multivariate_normal(mean_X3, cov1, nX3)

    X = np.concatenate([X1, X2, X3], axis=0)

    nY1 = int(num_samples / 2)
    nY2 = num_samples - nY1

    mean_Y1 = np.zeros(dimension2)
    mean_Y1[0], mean_Y1[1] = 0.5, 0.5

    mean_Y2 = np.zeros(dimension2)
    mean_Y2[0], mean_Y2[1] = -0.5, 0.5

    cov2 = sigma * np.eye(dimension2)

    Y1 = np.random.multivariate_normal(mean_Y1, cov2, nY1)
    Y2 = np.random.multivariate_normal(mean_Y2, cov2, nY2)

    Y = np.concatenate([Y1, Y2], axis=0)

    return X, Y


def simul_two_Gaussians(num_samples, dimension1, dimension2, seed=49):
    np.random.seed(seed)

    mean_X = np.zeros(dimension1)
    var = 1
    cov_X = var * np.eye(dimension1)
    X = np.random.multivariate_normal(mean_X, cov_X, num_samples)

    mean_Y = 4 * np.ones(dimension2)
    cov_Y = var * np.eye(dimension2)
    Y = np.random.multivariate_normal(mean_Y, cov_Y, num_samples)
    # norms = np.linalg.norm(Y, axis=1)
    # norm_max = np.max(norms)
    # Y = Y / norm_max

    return X, Y


def curve_2d_3d(num_samples):
    theta = np.linspace(-4 * np.pi, 4 * np.pi, num_samples)
    z = np.linspace(1, 2, num_samples)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    X = np.concatenate([x.reshape(-1, 1), z.reshape(-1, 1)], axis=1)
    Y = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=1)

    return X, Y


n, m = 1000, 1000  # nb samples
X, Y = curve_2d_3d(n)

fig = pl.figure()
ax1 = fig.add_subplot(121)
ax1.plot(X[:, 0], X[:, 1], "+b", label="Source samples")
ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color="r")
pl.show()


## Two Gaussians
# dimX, dimY = 10,15
# X,Y = simul_two_Gaussians(n,dimX, dimY,seed=49)
# Y = X.copy()


## Two Mixture of Gaussians
# dimX, dimY = 10,15
# sigma = 0.05
# dimX, dimY = 10, 15
# X,Y = Mixture_of_Gaussians(n, sigma, dimX, dimY, seed=49)


### Define the cost function: here we give several examples
Square_Euclidean_cost = lambda X, Y: utils.Square_Euclidean_Distance(X, Y)
L1_cost = lambda X, Y: utils.Lp_Distance(X, Y, p=1)
L3_cost = lambda X, Y: utils.Lp_Distance(X, Y, p=3)

cost = Square_Euclidean_cost

## Define the factorized cost function
rank_cost = 100
cost_factorized = lambda X, Y: utils.factorized_distance_cost(
    X, Y, rank_cost, cost, C_init=False, tol=1e-1, seed=50
)

## Here is an exact implementation of the factorized SE distance
cost_factorized = lambda X, Y: utils.factorized_square_Euclidean(X, Y)


## Compute the cost matrices
D1 = cost(X, X)
D11, D12 = cost_factorized(X, X)

D2 = cost(Y, Y)
D21, D22 = cost_factorized(Y, Y)


## Normalize the cost matrices
r1, r2 = D1.max(), D2.max()
D1, D2 = D1 / r1, D2 / r2
D11, D12 = D11 / np.sqrt(r1), D12 / np.sqrt(r1)
D21, D22 = D21 / np.sqrt(r2), D22 / np.sqrt(r2)


## Define the marginals
a, b = (1 / n) * np.ones(n), (1 / m) * np.ones(m)


### Compute GW cost with a trivial initialization
res = FastGromovWass.GW_init_factorized(D11, D12, D21, D22, a, b)
print(res)


### Entropic GW: cubic method
reg = 5 * 1e-3
res, acc, tim, num_ops, Couplings = FastGromovWass.GW_entropic_distance(
    D1,
    D2,
    reg,
    a,
    b,
    Init="lower_bound",
    seed_init=49,
    I=100,
    delta_sin=1e-3,
    num_iter_sin=10000,
    lam_sin=0,
    LSE=False,
    time_out=50,
)
print(res)

# Plot the coupling after an non-trivial initialization
pl.imshow(Couplings[0], interpolation="nearest", cmap="Greys", aspect="auto")

# Plot the final coupling obtained
pl.imshow(Couplings[-1], interpolation="nearest", cmap="Greys", aspect="auto")


### Entropic GW: quadratic method
reg = 5 * 1e-3
res, acc, tim, num_ops, Couplings = FastGromovWass.Quad_GW_entropic_distance(
    D11,
    D12,
    D21,
    D22,
    reg,
    a,
    b,
    Init="lower_bound",
    seed_init=49,
    I=100,
    delta_sin=1e-3,
    num_iter_sin=10000,
    lam_sin=0,
    LSE=False,
    time_out=50,
)
print(res)

# Plot the coupling after an non-trivial initialization
pl.imshow(Couplings[0], interpolation="nearest", cmap="Greys", aspect="auto")

# Plot the final coupling obtained
pl.imshow(Couplings[-1], interpolation="nearest", cmap="Greys", aspect="auto")


### LR-GW: Quadratic method
rank = 10
cost_SE = (D1, D2)
results = FastGromovWass.apply_quad_lr_gw(
    X, Y, a, b, rank, cost_SE, gamma_0=10, rescale_cost=False, time_out=50
)
res, Q, R, g = results
print(res)

# Plot the coupling obtained
P = np.dot(Q / g, R.T)
pl.imshow(P, interpolation="nearest", cmap="Greys", aspect="auto")

### LR-GW: Linear method
rank = 10
cost_SE = (D11, D12, D21, D22)
results = FastGromovWass.apply_lin_lr_gw(
    X, Y, a, b, rank, cost_SE, gamma_0=10, rescale_cost=False, time_out=50
)

res, Q, R, g = results
print(res)

# Plot the final coupling obtained
P = np.dot(Q / g, R.T)
pl.imshow(P, interpolation="nearest", cmap="Greys", aspect="auto")
