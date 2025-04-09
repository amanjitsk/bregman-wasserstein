from __future__ import division, print_function
from functools import partial
from numbers import Real
from typing import Literal, Optional, Any
from collections.abc import Callable, Sequence
from jax._src.prng import PRNGImpl
import numpy as np
import scipy.special
import jax
from jax import (
    numpy as jnp,
    nn as jnn,
    # tree_util as jtu,
    random as jr,
)
import jax.scipy.stats as jst
from jaxtyping import Array, Scalar, PyTree, ArrayLike, ScalarLike, Float, PRNGKeyArray

import equinox as eqx
import distrax as dx
import optax
import pcax
import jaxopt
from jaxopt import projection as jpr

from ott.math import utils as omu
from ott.geometry.costs import mean_and_cov_to_x
from ott.tools.gaussian_mixture.gaussian import Gaussian

TEMP = 1.0
SCALE = 4.0
INF = TEMP * SCALE

Potential = Callable[[Array], Scalar]
Map = Callable[[Array], Array]
Cost = Callable[[Array, Array], Scalar]

PotentialLike = Callable[[ArrayLike], ScalarLike]
MapLike = Callable[[ArrayLike], ArrayLike]
CostLike = Callable[[ArrayLike, ArrayLike], ScalarLike]


def custom_grad(f: Potential, f_grad: Map) -> Potential:
    @jax.custom_jvp
    @jax.custom_vjp
    def wrap_f(x):
        return f(x)

    def f_fwd(x):
        return f(x), (x,)

    def f_bwd(res: PyTree, g):
        (x,) = res
        return (f_grad(x) * g,)

    @wrap_f.defjvp
    def f_jvp(primals, tangents):
        (x,) = primals
        (x_dot,) = tangents
        return f(x), f_grad(x) @ x_dot

    wrap_f.defvjp(f_fwd, f_bwd)

    return wrap_f


def wrap_potentials(
    f: Potential, g: Potential, mirror: dx.Bijector
) -> tuple[Potential, Potential, dx.Bijector]:
    """Wrap Convex (Bregman) generator f and its conjugate g with the
    manually specified gradients in the bijector mirror, representing
    the mirror map and its inverse."""

    wrap_f = custom_grad(f, mirror.forward)  # pyright: ignore
    wrap_g = custom_grad(g, mirror.inverse)  # pyright: ignore
    return wrap_f, wrap_g, mirror
    # return f, g, mirror


def make_separable(
    f: Potential, g: Potential, mirror: dx.Bijector
) -> tuple[Potential, Potential, dx.Bijector]:
    """Wrap f and g into separable Bregman potential and its conjugate.
    This assumes that each component is separable and identical.
    """

    def potential(x):
        return jnp.sum(f(x))

    def conjugate(y):
        return jnp.sum(g(y))

    mirror = dx.Block(mirror, ndims=1)
    return wrap_potentials(potential, conjugate, mirror)


def _get_func_signature(in_ndims: int | Sequence[int], out_ndims: int | Sequence[int]):
    if isinstance(in_ndims, int):
        in_ndims = (in_ndims,)
    if isinstance(out_ndims, int):
        out_ndims = (out_ndims,)

    def _make_shape(n, prefix="k"):
        return "(" + ",".join([f"{prefix}_{i + 1}" for i in range(n)]) + ")"

    in_shapes = ",".join(
        [_make_shape(n, f"i{i + 1}") for (i, n) in enumerate(in_ndims)]
    )
    out_shapes = ",".join(
        [_make_shape(n, f"o{i + 1}") for (i, n) in enumerate(out_ndims)]
    )

    return in_shapes + "->" + out_shapes


def vectorize(
    f: Callable,
    in_ndims: int | Sequence[int] = 1,
    out_ndims: int | Sequence[int] = 1,
    **kwargs,
):
    return jnp.vectorize(
        f, signature=_get_func_signature(in_ndims, out_ndims), **kwargs
    )


def logdet(x: Array) -> Array:
    return jnp.linalg.slogdet(x).logabsdet


def linear_interpolation(n: int) -> np.ndarray:
    """Only works with 2 vertices (measures)!. Return array shape (n, 2)."""
    t = np.linspace(0, 1, n)
    # list of barycentric weights for the two measures (interpolations)
    W = np.asarray([1 - t, t]).T
    return W


def bilinear_interpolation(n: int) -> np.ndarray:
    """Only works with 4 corners (measures)!. Return array shape (n, n, 4)."""
    X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    # list of barycentric weights for four corners
    W = np.array([(1 - X) * (1 - Y), X * (1 - Y), (1 - X) * Y, X * Y])
    W = W.transpose([1, 2, 0])
    return W


def line(points, n=100):
    """Create a piecewise line between consecutive pairs of N d-dimensionaal points.
    Returns array of shape (n * (N - 1), d), where points is (N, d).
    """
    assert points.ndim == 2
    N, d = points.shape
    return jnp.r_[
        tuple(
            jnp.c_[
                tuple(jnp.linspace(points[i][j], points[i + 1][j], n) for j in range(d))
            ]
            for i in range(N - 1)
        )
    ]


def repeat_leading(x: Array, repeats: int | tuple[int, ...]) -> Array:
    """Repeat x along new leading dimension by repeats."""
    if isinstance(repeats, int):
        repeats = (repeats,)
    return jnp.tile(x, repeats + (1,) * x.ndim)


def repeat_trailing(x: Array, repeats: int | tuple[int, ...]) -> Array:
    """Repeat x along new trailing dimension by repeats."""
    return jnp.moveaxis(repeat_leading(x, repeats), 0, -1)


def pad_along_axis(array: Array, target_length: int, axis: int = 0) -> Array:
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return jnp.pad(array, pad_width=npad, mode="edge")


def pca(X, n_components=2):
    """Principal component analysis."""
    state = pcax.fit(X, n_components=n_components)
    X_pca = pcax.transform(state, X)
    recover_fn = lambda x_pca: pcax.recover(state, x_pca)
    return X_pca, recover_fn


def mirror_pca(bregman, X, n_components=2):
    """Principal component analysis via coordinate change."""
    Y = bregman.to_dual(X)
    Y_pca, dual_recover_fn = pca(Y, n_components=n_components)
    X_pca = bregman.to_primal(Y_pca)
    recover_fn = lambda x_pca: bregman.to_primal(
        dual_recover_fn(bregman.to_dual(x_pca))
    )
    return X_pca, Y_pca, recover_fn, dual_recover_fn


def negentropy(
    distribution: dx.Distribution,
    samples: Optional[Array] = None,
    lp: Optional[Array] = None,
):
    """Negative (Shannon) entropy."""
    try:
        return -1 * distribution.entropy()
    except NotImplementedError:
        assert not (samples is None and lp is None)
        # lp = distribution.log_prob(samples)
        # return jnp.average(lp, weights=jnp.exp(lp))
        if lp is None:
            return distribution.log_prob(samples).mean()
        else:
            return jnp.mean(lp)


def nsentropy_potential(p):
    """Negative Shannon Entropy (potential) defined on the positive orthant."""
    return p * (safe_log(p) - 1)


def nsentropy(p):
    """Negative Shannon Entropy defined on the positive orthant."""
    return jnp.dot(p, safe_log(p) - 1)


def nsentropy_minimal(s):
    """Negative Shannon Entropy on the Simplex (minimal parameterization)."""
    return nsentropy(s_to_p(s))


def logsumexp_minimal(x):
    """Conjugate of nsentropy_minimal."""
    return omu.logsumexp(jnp.pad(x, (1, 0)), axis=0)


def make_simplex_kl():
    return wrap_potentials(nsentropy_minimal, logsumexp_minimal, MinimalALRTransform)


def make_simplex_ilr():
    return MinimalILRTransform


def make_extended_kl(a: float = 1.0):
    return make_separable(
        lambda x: nsentropy_potential(a * x) / (a**2),
        lambda y: jnp.exp(a * y) / (a**2),
        dx.Chain(
            [
                dx.ScalarAffine(shift=0, scale=1 / a),
                dx.Lambda(safe_log, jnp.exp),  # pyright: ignore
                dx.ScalarAffine(shift=0, scale=a),
            ]
        ),
    )


def x_to_means_and_covs(
    x: Array, dimension: int, squeeze: bool = False
) -> tuple[Array, Array]:
    """Extract means and covariance matrices of Gaussians from raveled vector.

    Args:
      x: [num_gaussians, dimension, (1 + dimension)] array of concatenated means
        and covariances (raveled) dimension: the dimension of the Gaussians.
      dimension: Dimensionality of the Gaussians.

    Returns:
      Means and covariances of shape ``[num_gaussian, dimension]``.
    """
    is_1d = x.ndim == 1
    x = jnp.atleast_2d(x)
    leading_shape = x.shape[:-1]
    means = x[..., :dimension]
    covariances = jnp.reshape(
        x[..., dimension : dimension + dimension**2],
        leading_shape + (dimension, dimension),
    )
    if is_1d:
        means = jnp.squeeze(means, axis=0)
        covariances = jnp.squeeze(covariances, axis=0)
    if squeeze:
        means = jnp.squeeze(means)
        covariances = jnp.squeeze(covariances)
    return means, covariances


def gaussian_eflpf(x, dim: int):
    mean, cov = x_to_means_and_covs(x, dim)
    omega = -2 * cov
    return 0.5 * (mean @ jnp.linalg.inv(omega) @ mean - logdet(omega))


def gaussian_eflpf_conjugate(y, dim: int):
    # Page 111 of Hessian structures (Shima 2007)
    mean, cov = x_to_means_and_covs(y, dim)
    rho = cov - jnp.outer(mean, mean)
    return 0.5 * (logdet(rho) - dim)


def gaussian_dual(x, dim: int):
    mean, cov = x_to_means_and_covs(x, dim)
    omega_inv = jnp.linalg.inv(-2 * cov)
    new_mean = omega_inv @ mean
    new_cov = jnp.outer(new_mean, new_mean) + omega_inv
    return mean_and_cov_to_x(new_mean, new_cov, dim)


def gaussian_primal(y, dim: int):
    mean, cov = x_to_means_and_covs(y, dim)
    new_cov = 0.5 * jnp.linalg.inv(
        project_definite(jnp.outer(mean, mean) - cov, pos=False)
    )
    new_cov = 0.5 * jnp.linalg.inv(jnp.outer(mean, mean) - cov)
    new_mean = -2.0 * new_cov @ mean
    return mean_and_cov_to_x(new_mean, new_cov, dim)


class GaussianMirror(dx.Bijector):
    def __init__(self, dim: int):
        self.dim = dim
        self._forward = lambda x: gaussian_dual(x, self.dim)
        self._inverse = lambda y: gaussian_primal(y, self.dim)
        self._jacfwd = jax.jacfwd(self._forward)
        self._jacinv = jax.jacfwd(self._inverse)
        super().__init__(event_ndims_in=1)

    def forward(self, x):
        return self._forward(x)

    def inverse(self, y):
        return self._inverse(y)

    def forward_and_log_det(self, x):
        y = self._forward(x)
        return y, logdet(self._jacfwd(x))

    def inverse_and_log_det(self, y):
        x = self._inverse(y)
        return x, logdet(self._jacinv(y))


def gaussian_natural(x, dim: int):
    """Map canonical parameters to natural parameters."""
    mean, cov = x_to_means_and_covs(x, dim)
    precision = jnp.linalg.inv(cov)
    new_mean = precision @ mean
    new_cov = -0.5 * precision
    return mean_and_cov_to_x(new_mean, new_cov, dim)


def gaussian_canonical(y, dim: int):
    """Map natural parameters to canonical parameters."""
    mean, cov = x_to_means_and_covs(y, dim)
    new_cov = -0.5 * jnp.linalg.inv(cov)
    new_mean = new_cov @ mean
    return mean_and_cov_to_x(new_mean, new_cov, dim)


def gaussian_kl(x1: Float[Array, " d"], x2: Float[Array, " d"], dim: int):
    mu1, sigma1 = x_to_means_and_covs(x1, dim)
    mu2, sigma2 = x_to_means_and_covs(x2, dim)
    precision2 = jnp.linalg.inv(sigma2)
    return 0.5 * (
        logdet(sigma2)
        - logdet(sigma1)
        - dim
        + jnp.trace(precision2 @ sigma1)
        + (mu2 - mu1) @ precision2 @ (mu2 - mu1)
    )


class GaussianNatural(dx.Bijector):
    def __init__(self, dim: int):
        self.dim = dim
        self._forward = lambda x: gaussian_natural(x, self.dim)
        self._inverse = lambda y: gaussian_canonical(y, self.dim)
        self._jacfwd = jax.jacfwd(self._forward)
        self._jacinv = jax.jacfwd(self._inverse)
        super().__init__(event_ndims_in=1)

    def forward(self, x):
        return self._forward(x)

    def inverse(self, y):
        return self._inverse(y)

    def forward_and_log_det(self, x):
        y = self._forward(x)
        return y, logdet(self._jacfwd(x))

    def inverse_and_log_det(self, y):
        x = self._inverse(y)
        return x, logdet(self._jacinv(y))


def make_gaussian_eflpf(dim: int):
    return wrap_potentials(
        lambda x: gaussian_eflpf(x, dim),
        lambda y: gaussian_eflpf_conjugate(y, dim),
        GaussianMirror(dim),
    )


def hnn_potential(x, beta):
    """Bregman potential for Hopfield Neural Network example."""
    return 1 / (2 * beta) * (x * safe_log(x) + (1 - x) * safe_log(1 - x))


def hnn_conjugate(y, beta):
    """Conjugate for Hopfield Neural Network example."""
    return safe_log(1 + jnp.exp(2 * beta * y)) / (2 * beta)


def make_hnn(beta: float = 1.0):
    return make_separable(
        lambda x: hnn_potential(x, beta),
        lambda y: hnn_conjugate(y, beta),
        dx.Chain(
            [
                dx.ScalarAffine(shift=0, scale=1 / (2 * beta)),
                dx.Inverse(dx.Sigmoid()),
            ]
        ),
    )


def make_r_hnn(beta) -> dx.Bijector:
    return dx.Block(
        dx.Lambda(
            lambda x: jnp.arcsin(jnp.sqrt(x)) / beta, lambda y: jnp.sin(beta * y) ** 2
        ),
        ndims=1,
    )


def make_tanh():
    potential = lambda x: x * jnp.arctanh(x) + 0.5 * safe_log(1 - x**2)
    conjugate = lambda y: safe_log(jnp.cosh(y))
    mirror = dx.Inverse(dx.Tanh())
    return wrap_potentials(potential, conjugate, mirror)


def make_r_tanh():
    return dx.Block(dx.Lambda(jnp.arcsin, jnp.sin), ndims=1)


def make_arctan():
    potential = lambda x: -safe_log(jnp.cos(x))
    conjugate = lambda y: y * jnp.arctan(y) - 0.5 * safe_log(1 + y**2)
    mirror = dx.Lambda(jnp.tan, jnp.arctan)
    return wrap_potentials(potential, conjugate, mirror)


def make_r_arctan():
    bijector = dx.Lambda(
        lambda x: safe_log(1 + jnp.tan(x / 2)) - safe_log(1 - jnp.tan(x / 2)),
        lambda y: 2 * jnp.arctan((jnp.exp(y) - 1) / (jnp.exp(y) + 1)),
    )
    return dx.Block(bijector, ndims=1)


def norm(
    x,
    order: None | int | str = None,
    axis: None | int | Sequence[int] = None,
    keepdims: bool = False,
):
    """Norm w/ custom gradients for numerical stability."""
    return omu.norm(x, ord=order, axis=axis, keepdims=keepdims)


def normalize(
    x,
    order: None | int | str = None,
    axis: None | int | Sequence[int] = None,
    eps: None | float = None,
):
    if eps is None:
        eps = jnp.finfo(x.dtype).eps  #  pyright: ignore
    x_norm = norm(x, order=order, axis=axis, keepdims=True)
    return x / jnp.maximum(x_norm, eps)  #  pyright: ignore


def diag_mvn(
    dim: int, loc: Optional[float | Array] = 0.0, scale: Optional[float | Array] = 1.0
) -> dx.Distribution:
    if isinstance(loc, Real) and isinstance(scale, Real):
        return dx.MultivariateNormalDiag(loc * jnp.ones(dim), scale * jnp.ones(dim))
    elif isinstance(loc, Array) and isinstance(scale, Real):
        assert loc.shape[-1] == dim
        return dx.MultivariateNormalDiag(loc, scale * jnp.ones_like(loc))
    elif isinstance(loc, Real) and isinstance(scale, Array):
        assert scale.shape[-1] == dim
        return dx.MultivariateNormalDiag(loc * jnp.ones(dim), scale)
    else:
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)
        assert loc.shape[-1] == scale.shape[-1] == dim
        return dx.MultivariateNormalDiag(loc, scale)


def mvn_from_samples(samples):
    """Estimate MVN from samples."""
    gaussian = Gaussian.from_samples(samples)
    return dx.MultivariateNormalFullCovariance(gaussian.loc, gaussian.covariance())


def isotropic_mvn(
    dim: int, loc: Optional[float | Array] = 0.0, std: float = 1.0
) -> dx.Distribution:
    return diag_mvn(dim, loc, std)


def symmetric_dirichlet(dim: int, alpha: float = 1.0, minimal=False) -> dx.Distribution:
    dist = dx.Dirichlet(jnp.ones(dim) * alpha)
    if minimal:
        return MinimalSimplex(dist)
    return dist


def uniform(
    dim: int, low: float | Array = 0.0, high: float | Array = 1.0
) -> dx.Distribution:
    if isinstance(low, Array):
        if low.size == 1:
            low = low.item()
        elif low.shape[-1] != dim:
            low = repeat_trailing(low, dim).squeeze()
    if isinstance(high, Array):
        if high.size == 1:
            high = high.item()
        elif high.shape[-1] != dim:
            high = repeat_trailing(high, dim).squeeze()
    if isinstance(low, float):
        low = jnp.ones(dim) * low
    if isinstance(high, float):
        high = jnp.ones(dim) * high
    return dx.Independent(dx.Uniform(low, high), 1)


def diag_logistic_mvn(
    dim: int,
    loc: Optional[float | Array] = 0.0,
    scale: Optional[float | Array] = 1.0,
    tranform: Literal["alr", "ilr"] = "alr",
) -> dx.Distribution:
    mvn = diag_mvn(dim, loc, scale)
    return SimplexFromPlane(mvn, tranform, minimal=False)


def mixture_mvn(
    key,
    dim: int,
    n_components: int,
    span: tuple[float, float] | float = 5.0,
    scale: float = 1.0,
    debug: bool = False,
) -> dx.MixtureSameFamily:
    """Create a random mixture of Gaussians with n_components components."""
    if not isinstance(span, tuple):
        span = (-abs(span), abs(span))
    locs = jr.uniform(key, (n_components, dim), minval=span[0], maxval=span[1])
    scales = scale * jnp.ones_like(locs)
    comp = dx.MultivariateNormalDiag(locs, scales)
    if debug:
        print("Mean: ", comp.mean())
        print("Covariance: ", comp.covariance())
    mix = dx.Categorical(jnp.ones(n_components))
    return dx.MixtureSameFamily(mix, comp)


def mixture_mvsn(
    key,
    dim: int,
    n_components: int,
    span: tuple[float, float] | float = 5.0,
    scale: float = 1.0,
    skew: tuple[float, float] | float = 0.0,
    debug: bool = False,
) -> dx.MixtureSameFamily:
    """Create a random mixture of Skewed Gaussians with n_components components."""
    if not isinstance(span, tuple):
        span = (-abs(span), abs(span))
    if not isinstance(skew, tuple):
        skew = (-abs(skew), abs(skew))

    mog = mixture_mvn(key, dim, n_components, span, scale, debug=debug)
    if skew == (0, 0):
        return mog
    else:
        alpha = jr.uniform(key, (n_components, dim), minval=skew[0], maxval=skew[1])
        if debug:
            print("Skewness: ", alpha)
        comp = MultivariateSkewNormal(mog.components_distribution, alpha)
        return dx.MixtureSameFamily(mog.mixture_distribution, comp)


def mixture_lognormal(key, dim: int, *args, **kwargs) -> dx.Distribution:
    mog = mixture_mvsn(key, dim, *args, **kwargs)
    transform = dx.Block(dx.Lambda(jnp.exp, jnp.log), ndims=1)
    return dx.Transformed(mog, transform)


def mvn_dplr(
    key,
    dim: int,
    loc_low: float = 0.0,
    loc_high: float = 1.0,
    lr_low: float = 0.0,
    lr_high: float = 1.0,
    lr_scale: float = 1.0,
    sigma_low: float = 0.0,
    sigma_high: float = 1.0,
    debug: bool = False,
) -> dx.Distribution:
    """Create Low rank + Diagonal Gaussian."""
    loc = jr.uniform(key, (dim,), minval=loc_low, maxval=loc_high)
    scale_diag = jr.uniform(key, (dim,), minval=sigma_low, maxval=sigma_high)
    scale_u_matrix = lr_scale * jr.uniform(key, (dim, 1), minval=lr_low, maxval=lr_high)
    if debug:
        print("Loc: ", loc)
        print("Scale_diag: ", scale_diag)
        print("Scale_u_matrix: ", scale_u_matrix)
    return dx.MultivariateNormalDiagPlusLowRank(loc, scale_diag, scale_u_matrix)


def dro_base(key, dim: int, kind: str, debug: bool = False) -> dx.Distribution:
    """Create base distribution for the Kelly Wasserstein problem.
    There are two types: jump-diffusion like, and heavy-tailed.
    """
    assert kind in ("jump", "heavy")
    if kind == "jump":
        normal = mvn_dplr(
            key,
            dim,
            loc_low=-0.1,
            loc_high=0.1,
            lr_low=-0.5,
            lr_high=1.5,
            lr_scale=0.3,
            sigma_low=0.3,
            sigma_high=0.5,
            debug=debug,
        )
        crash = mvn_dplr(
            key,
            dim,
            loc_low=-1.0,
            loc_high=-0.5,
            lr_low=0.5,
            lr_high=1.5,
            lr_scale=0.9,
            sigma_low=0.1,
            sigma_high=0.3,
            debug=debug,
        )
        return dx.MixtureOfTwo(0.95, normal, crash)
    else:
        raise NotImplementedError


def mixture_logistic_mvsn(
    key,
    dim: int,
    n_components: int,
    span: tuple[float, float] | float = 5.0,
    scale: float = 1.0,
    skew: tuple[float, float] | float = 0.0,
    transform: Literal["alr", "ilr"] = "alr",
    minimal: bool = False,
) -> dx.Distribution:
    """Create a random mixture of Logistic MVNs with n_components components."""
    mixture = mixture_mvsn(key, dim - 1, n_components, span, scale, skew)
    return SimplexFromPlane(mixture, transform, minimal=minimal)


def mixture_dirichlet(
    key,
    dim: int,
    n_components: int,
    scale: float = 5.0,
    minimal: bool = False,
    debug: bool = False,
) -> dx.MixtureSameFamily:
    """Create a random mixture of Gaussians with n_components components."""
    locs = jr.dirichlet(key, jnp.ones((dim,)), (n_components,))
    alphas = jnp.clip(scale * locs, 1.0, 5.0)
    comp = dx.Dirichlet(alphas)
    if debug:
        print("Locs: ", locs)
        print("Concentration: ", comp.concentration)
    mix = dx.Categorical(jnp.ones(n_components))
    mod = dx.MixtureSameFamily(mix, comp)
    if minimal:
        return MinimalSimplex(mod)
    else:
        return mod


def symmetric_beta(
    dim: int, alpha: float = 2.0, a: float = 0.0, b: float = 1.0
) -> dx.Distribution:
    assert a < b
    base = dx.Independent(dx.Beta(alpha * jnp.ones(dim), alpha * jnp.ones(dim)), 1)
    transform = dx.Block(dx.ScalarAffine(shift=a, scale=b - a), ndims=1)
    return dx.Transformed(base, transform)


def mixture_beta(
    key,
    dim: int,
    n_components: int | tuple[int, ...],
    alpha_min: float = 1.0,
    alpha_max: float = 5.0,
    beta_min: float = 1.0,
    beta_max: float = 5.0,
    a: float = 0.0,
    b: float = 1.0,
):
    assert a < b
    k1, k2 = jr.split(key)
    if isinstance(n_components, int):
        n_components = (n_components,)
    mixture = dx.MixtureSameFamily(
        dx.Categorical(jnp.zeros(n_components)),
        dx.Independent(
            dx.Beta(
                alpha=jr.uniform(
                    k1, n_components + (dim,), minval=alpha_min, maxval=alpha_max
                ),
                beta=jr.uniform(
                    k2, n_components + (dim,), minval=beta_min, maxval=beta_max
                ),
            ),
            reinterpreted_batch_ndims=1,
        ),
    )
    transform = dx.Block(dx.ScalarAffine(shift=a, scale=b - a), ndims=1)
    return dx.Transformed(mixture, transform)


def mixture_gamma(
    key,
    dim: int,
    n_components: int | tuple[int, ...],
    alpha_min: float = 1.0,
    alpha_max: float = 5.0,
    theta_min: float = 1.0,
    theta_max: float = 5.0,
):
    k1, k2 = jr.split(key)
    if isinstance(n_components, int):
        n_components = (n_components,)
    mixture = dx.MixtureSameFamily(
        dx.Categorical(jnp.zeros(n_components)),
        dx.Independent(
            dx.Gamma(
                concentration=jr.uniform(
                    k1, n_components + (dim,), minval=alpha_min, maxval=alpha_max
                ),
                rate=jnp.reciprocal(
                    jr.uniform(
                        k2, n_components + (dim,), minval=theta_min, maxval=theta_max
                    ),
                ),
            ),
            reinterpreted_batch_ndims=1,
        ),
    )
    return mixture


def make_lr_schedule(steps_and_lr: list[Sequence], step_scale: int) -> Any:
    steps, lr = zip(*steps_and_lr)
    scales = np.cumprod(lr[1:]) / np.cumprod(lr[:-1])
    boundaries = step_scale * np.cumprod(steps)[1:]
    boundaries_and_scales = dict(zip(boundaries, scales))
    init_value = lr[0]
    return sum(steps), optax.piecewise_constant_schedule(
        init_value, boundaries_and_scales
    )


def safe_clip(x, a_min=None, a_max=None, eps=None):
    if a_min is None and a_max is None:
        return x
    if eps is None:
        eps = jnp.finfo(x.dtype).eps
    if a_min is not None:
        a_min = a_min + eps
    if a_max is not None:
        a_max = a_max - eps
    return jnp.clip(x, a_min, a_max)


def safe_log(x: Array, *, eps: Optional[float] = None) -> Array:
    """Double where trick to avoid NaNs in backward pass.
    https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
    """
    tiny = jnp.finfo(x.dtype).tiny
    if eps is None:
        eps = tiny
    x_clip = jnp.where(x > 0.0, x, 1.0)
    return jnp.where(x > tiny, jnp.log(x_clip), jnp.log(eps))  # pyright: ignore


def safe_exp(x: Array) -> Array:
    large = jnp.finfo(x.dtype).max
    return jnp.where(x < jnp.floor(jnp.log(large)), jnp.exp(x), large)  # pyright: ignore


@jax.custom_jvp
def lambertw(z: Array) -> Array:
    # NOTE: Probably want to use omu.lambertw instead for GPUs.
    # TODO(michalk8): use tfp
    result_shape = jax.ShapeDtypeStruct(z.shape, z.dtype)
    return jax.pure_callback(  # pyright: ignore
        lambda z: scipy.special.lambertw(z).real.astype(z.dtype),
        result_shape,
        z,
        vectorized=True,
    )


@lambertw.defjvp
def lambertw_jvp(primals, tangents):
    (z,) = primals
    (z_dot,) = tangents
    primal_out = lambertw(z)
    # https://en.wikipedia.org/wiki/Lambert_W_function#Derivative
    tangent_out = jnp.reciprocal(z + jnp.exp(primal_out)) * z_dot
    return primal_out, tangent_out


def deformed_log(x, a: float, b: float):
    """Compute the deformed (a, b) logarithm function.
    This generalizes many common deformations of the logarithm.
    See http://arxiv.org/abs/2304.01941v1 for details.
    See also https://arxiv.org/abs/0907.4067.

    Originally introduced in:
    E.P. Borges and I. Roditi. A family of nonextensive entropies.
    Physics Letters A, 246(5):399–402, 1998.

    a and b must satisfy either
    - 0 < a <= 1 <= b, or
    - 0 < b <= 1 <= a.
    """
    if a == b:
        return safe_log(x)
    return (x ** (a - 1) - x ** (b - 1)) / (a - b)


def tsallis_log(x, t: float):
    """Compute the Tsallis logarithm."""
    return deformed_log(x, 2 - t, 1)


def tsallis_exp(x, t: float):
    """Compute the Tsallis exponential."""
    eps = jnp.finfo(x.dtype).eps if t > 1 else 0
    if t == 1:
        return jnp.exp(x)
    return jnp.maximum(eps, 1 + (1 - t) * x) ** (1 / (1 - t))


def tsallis_clog(x, t: float):
    """Compute the complete Tsallis logarithm.
    See Equation (15) of https://arxiv.org/abs/0907.4067.
    """
    if t == 1:
        return safe_log(x)
    return jnp.where(x <= 1, tsallis_log(x, 2 - t), tsallis_log(x, t))


def tsallis_cexp(x, t: float):
    """Compute the complete Tsallis exponential.
    See Equation (15) of https://arxiv.org/abs/0907.4067.
    """
    if t == 1:
        return jnp.exp(x)
    return jnp.where(x <= 0, tsallis_exp(x, 2 - t), tsallis_exp(x, t))


def kaniadakis_log(x, k: float):
    """Compute the Kaniadakis logarithm function.
    k should be in the range [-1, 1].
    """
    return deformed_log(x, 1 + k, 1 - k)


def kaniadakis_exp(x, k: float):
    """Compute the Kaniadakis exponential function.
    k should be in the range [-1, 1].
    """
    if k == 0:
        return jnp.exp(x)
    return (k * x + jnp.sqrt(1 + (k * x) ** 2)) ** (1 / k)


def gamma_log(x, g: float):
    """Compute the gamma logarithm function.
    g should be in the range [-1/2, 1/2].
    """
    return deformed_log(x, 1 + 2 * g, 1 - g)


def gamma_exp(x, g: float):
    """Compute the gamma exponential function.
    g should be in the range [-1/2, 1/2].
    See https://arxiv.org/abs/cond-mat/0409683
    """
    if g == 0:
        return jnp.exp(x)
    c = jnp.sqrt(1 - 4 * (g * x) ** 3)
    return jnp.float_power(jnp.cbrt((1 + c) / 2) + jnp.cbrt((1 - c) / 2), 1 / g)


def newton_log(x):
    """Compute the Newton logarithm function."""
    return 0.5 * (safe_log(x) + x - 1)


def newton_exp(x):
    """Compute the Newton exponential function."""
    return omu.lambertw(jnp.exp(2 * x + 1))


# From ott.math.utils
# TODO(michalk8): add axis argument
def kl(p: Array, q: Array) -> Array:
    """Kullback-Leibler divergence."""
    return jnp.vdot(p, (safe_log(p) - safe_log(q)))


def gen_kl(p: Array, q: Array) -> Array:
    """Generalized Kullback-Leibler divergence."""
    return jnp.vdot(p, (safe_log(p) - safe_log(q))) - jnp.sum(p) + jnp.sum(q)


def kl_minimal(pb: Array, qb: Array) -> Array:
    """Compute the Kullback-Leibler divergence on minimal parameterization
    of the Simplex."""
    return kl(s_to_p(pb), s_to_p(qb))


def helmert(n, full=False):
    """
    Create an Helmert matrix of order `n`.

    This has applications in statistics, compositional or simplicial analysis,
    and in Aitchison geometry.

    Parameters
    ----------
    n : int
        The size of the array to create.
    full : bool, optional
        If True the (n, n) ndarray will be returned.
        Otherwise the submatrix that does not include the first
        row will be returned.
        Default: False.

    Returns
    -------
    M : ndarray
        The Helmert matrix.
        The shape is (n, n) or (n-1, n) depending on the `full` argument.

    Examples
    --------
    >>> result = helmert(5, full=True)
    >>> from scipy.linalg import helmert as scipy_helmert
    >>> assert jnp.allclose(result, scipy_helmert(5, full=True))
    """
    H = jnp.tril(jnp.ones((n, n)), -1) - jnp.diag(jnp.arange(n))
    d = jnp.arange(n) * jnp.arange(1, n + 1)
    H = H.at[0].set(1)
    d = d.at[0].set(n)
    H_full = H / jnp.sqrt(d)[:, jnp.newaxis]
    if full:
        return H_full
    else:
        return H_full[1:]


def s_to_p(s):
    p0 = 1 - jnp.sum(s, axis=-1, keepdims=True)
    # p0 = safe_clip(p0, 0.0, 1.0, 1e-6)
    return jnp.concatenate([p0, s], axis=-1)


def p_to_s(p):
    return p[..., 1:]


# Aitchison Geometry and Algebra
def closure(p):
    return p / jnp.sum(p, axis=-1, keepdims=True)


def perturb(p, q):
    """Add p and q under Aitchison geometry."""
    return closure(p * q)


def power(p, alpha):
    """Power (also known as escort) transform."""
    return closure(p**alpha)


def inverse(p):
    """Inverse transform."""
    return power(p, -1)


def subtract(p, q):
    """Subtract q from p under Aitchison geometry."""
    return perturb(p, inverse(q))


escort = power


def aitchison_dot(p, q, temp=TEMP):
    """Aitchison inner product."""
    return jnp.dot(clr(p, temp), clr(q, temp))


def aitchison_norm(p, temp=TEMP):
    """Aitchison norm."""
    # return jnp.sqrt(aitchison_dot(p, p, temp))
    # better to avoid NaN gradients
    return omu.norm(clr(p, temp), ord=2, axis=-1)


def aitchison_distance(p, q, temp=TEMP):
    """Aitchison distance."""
    return aitchison_norm(subtract(p, q), temp)


def aitchison_sqdist(p, q, temp=TEMP):
    """Squared Aitchison distance."""
    return aitchison_dot(subtract(p, q), subtract(p, q), temp)


def alr(p, temp=TEMP):
    """Additive log-ratio transform."""
    # return temp * jnp.log(p[..., 1:] / p[..., :1])
    p0 = safe_clip(p[..., :1], 0.0, 1.0)
    raw = temp * safe_log(p[..., 1:] / p0)
    return raw
    # inf = SCALE * temp
    # clipped = jnp.clip(raw, -inf, inf)
    # return clipped
    # return clipped + lax.stop_gradient(raw - clipped)


def alr_inverse(x, temp=TEMP):
    """Inverse of the ALR transform."""
    padding = ((0, 0),) * (x.ndim - 1) + ((1, 0),)
    return jnn.softmax(jnp.pad(x, padding) / temp)


def alr_logdet(p, temp=TEMP):
    """Log determinant of the Jacobian of the ALR transform.
    This is done using the minimal parameterization.
    """
    real_dim = p.shape[-1] - 1
    constant = real_dim * jnp.log(jnp.abs(temp))
    # return -safe_log(jnp.prod(p, axis=-1)) + constant
    return -jnp.sum(safe_log(p), axis=-1) + constant


def alr_inverse_logdet(x, temp=TEMP):
    """Log determinant of the Jacobian of the inverse of the ALR transform."""
    return -alr_logdet(alr_inverse(x, temp), temp)


def tstar(t: float):
    return 1 / (2 - t)


def tempered_s_to_p(s, t: float = 1.0):
    """Co-density to density transform."""
    return s_to_p(s ** (1 / tstar(t)))


def tempered_p_to_s(p, t: float = 1.0):
    """Density to co-density transform."""
    return p_to_s(p ** tstar(t))


def extended_alr(s, t: float = 1.0):
    """Extended ALR (tempered) transform from https://arxiv.org/abs/2311.13459."""
    ts = tstar(t)
    s0 = (1 - jnp.sum(s ** (1 / ts), axis=-1, keepdims=True)) ** ts
    return tsallis_log(s / s0, t)


def extended_alr_inverse(x, t: float = 1.0):
    """Inverse of the extended ALR (tempered) transform."""
    ts = tstar(t)
    normalizer = 1 + jnp.sum(tsallis_exp(x, t) ** (1 / ts), axis=-1, keepdims=True)
    return tsallis_exp(x, t) / normalizer**ts


def tempered_alr(p, t: float = 1.0):
    """Tempered ALR (tempered) transform from https://arxiv.org/abs/2311.13459.
    From the Simplex to the domain of the Tsallis exponential.
    """
    return extended_alr(tempered_p_to_s(p, t), t)


def tempered_alr_inverse(x, t: float = 1.0):
    """Inverse of the tempered ALR (tempered) transform."""
    return tempered_s_to_p(extended_alr_inverse(x, t))


def _A(d):
    """(d-1) x d matrix."""
    return (
        jnp.block([jnp.zeros((d - 1, 1)), d * jnp.eye(d - 1)]) - jnp.ones((d - 1, d))
    ) / d


def _U(d):
    """(d-1) x d matrix."""
    return helmert(d)


def _Upi(d):
    """Pseudo-inverse of _U(d) - d x (d-1) matrix."""
    return _U(d).T


def clr(p, temp=TEMP):
    """Centered log-ratio transform."""
    # see relationship between clr and alr in Egozcue et al. (2003)
    # return jnp.log(p) -  jnp.mean(jnp.log(p), axis=-1, keepdims=True)
    return alr(p, temp) @ _A(p.shape[-1])


def clr_inverse(x, temp=TEMP):
    """Inverse of the clr transform."""
    return jnn.softmax(x / temp, axis=-1)


def ilr(p, temp=TEMP):
    """Isometric log-ratio transform.

    >>> rng = jr.key(0)
    >>> P = jr.dirichlet(rng, 5 * jnp.ones((3,)), (5,))
    >>> assert jnp.allclose(P, ilr_inverse(ilr(P)))
    """
    return clr(p, temp) @ _Upi(p.shape[-1])


def ilr_inverse(x, temp=TEMP):
    """Inverse of the ilr transform.

    >>> rng = jr.key(0)
    >>> X = jr.normal(rng, (5, 2))
    >>> assert jnp.allclose(X, ilr(ilr_inverse(X)))
    """
    return clr_inverse(x @ _U(x.shape[-1] + 1), temp)


def ilr_logdet(p, temp=TEMP):
    """Log determinant of the Jacobian of the ILR transform.

    >>> rng = jr.key(0)
    >>> P = jr.dirichlet(rng, 5 * jnp.ones((3,)), (5,))
    >>> jac_fn = lambda p: jax.jacobian(lambda s: ilr(s_to_p(s)))(p_to_s(p))
    >>> logdet_fn = jax.vmap(lambda p: logdet(jac_fn(p)))
    >>> assert jnp.allclose(logdet_fn(P), ilr_logdet(P))
    """
    d = p.shape[-1]
    # return alr_logdet(p, temp) + logdet(_A(d) @ _Upi(d))
    return alr_logdet(p, temp) - 0.5 * jnp.log(d)


def ilr_inverse_logdet(x, temp=TEMP):
    """Log determinant of the Jacobian of the inverse ILR transform.

    >>> rng = jr.key(0)
    >>> X = jr.normal(rng, (5, 2))
    >>> jac_fn = lambda x: jax.jacobian(lambda xi: p_to_s(ilr_inverse(xi)))(x)
    >>> logdet_fn = jax.vmap(lambda x: logdet(jac_fn(x)))
    >>> assert jnp.allclose(logdet_fn(X), ilr_inverse_logdet(X))
    """
    return -ilr_logdet(ilr_inverse(x, temp), temp)


def _get_diag_bijector(mvn: dx.MultivariateNormalFromBijector) -> dx.Bijector:
    loc, scale = mvn.loc, mvn.scale
    sqrt_diag = jnp.vectorize(
        lambda cov: jnp.sqrt(jnp.diag(cov)), signature="(k,k)->(k)"
    )(mvn.covariance())
    scale = dx.DiagLinear(sqrt_diag)
    bijector = dx.Chain([dx.Block(dx.Shift(loc), ndims=1), scale])
    return bijector


class PotentialDistribution(dx.Distribution):
    """Unnormalized distribution generated by convex potential V."""

    def __init__(
        self, potential: Callable[[Array], Array], dim: int, beta: float = 1.0
    ):
        """
        Args:
            potential: potential function
            dim: dimension of the space
            beta: temperature
        """
        self._potential = potential
        self._dim = dim
        self._beta = beta

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._dim,)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.float32

    def log_prob(self, value: Array) -> Array:
        return -self._beta * self._potential(value)


class KDE(dx.Distribution):
    def __init__(
        self,
        samples: Float[Array, "num_samples dim"],
        bw_method=None,
        weights=None,
    ):
        samples = jnp.squeeze(samples)
        if samples.ndim == 1:
            self._batch_shape = ()
            self._event_shape = ()
        else:
            self._batch_shape = samples.shape[:-1]
            self._event_shape = samples.shape[-1:]
        self._dtype = samples.dtype
        self.kde = jst.gaussian_kde(samples.T, bw_method=bw_method, weights=weights)
        # self._samples = samples

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def bandwidth(self):
        return jnp.sqrt(jnp.diag(self.kde.covariance).squeeze())

    def log_prob(self, value: Array) -> Scalar:
        return self.kde.logpdf(value).squeeze()

    def _sample_n(self, key: PRNGKeyArray, n: int) -> Array:
        samples = self.kde.resample(key, (n,))
        return jnp.moveaxis(samples, 0, -1).squeeze(axis=-1)


class MultivariateSkewNormal(dx.Distribution):
    """Multivariate skew normal on R^k."""

    def __init__(
        self,
        mvn: dx.MultivariateNormalFromBijector,
        shape: jax.Array,
    ):
        self._mvn = mvn
        self._batch_shape = jnp.broadcast_shapes(mvn.batch_shape, shape.shape[:-1])
        self._shape = jnp.broadcast_to(shape, self._batch_shape + self._mvn.event_shape)
        self._is_diag = isinstance(mvn, dx.MultivariateNormalDiag)
        self._covz = None
        self._delta_fn = jnp.vectorize(
            lambda s: jnp.reciprocal(jnp.sqrt(1 + s @ s)) * s, signature="(k)->(k)"
        )
        if not self._is_diag:
            self._covz = jnp.vectorize(
                lambda istd, cov: jnp.diag(istd) @ cov @ jnp.diag(istd),
                signature="(k),(k,k)->(k,k)",
            )(jnp.reciprocal(mvn.stddev()), mvn.covariance())
            self._delta_fn_double = jnp.vectorize(
                lambda s, covz: jnp.reciprocal(jnp.sqrt(1 + s @ covz @ s)) * (covz @ s),
                signature="(k),(k,k)->(k)",
            )
            self._delta_fn = lambda s: self._delta_fn_double(s, self._covz)

        self._bijector = _get_diag_bijector(mvn)
        # self._bijector = mvn.bijector
        self._from_z = self._bijector.forward
        self._to_z = self._bijector.inverse
        self._snorm = dx.Normal(0, 1)
        self._log_cdf = jnp.vectorize(
            lambda z, shape: self._snorm.log_cdf(z @ shape), signature="(k),(k)->()"
        )

    @property
    def event_shape(self):
        return self._mvn.event_shape

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def _covstar_fn(self):
        if not self._is_diag:

            def _double(delta, covz):
                return jnp.block(
                    [
                        [jnp.ones((1,)), delta],
                        [delta[:, None], covz],
                    ]
                )

            return lambda delta: jnp.vectorize(_double, signature="(k),(k,k)->(p,p)")(
                delta, self._covz
            )
        else:

            def _single(delta):
                return jnp.block(
                    [
                        [jnp.ones((1,)), delta],
                        [delta[:, None], jnp.eye(delta.size)],
                    ]
                )

            return jnp.vectorize(_single, signature="(k)->(p,p)")

    @property
    def delta(self):
        return self._delta_fn(self._shape)

    def log_prob(self, value) -> jax.Array:
        z = self._to_z(value)
        return jnp.log(2) + self._mvn.log_prob(value) + self._log_cdf(z, self._shape)

    def _reparam_sample(self, key, n: int):
        Z = dx.MultivariateNormalFullCovariance(
            covariance_matrix=self._covstar_fn(self.delta)
        ).sample(seed=key, sample_shape=(n,))
        X_0, X = Z[..., :1], Z[..., 1:]
        X = jnp.where(X_0 <= 0, -X, X)
        return jax.vmap(self._from_z)(X)

    def _sample_n(self, key, n: int):
        return self._reparam_sample(key, n)


class GaussianCopula(dx.Distribution):
    """Gaussian copula."""

    def __init__(self, corr: Array):
        def _project(cov):
            var = jnp.reciprocal(jnp.sqrt(jnp.diag(cov)))
            return jnp.diag(var) @ cov @ jnp.diag(var)

        self._corr = jnp.vectorize(_project, signature="(k,k)->(k,k)")(corr)
        self._mvn = dx.MultivariateNormalFullCovariance(covariance_matrix=self._corr)
        self._uvn = jst.norm

    @property
    def mvn(self):
        return self._mvn

    @property
    def uvn(self):
        return self._uvn

    @property
    def event_shape(self):
        return self._mvn.event_shape

    @property
    def batch_shape(self):
        return self._mvn.batch_shape

    @property
    def in_shape(self):
        return self.batch_shape + self.event_shape

    def log_prob(self, value) -> jax.Array:
        value_batch = value.shape[: -len(self.in_shape)]
        value = jnp.broadcast_to(value, value_batch + self.in_shape)
        ppf = self._uvn.ppf(value)
        mvn_logpdf = self._mvn.log_prob(ppf)
        uvn_logpdf = self._uvn.logpdf(ppf)
        return mvn_logpdf - jnp.sum(uvn_logpdf, axis=-1)  # pyright: ignore

    def _sample_n(self, key, n: int):
        return self._uvn.cdf(self._mvn.sample(seed=key, sample_shape=(n,)))


def project_simplex(x: Array) -> Array:
    """Project x onto the minimal Simplex."""
    return jpr.projection_l1_ball(jpr.projection_non_negative(x))


def project_definite(A: Array, pos: bool = True) -> Array:
    """Project A onto the set of positive/negative definite matrices.
    Precondition: A is symmetric.
    """
    eigvals, eigvecs = jnp.linalg.eigh(A)
    if pos:
        eigvals = safe_clip(eigvals, a_min=0.0)
    else:
        eigvals = safe_clip(eigvals, a_max=0.0)
    return eigvecs @ jnp.diag(eigvals) @ eigvecs.T


class SLBTransform(dx.Bijector):
    """Map Simplex S^{dim-1} in R^{dim} to Euclidean R^{dim} via the
    Simplex log-barrier (see https://openreview.net/forum?id=vh7qBSDZW3G E.2).
    NOTE: In contrast to the ALRTransform, this bijector operates directly
    on the minimal parameterization of the simplex.
    """

    def __init__(self, concentration: Optional[Array] = None, root: bool = True):
        """
        Args:
            concentration: The concentration parameter of the simplex
            root: Whether to use fixed point iteration or root finding
        """
        if concentration is not None:
            concentration = jnp.atleast_1d(concentration)
            assert concentration.ndim == 1
            self._dim = concentration.size - 1  # real simplex dimension
            self._a0 = concentration[0]
            self._concentration = concentration[1:]
        else:
            self._dim = None
            self._a0 = 1.0
            self._concentration = None
        # run fixed point iteration (False) or root finding (True)
        self._root = root
        super().__init__(event_ndims_in=1)

    @property
    def a0(self):
        return self._a0

    def concentration(self, value):
        self._check_dim(value)
        a = self._concentration
        if a is None:
            a = jnp.ones_like(value)
        return a

    def _check_dim(self, value):
        assert value.ndim == 1
        if self._dim is not None:
            assert value.size == self._dim

    def _fldj(self, x, a, x0):
        return jnp.log1p(self.a0 / x0**2 * jnp.sum(x**2 / a)) + jnp.sum(
            safe_log(a) - 2 * safe_log(x)  # pyright: ignore
        )

    def _fldj_x0_a(self, x):
        a = self.concentration(x)
        x0 = 1 - jnp.sum(x)
        fldj = self._fldj(x, a, x0)
        return fldj, x0, a

    @property
    def _fxpt_func(self):
        """The fixed point function satisfied by x0."""

        def T(x, theta):
            a = self.concentration(theta)
            return jnp.reciprocal(1 + jnp.sum(a / (self.a0 - x * theta)))

        return T

    def init_x0(self, y, c=0.0):
        return c * jnp.ones_like(y, shape=())

    def _fxpt_x0(self, y):
        self._check_dim(y)
        x0 = (
            jaxopt.AndersonAcceleration(
                fixed_point_fun=self._fxpt_func,
                history_size=5,
                beta=1.0,
                maxiter=100,
                tol=1e-5,
                ridge=1e-5,
            )
            .run(self.init_x0(y), y)
            .params
        )
        return x0

    def _root_func(self, squared: bool = False, inverse: bool = False):
        def F(x, theta):
            if inverse:
                residual = jnp.reciprocal(self._fxpt_func(x, theta)) - jnp.reciprocal(x)
            else:
                residual = self._fxpt_func(x, theta) - x
            if squared:
                residual = residual**2
            return residual

        return F

    def _root_x0(self, y):
        self._check_dim(y)
        x0 = (
            jaxopt.Broyden(
                fun=self._root_func(squared=False, inverse=False),
                maxiter=100,
                tol=1e-5,
                condition="goldstein",
                maxls=20,
            )
            .run(self.init_x0(y), y)
            .params
        )
        return x0

    def forward_and_log_det(self, x):
        fldj, x0, a = self._fldj_x0_a(x)
        y = self.a0 / x0 - a / x  # pyright: ignore
        return y, fldj

    def inverse_and_log_det(self, y):
        if self._root:
            x0 = self._root_x0(y)
        else:
            x0 = self._fxpt_x0(y)
        x0 = safe_clip(x0, 0, 1)
        a = self.concentration(y)
        x = project_simplex(x0 * a / (self.a0 - x0 * y))
        ildj = -self._fldj(x, a, x0)
        return x, ildj


class ALRTransform(dx.Lambda):
    """Map Simplex S^{dim-1} in R^{dim} to Euclidean R^{dim-1} via ALR."""

    def __init__(self, temp=TEMP):
        super().__init__(
            forward=partial(alr, temp=temp),
            inverse=partial(alr_inverse, temp=temp),
            forward_log_det_jacobian=partial(alr_logdet, temp=temp),
            inverse_log_det_jacobian=partial(alr_inverse_logdet, temp=temp),
            event_ndims_in=1,
            # event_ndims_out=1,
            is_constant_jacobian=False,
        )
        self.temp = temp


class CLRTransform(dx.Lambda):
    """Map Simplex S^{dim-1} in R^{dim} to Euclidean R^{dim} via CLR."""

    def __init__(self, temp=TEMP):
        super().__init__(
            forward=partial(clr, temp=temp),
            inverse=partial(clr_inverse, temp=temp),
            forward_log_det_jacobian=partial(alr_logdet, temp=temp),
            inverse_log_det_jacobian=partial(alr_inverse_logdet, temp=temp),
            event_ndims_in=1,
            # event_ndims_out=1,
            is_constant_jacobian=False,
        )
        self.temp = temp


class ILRTransform(dx.Lambda):
    """Map Simplex S^{dim-1} in R^{dim} to Euclidean R^{dim-1} via ILR."""

    def __init__(self, temp=TEMP):
        super().__init__(
            forward=partial(ilr, temp=temp),
            inverse=partial(ilr_inverse, temp=temp),
            forward_log_det_jacobian=partial(ilr_logdet, temp=temp),
            inverse_log_det_jacobian=partial(ilr_inverse_logdet, temp=temp),
            event_ndims_in=1,
            # event_ndims_out=1,
            is_constant_jacobian=False,
        )
        self.temp = temp


class SimplexToBall(dx.Lambda):
    """Map Simplex S^{dim-1} in R^{dim} to Euclidean R^{dim-1} by
    dropping the first coordinate."""

    def __init__(self):
        super().__init__(
            forward=p_to_s,
            inverse=s_to_p,
            forward_log_det_jacobian=lambda p: jnp.zeros_like(p, shape=p.shape[:-1]),
            inverse_log_det_jacobian=lambda s: jnp.zeros_like(s, shape=s.shape[:-1]),
            event_ndims_in=1,
            # event_ndims_out=1,
            is_constant_jacobian=True,
        )


MinimalALRTransform = dx.Chain([ALRTransform(), dx.Inverse(SimplexToBall())])
MinimalILRTransform = dx.Chain([ILRTransform(), dx.Inverse(SimplexToBall())])


class SimplexFromPlane(dx.Transformed):
    """Simplex distribution induced by the push-forward of a Plane distribution."""

    def __init__(
        self,
        base: dx.Distribution,
        transform: Literal["alr", "ilr"] = "alr",
        minimal: bool = False,
        temp: float = 1.0,
    ):
        assert transform in ("alr", "ilr")
        bijector = dx.Inverse(
            ALRTransform(temp) if transform == "alr" else ILRTransform(temp)
        )
        if minimal:
            bijector = dx.Chain([SimplexToBall(), bijector])
        super().__init__(base, bijector)


class PlaneFromSimplex(dx.Transformed):
    """Plane distribution induced by the push-forward of a Simplex distribution."""

    def __init__(
        self,
        base: dx.Distribution,
        transform: Literal["alr", "ilr"] = "alr",
        minimal: bool = False,
        temp: float = 1.0,
    ):
        assert transform in ("alr", "ilr")
        bijector = ALRTransform(temp) if transform == "alr" else ILRTransform(temp)
        if minimal:
            bijector = dx.Chain([bijector, dx.Inverse(SimplexToBall())])
        super().__init__(base, bijector)


class MinimalSimplex(dx.Transformed):
    """Produce a distribution on the minimal parameterization of the Simplex
    by simply dropping the first coordinate."""

    def __init__(self, simplex: dx.Distribution):
        super().__init__(simplex, SimplexToBall())


class OverparamSimplex(dx.Transformed):
    """Produce a distribution on the over parameterization of the Simplex
    by concatenating a new coordinate given by 1 - sum_i x_i."""

    def __init__(self, minimal: dx.Distribution):
        super().__init__(minimal, dx.Inverse(SimplexToBall()))


class LogisticSkewNormal(SimplexFromPlane):
    """Logistic Skew-Normal distribution on the Simplex."""

    def __init__(
        self,
        mvn: dx.MultivariateNormalFromBijector,
        shape: jax.Array,
        transform: Literal["alr", "ilr"] = "alr",
        minimal: bool = False,
    ):
        mvsn = MultivariateSkewNormal(mvn, shape)
        super().__init__(mvsn, transform, minimal)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
