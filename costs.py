import abc
import math
from collections.abc import Callable

from jax import numpy as jnp, tree_util as jtu, vmap, jacfwd, jvp
from jaxtyping import Array, Scalar, PyTree, Float
import equinox as eqx
from equinox import field
import distrax as dx
from ott.geometry import costs

import math_utils as mu
from math_utils import vectorize, Potential

Identity = dx.Lambda(lambda x: x)


class AbstractCost(eqx.Module, costs.CostFn):
    def __call__(self, X: Array, Y: Array):
        """Vectorized version of the cost function."""
        return vectorize(self.compute, in_ndims=(1, 1), out_ndims=0)(X, Y)

    def all_pairs(self, X: Array, Y: Array) -> Array:
        """Compute matrix of all pairwise costs.

        Args:
            X: Array of shape ``[n, ...]``.
            Y: Array of shape ``[m, ...]``.

        Returns:
            Array of shape ``[n, m]`` of cost evaluations.
        """
        return vmap(lambda x_: vmap(lambda y_: self.compute(x_, y_))(Y))(X)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def right(self) -> bool:
        """Useful for barycenter method for asymmetric costs."""
        return False

    @abc.abstractmethod
    def compute(self, x: Array, y: Array) -> Scalar:
        """Return the cost i.e. :math:`c(x,y)`."""
        raise NotImplementedError

    def barycenter(
        self, weights: Array, xs: Array, right: bool | None = None
    ) -> tuple[Array, PyTree]:
        """Barycentric operator.

        Args:
            weights: Convex set of weights.
            xs: Points.
            right: Index of the left to optimize. If True, compute
            the right barycenter, if False, compute the left barycenter.

        Returns:
            A tuple, whose first element is the barycenter of `xs` using `weights`
            coefficients, followed by auxiliary information on the convergence of
            the algorithm (or any other information).
        """
        raise NotImplementedError("Barycenter is not implemented.")

    def twist_operator(self, vec: Array, dual_vec: Array, right: bool) -> Array:
        r"""Twist inverse operator of the cost function.

        Given a cost function :math:`c`, the twist operator returns
        :math:`\nabla_{1}c(x, \cdot)^{-1}(z)` if ``left`` is ``0``,
        and :math:`\nabla_{2}c(\cdot, y)^{-1}(z)` if ``left`` is ``1``, for
        :math:`x=y` equal to ``vec`` and :math:`z` equal to ``dual_vec``.

        Args:
          vec: ``[p,]`` point at which the twist inverse operator is evaluated.
          dual_vec: ``[q,]`` point to invert by the operator.
          right: apply twist inverse operator on first (i.e. value set to ``0``
            or equivalently ``False``) or second (``1`` or ``True``) coordinate.

        Returns:
          A vector.
        """
        raise NotImplementedError("Twist operator is not implemented.")

    def c_exp(self, x: Array, p: Array, right: bool) -> Array:
        """Compute the c-exponential of the cost function.
        See Definition 12.29 in Villani (2009).
        This is useful for computing the transport map given the Kantorovich map.
        The Kantorovich map is the (Riemannian) gradient of the Kantorovich potential.

        Args:
            x: Array.
            p: Array.

        Returns:
            The c-exponential c-exp_x(p).
        """
        return self.twist_operator(x, -p, right=right)

    def reversed(self):
        return DualCost(self)

    def scaled(self, scale: float):
        return ScaledCost(self, scale=scale)


class ScaledCost(AbstractCost):
    cost: AbstractCost
    scale: float = field(default=1.0, static=True)  # default to 1.0

    def __check__init__(self):
        assert self.scale > 0

    @property
    def name(self) -> str:
        return "Scaled" + self.cost.name

    def compute(self, x: Array, y: Array) -> Scalar:
        return self.scale * self.cost.compute(x, y)

    def barycenter(
        self, weights: Array, xs: Array, right: bool | None = None
    ) -> tuple[Array, PyTree]:
        return self.cost.barycenter(weights, xs, right=right)

    def twist_operator(self, vec: Array, dual_vec: Array, right: bool) -> Array:
        return self.cost.twist_operator(vec, dual_vec / self.scale, right=right)

    def _padder(self, dim: int) -> Array:
        return self.cost._padder(dim)


class DualCost(AbstractCost):
    cost: AbstractCost

    @property
    def name(self) -> str:
        return "Dual" + self.cost.name

    @property
    def right(self) -> bool:
        return not self.cost.right

    def compute(self, x: Array, y: Array) -> Scalar:
        return self.cost.compute(y, x)

    def barycenter(
        self, weights: Array, xs: Array, right: bool | None = None
    ) -> tuple[Array, PyTree]:
        right = self.right if right is None else right
        return self.cost.barycenter(weights, xs, right=right)

    def twist_operator(self, vec: Array, dual_vec: Array, right: bool) -> Array:
        return self.cost.twist_operator(vec, dual_vec, right=not right)

    def _padder(self, dim: int) -> Array:
        return self.cost._padder(dim)


class InnerProduct(AbstractCost):
    def compute(self, x: Array, y: Array) -> Scalar:
        return -jnp.dot(x, y)

    def barycenter(
        self, weights: Array, xs: Array, right: bool | None = None
    ) -> tuple[Array, PyTree]:
        del right
        return jnp.average(xs, weights=weights, axis=0), None

    def twist_operator(self, vec: Array, dual_vec: Array, right: bool) -> Array:
        del vec, right
        return -dual_vec


class AbstractPrimalDual(AbstractCost):
    """Abstract class for primal-dual cost functions.
    These can represent constrained (primal and dual)
    spaces, e.g. using Bregman divergences or others.
    """

    bijector = eqx.AbstractVar[dx.Bijector]

    def to_dual(self, X: Array) -> Array:
        return vectorize(self.bijector.forward)(X)  # pyright: ignore

    def to_primal(self, Y: Array) -> Array:
        return vectorize(self.bijector.inverse)(Y)  # pyright: ignore

    def dual_dist(self, dist: dx.Distribution) -> dx.Distribution:
        return dx.Transformed(dist, self.bijector)

    def primal_dist(self, dist: dx.Distribution) -> dx.Distribution:
        return dx.Transformed(dist, dx.Inverse(self.bijector))

    @abc.abstractmethod
    def project_primal(self, x: Array) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def project_dual(self, y: Array) -> Array:
        raise NotImplementedError

    def projection(self, dual: bool = False) -> Callable[[Array, PyTree], Array]:
        """Primal and dual projection functions."""
        if dual:
            return lambda y, _: self.project_dual(y)
        else:
            return lambda x, _: self.project_primal(x)

    def to_dual_(self, X: Array) -> Array:
        """Map to dual coordinates, but apply primal constraint to X first."""
        return self.to_dual(vectorize(self.project_primal)(X))

    def to_primal_(self, Y: Array) -> Array:
        """Map to primal coordinates, but apply dual constraint to Y first."""
        return self.to_primal(vectorize(self.project_dual)(Y))


class AbstractBregman(AbstractPrimalDual):
    """Bregman cost functions."""

    potential: eqx.AbstractVar[Potential]
    conjugate: eqx.AbstractVar[Potential]

    @property
    def mirror(self) -> dx.Bijector:
        return self.bijector

    def compute(self, x: Array, y: Array) -> Scalar:
        r"""Bregman cost i.e. :math:`c(x,y) = B_{\Omega}(x,y)`."""
        return self.primal(x, y)

    def metric(self, X: Array) -> Array:
        """Compute the Hessian Riemannian metric (matrix)."""
        return vectorize(jacfwd(self.mirror.forward), out_dims=2)(X)

    def onsager(self, Y: Array) -> Array:
        """Compute the Onsager (matrix) operator (inverse of the Riemannian metric)."""
        return vectorize(jacfwd(self.mirror.inverse), out_dims=2)(Y)

    def barycenter(
        self, weights: Array, xs: Array, right: bool | None = None
    ) -> tuple[Array, PyTree]:
        right = self.right if right is None else right
        if not right:
            bar = self.to_primal(jnp.average(self.to_dual(xs), weights=weights, axis=0))
        else:
            bar = jnp.average(xs, weights=weights, axis=0)

        return bar, None

    def twist_operator(self, vec: Array, dual_vec: Array, right: bool) -> Array:
        if not right:
            return self.to_primal(self.to_dual(vec) - dual_vec)
        else:
            return vec - jvp(self.mirror.inverse, (self.to_dual(vec),), (dual_vec,))[1]

    def primal(self, x1: Array, x2: Array) -> Scalar:
        return (
            self.potential(x1)
            - self.potential(x2)
            - jnp.dot(self.mirror.forward(x2), x1 - x2)
        )

    def dual(self, y1: Array, y2: Array) -> Scalar:
        return (
            self.conjugate(y1)
            - self.conjugate(y2)
            - jnp.dot(self.mirror.inverse(y2), y1 - y2)
        )

    def mixed(self, x: Array, y: Array) -> Scalar:
        return self.potential(x) + self.conjugate(y) - jnp.dot(x, y)

    def dualized(self):
        return ConjugateBregman(self)


@jtu.register_pytree_node_class
class SqEuclidean(costs.TICost):
    r"""Squared Euclidean distance.

    Implemented as a translation invariant cost, :math:`h(z) = \|z\|^2 / 2`.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        assert scale > 0
        self.scale = scale

    def norm(self, x: Array) -> Array:
        """Compute squared Euclidean norm for vector."""
        return 0.5 * jnp.sum(x**2, axis=-1)

    def __call__(self, x: Array, y: Array) -> Scalar:
        """Compute minus twice the dot-product between vectors."""
        return self.scale * (self.norm(x) + self.norm(y) - jnp.vdot(x, y))

    def _h(self, z: Array) -> Scalar:
        return 0.5 * jnp.sum(z**2)

    def h(self, z: jnp.ndarray) -> Scalar:
        return self.scale * self._h(z)

    def h_legendre(self, z: jnp.ndarray) -> Scalar:
        return self.scale * self._h(z / self.scale)

    def barycenter(self, weights: Array, xs: Array) -> tuple[Array, None]:
        """Output barycenter of vectors when using squared-Euclidean distance."""
        return jnp.average(xs, weights=weights, axis=0), None


class Euclidean(AbstractBregman):
    """Euclidean cost."""

    potential: Potential = field(static=True)
    conjugate: Potential = field(static=True)
    bijector: dx.Bijector

    def __init__(self):
        self.bijector = dx.Block(Identity, ndims=1)
        self.potential = lambda x: 0.5 * jnp.sum(x**2)
        self.conjugate = lambda y: 0.5 * jnp.sum(y**2)

    def compute(self, x: Array, y: Array) -> Scalar:
        return 0.5 * jnp.sum((x - y) ** 2)

    def barycenter(
        self, weights: Array, xs: Array, right: bool | None = None
    ) -> tuple[Array, PyTree]:
        del right
        return jnp.average(xs, weights=weights, axis=0), None

    def twist_operator(self, vec: Array, dual_vec: Array, right: bool) -> Array:
        del right
        return vec - dual_vec

    def project_primal(self, x: Array) -> Array:
        return x

    def project_dual(self, y: Array) -> Array:
        return y


class Mahalanobis(AbstractBregman):
    potential: Potential = field(static=True)
    conjugate: Potential = field(static=True)
    bijector: dx.Bijector
    Q: Float[Array, "d d"]
    Qinv: Float[Array, "d d"]
    _name: str = field(static=True)

    def __init__(self, matrix: Float[Array, "d d"], name: str = ""):
        # symmetricize matrix
        Q = 0.5 * (matrix + matrix.T)
        Qinv = jnp.linalg.inv(Q)
        logdet = mu.logdet(Q)
        assert jnp.isfinite(logdet)
        self.potential = lambda x: 0.5 * x.T @ Q @ x
        self.conjugate = lambda y: 0.5 * y.T @ Qinv @ y
        self.bijector = dx.Lambda(
            forward=lambda x: matrix @ x,
            inverse=lambda y: Qinv @ y,
            forward_log_det_jacobian=lambda _: logdet,
            inverse_log_det_jacobian=lambda _: -logdet,
            event_ndims_in=1,
            is_constant_jacobian=True,
        )
        self.Q = Q
        self.Qinv = Qinv
        self._name = name if name else str(hash(str(Q)))

    @property
    def name(self) -> str:
        return self.__class__.__name__ + self._name

    def compute(self, x: Array, y: Array) -> Scalar:
        return 0.5 * (x - y).T @ self.Q @ (x - y)

    def barycenter(
        self, weights: Array, xs: Array, right: bool | None = None
    ) -> tuple[Array, PyTree]:
        del right
        return jnp.average(xs, weights=weights, axis=0), None

    def twist_operator(self, vec: Array, dual_vec: Array, right: bool) -> Array:
        del right
        return vec - self.Qinv @ dual_vec

    def project_primal(self, x: Array) -> Array:
        return x

    def project_dual(self, y: Array) -> Array:
        return y


class ScaledBregman(AbstractBregman):
    base: AbstractBregman
    potential: Potential = field(static=True)
    conjugate: Potential = field(static=True)
    bijector: dx.Bijector

    def __init__(self, bregman: AbstractBregman, scale: float = 1.0):
        self.base = bregman

        def potential(x):
            return scale * bregman.potential(x)

        def conjugate(y):
            return scale * bregman.conjugate(y / scale)

        self.potential = potential
        self.conjugate = conjugate
        mirror = bregman.mirror
        self.bijector = dx.Chain(
            [
                dx.Block(
                    dx.ScalarAffine(shift=0, scale=scale), ndims=mirror.event_ndims_out
                ),
                bregman.mirror,
            ]
        )

    @property
    def name(self) -> str:
        return "Scaled" + self.base.name

    def project_primal(self, x: Array) -> Array:
        return self.base.project_dual(x)

    def project_dual(self, y: Array) -> Array:
        return self.base.project_primal(y)

    def _padder(self, dim: int) -> Array:
        return self.base._padder(dim)


class ConjugateBregman(AbstractBregman):
    """Bregman divergence generated by the (convex) conjugate of the potential."""

    base: AbstractBregman
    potential: Potential = field(static=True)
    conjugate: Potential = field(static=True)
    bijector: dx.Bijector

    def __init__(self, bregman: AbstractBregman):
        self.base = bregman
        self.potential = bregman.conjugate
        self.conjugate = bregman.potential
        self.bijector = dx.Inverse(bregman.mirror)

    @property
    def name(self):
        return "Conjugate" + self.base.name

    def project_primal(self, x: Array) -> Array:
        return self.base.project_dual(x)

    def project_dual(self, y: Array) -> Array:
        return self.base.project_primal(y)


class AbstractComposite(AbstractPrimalDual):
    """Cost upto change of coordinates, i.e. c(T(x),T(y))."""

    cost: eqx.AbstractVar[AbstractCost]

    def compute(self, x: Array, y: Array) -> Scalar:
        Tx = self.bijector.forward(x)  # pyright: ignore
        Ty = self.bijector.forward(y)  # pyright: ignore
        return self.cost.compute(Tx, Ty)

    def barycenter(
        self, weights: Array, xs: Array, right: bool | None = None
    ) -> tuple[Array, PyTree]:
        base_bary, base_info = self.cost.barycenter(
            weights, self.to_dual(xs), right=right
        )
        bary = self.to_primal(base_bary)
        return bary, base_info

    def twist_operator(self, vec: Array, dual_vec: Array, right: bool) -> Array:
        t_vec = self.bijector.forward(vec)  # pyright: ignore
        return self.bijector.inverse(
            self.cost.twist_operator(
                vec=t_vec,
                dual_vec=jvp(self.bijector.inverse, (t_vec,), (dual_vec,))[1],
                right=right,
            )
        )  # pyright: ignore

    def _padder(self, dim: int) -> Array:
        return self.to_primal(self.cost._padder(dim))


class BregmanEuclidean(AbstractComposite):
    """Euclidean cost with change of variables given by a mirror map."""

    base: AbstractBregman
    bijector: dx.Bijector
    cost: AbstractCost

    def __init__(self, bregman: AbstractBregman):
        self.base = bregman
        self.bijector = bregman.mirror
        self.cost = Euclidean()

    def project_primal(self, x: Array) -> Array:
        return self.base.project_primal(x)

    def project_dual(self, y: Array) -> Array:
        return self.base.project_dual(y)


class SimplexKL(AbstractBregman):
    potential: Potential = field(static=True)
    conjugate: Potential = field(static=True)
    bijector: dx.Bijector

    def __init__(self):
        self.potential, self.conjugate, self.bijector = mu.make_simplex_kl()

    def compute(self, x: Array, y: Array) -> Scalar:
        return mu.kl_minimal(x, y)

    def project_primal(self, x: Array) -> Array:
        return mu.project_simplex(x)

    def project_dual(self, y: Array) -> Array:
        return mu.safe_clip(y, -4.0, 4.0)


class SimplexILR(AbstractComposite):
    bijector: dx.Bijector
    cost: AbstractCost

    def __init__(self):
        self.bijector = mu.make_simplex_ilr()
        self.cost = Euclidean()

    def project_primal(self, x: Array) -> Array:
        return mu.project_simplex(x)

    def project_dual(self, y: Array) -> Array:
        return y


class ExtendedKL(AbstractBregman):
    potential: Potential = field(static=True)
    conjugate: Potential = field(static=True)
    bijector: dx.Bijector
    a: float = field(static=True)

    def __init__(self, a: float = 1.0):
        self.potential, self.conjugate, self.bijector = mu.make_extended_kl(a)
        self.a = a

    def __check__init__(self):
        assert self.a != 0, "Use Euclidean cost function directly!"

    @property
    def name(self) -> str:
        a = self.a
        return self.__class__.__name__ + f" ({a=})"

    def compute(self, x: Array, y: Array) -> Scalar:
        return mu.gen_kl(self.a * x, self.a * y) / (self.a**2)

    def project_primal(self, x: Array) -> Array:
        if self.a > 0:
            return mu.safe_clip(x, a_min=0.0)
        else:
            return mu.safe_clip(x, a_max=0.0)

    def project_dual(self, y: Array) -> Array:
        return y

    def _padder(self, dim: int) -> Array:
        return jnp.sign(self.a) * jnp.ones((dim,))


class GaussianEFLPF(AbstractBregman):
    """Gaussian log partition function (exponential family)."""

    potential: Potential = field(static=True)
    conjugate: Potential = field(static=True)
    bijector: dx.Bijector
    dim: int = field(static=True)

    def __init__(self, dim: int):
        self.potential, self.conjugate, self.bijector = mu.make_gaussian_eflpf(dim)
        self.dim = dim

    @classmethod
    def _padder(cls, dim: int) -> jnp.ndarray:
        dimension = int((-1 + math.sqrt(1 + 4 * dim)) / 2)
        padding = costs.mean_and_cov_to_x(
            jnp.zeros((dimension,)), -0.5 * jnp.eye(dimension), dimension
        )
        return padding[jnp.newaxis, :]

    def project_primal(self, x):
        return x

    def project_dual(self, y):
        return y


class GaussianLPF(AbstractComposite):
    bijector: dx.Bijector
    cost: GaussianEFLPF

    def __init__(self, dim: int):
        self.bijector = mu.GaussianNatural(dim)
        self.cost = GaussianEFLPF(dim)  # pyright: ignore

    # @classmethod
    # def _padder(cls, dim: int) -> jnp.ndarray:
    #     dimension = int((-1 + math.sqrt(1 + 4 * dim)) / 2)
    #     padding = costs.mean_and_cov_to_x(
    #         jnp.zeros((dimension,)), jnp.eye(dimension), dimension
    #     )
    #     return padding[jnp.newaxis, :]

    def compute(self, x, y):
        return mu.gaussian_kl(y, x, self.cost.dim)

    def project_primal(self, x):
        return x

    def project_dual(self, y):
        return y


class EuclideanGaussianLPF(AbstractComposite):
    bijector: dx.Bijector
    cost: AbstractCost

    def __init__(self, dim: int):
        self.bijector = mu.GaussianNatural(dim)
        self.cost = Euclidean().scaled(2.0)

    @classmethod
    def _padder(cls, dim: int) -> jnp.ndarray:
        dimension = int((-1 + math.sqrt(1 + 4 * dim)) / 2)
        padding = costs.mean_and_cov_to_x(
            jnp.zeros((dimension,)), jnp.eye(dimension), dimension
        )
        return padding[jnp.newaxis, :]

    def project_primal(self, x):
        return x

    def project_dual(self, y):
        return y


class HNNTanh(AbstractBregman):
    potential: Potential = field(static=True)
    conjugate: Potential = field(static=True)
    bijector: dx.Bijector
    beta: float = field(static=True)

    def __init__(self, beta: float = 1.0):
        self.potential, self.conjugate, self.bijector = mu.make_hnn(beta)
        self.beta = beta

    @property
    def name(self) -> str:
        beta = self.beta
        return self.__class__.__name__ + f" ({beta=})"

    def project_primal(self, x: Array) -> Array:
        return mu.safe_clip(x, 0.0, 1.0)

    def project_dual(self, y: Array) -> Array:
        return y


class RiemannHNNTanh(AbstractComposite):
    bijector: dx.Bijector
    cost: AbstractCost

    def __init__(self, beta: float = 1.0):
        self.bijector = mu.make_r_hnn(beta)
        self.cost = Euclidean()

    def project_primal(self, x: Array) -> Array:
        return mu.safe_clip(x, 0.0, 1.0)

    def project_dual(self, y: Array) -> Array:
        return y


class Arctan(AbstractBregman):
    potential: Potential = field(static=True)
    conjugate: Potential = field(static=True)
    bijector: dx.Bijector

    def __init__(self):
        self.potential, self.conjugate, self.bijector = mu.make_arctan()

    def project_primal(self, x: Array) -> Array:
        return mu.safe_clip(x, -math.pi / 2, math.pi / 2)

    def project_dual(self, y: Array) -> Array:
        return y


class RiemannArctan(AbstractComposite):
    bijector: dx.Bijector
    cost: AbstractCost

    def __init__(self):
        self.bijector = mu.make_r_arctan()
        self.cost = Euclidean()

    def project_primal(self, x: Array) -> Array:
        return mu.safe_clip(x, -math.pi / 2, math.pi / 2)

    def project_dual(self, y: Array) -> Array:
        return y


class Tanh(AbstractBregman):
    potential: Potential = field(static=True)
    conjugate: Potential = field(static=True)
    bijector: dx.Bijector

    def __init__(self):
        self.potential, self.conjugate, self.bijector = mu.make_tanh()

    def project_primal(self, x: Array) -> Array:
        return mu.safe_clip(x, -1.0, 1.0)

    def project_dual(self, y: Array) -> Array:
        return y


class RiemannTanh(AbstractComposite):
    bijector: dx.Bijector
    cost: AbstractCost

    def __init__(self):
        self.bijector = mu.make_r_tanh()
        self.cost = Euclidean()

    def project_primal(self, x: Array) -> Array:
        return mu.safe_clip(x, -1.0, 1.0)

    def project_dual(self, y: Array) -> Array:
        return mu.safe_clip(y, -math.pi / 2, math.pi / 2)
