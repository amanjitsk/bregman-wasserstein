import abc
from collections.abc import Callable
from math import prod
from typing import Any, Literal, Optional, get_args

import jax
from jax import numpy as jnp, random as jr
import equinox as eqx
from equinox import field
from equinox import AbstractVar
import distrax as dx
from jaxtyping import Array, ArrayLike, Scalar, PRNGKeyArray
import sklearn.datasets

from ott import datasets
import costs

OptionalKey = Optional[PRNGKeyArray]
SklearnDatasets = Literal[
    "moon_upper", "moon_lower", "circle_small", "circle_big", "swiss", "s_curve"
]
GaussianDists = Literal["simple", "circle", "square_five", "square_four"]
Samplers = SklearnDatasets | GaussianDists | Literal["maf_moon", "rings"]
Datasets = Literal[
    "demo",
    "makkuva_1",
    "makkuva_2",
    "rout_1",
    "rout_2",
    "rout_3",
    "rout_4",
    "huang_1",
    "huang_2",
    "huang_3",
]
Loader = tuple[datasets.Dataset, datasets.Dataset, int]


class AbstractSampler(eqx.Module):
    @abc.abstractmethod
    def _sample_n(self, key: PRNGKeyArray, n: int) -> Array:
        raise NotImplementedError

    def sample(self, key: PRNGKeyArray, shape: int | tuple[int, ...]) -> Array:
        if isinstance(shape, int):
            shape = (shape,)
        samples = self._sample_n(key, prod(shape))
        return samples.reshape(shape + samples.shape[-1:])


class Sampler(AbstractSampler):
    sample_fn: Callable[[PRNGKeyArray, int], Array]

    def _sample_n(self, key, n):
        return self.sample_fn(key, n)


class AbstractPairedSampler(eqx.Module):
    @abc.abstractmethod
    def _sample_n(self, key: PRNGKeyArray, n: int) -> tuple[Array, Array]:
        raise NotImplementedError

    def sample(
        self, key: PRNGKeyArray, shape: int | tuple[int, ...]
    ) -> tuple[Array, Array]:
        if isinstance(shape, int):
            shape = (shape,)
        source, target = self._sample_n(key, prod(shape))
        source = source.reshape(shape + source.shape[-1:])
        target = target.reshape(shape + target.shape[-1:])
        return source, target

    def decouple(self) -> tuple[AbstractSampler, AbstractSampler]:
        """Return the two component samplers."""
        source = Sampler(lambda key, n: self._sample_n(key, n)[0])
        target = Sampler(lambda key, n: self._sample_n(key, n)[1])
        return source, target


class IterableSampler(eqx.Module):
    sampler: AbstractSampler | AbstractPairedSampler
    batch_size: int
    rng: OptionalKey = None

    def __iter__(self):
        rng = self.rng if self.rng is not None else jr.key(0)
        while True:
            rng, sample_key = jax.random.split(rng, 2)
            yield self.sampler._sample_n(sample_key, self.batch_size)

    def iterator(self):
        return iter(self)


class MirrorSampler(AbstractPairedSampler):
    sampler: AbstractPairedSampler
    bregman: costs.AbstractBregman
    dual_interpolation: bool = field(static=True, default=True)

    def transform(self, source, target):
        if self.dual_interpolation:
            return source, self.bregman.to_dual(target)
        else:
            return self.bregman.to_dual(source), target

    def _sample_n(self, key, n):
        source, target = self.sampler._sample_n(key, n)
        return self.transform(source, target)


class AbstractCoupledSampler(AbstractPairedSampler):
    source: AbstractVar[AbstractSampler]
    target: AbstractVar[AbstractSampler]

    @abc.abstractmethod
    def transform(self, source: Array, target: Array) -> tuple[Array, Array]:
        raise NotImplementedError

    def _sample_n(self, key: PRNGKeyArray, n: int) -> tuple[Array, Array]:
        k1, k2 = jr.split(key, 2)
        source = self.source._sample_n(k1, n)
        target = self.target._sample_n(k2, n)
        return self.transform(source, target)

    def decouple(self) -> tuple[AbstractSampler, AbstractSampler]:
        """Return the two component samplers."""
        source = Sampler(lambda key, n: self._sample_n(key, n)[0])
        target = Sampler(lambda key, n: self._sample_n(key, n)[1])
        return source, target


class Independent(AbstractCoupledSampler):
    source: AbstractSampler
    target: AbstractSampler

    def transform(self, source: Array, target: Array) -> tuple[Array, Array]:
        return source, target


class Hypercube(AbstractCoupledSampler):
    source: AbstractSampler
    target: AbstractSampler
    a: float = field(static=True, default=0.05)
    b: float = field(static=True, default=0.95)

    def __check__init__(self):
        assert self.a < self.b

    def _transform(self, X: Array, min_x: Scalar, max_x: Scalar) -> Array:
        return self.a + (self.b - self.a) * (X - min_x) / (max_x - min_x)

    def transform(self, source, target):
        joint = jnp.concatenate([source, target], axis=1).ravel()
        x_min, x_max = joint.min(), joint.max()
        return self._transform(source, x_min, x_max), self._transform(
            target, x_min, x_max
        )


class MAFMoonSampler(AbstractSampler):
    scale: float = field(static=True, default=0.5)
    translation: float = field(static=True, default=-2.0)

    def _sample_n(self, key, n):
        x = self.scale * jax.random.normal(key, shape=[n, 2])
        x = x.at[:, 0].add(x[:, 1] ** 2)
        x = x.at[:, 0].mul(self.scale)
        x = x.at[:, 0].add(self.translation)
        return x


class RingSampler(AbstractSampler):
    scale: float = field(default=3.0, static=True)
    noise: float = field(default=0.08, static=True)

    def _sample_n(self, key, n):
        n_samples4 = n_samples3 = n_samples2 = n // 4
        n_samples1 = n - n_samples4 - n_samples3 - n_samples2

        linspace4 = jnp.linspace(0, 2 * jnp.pi, n_samples4, endpoint=False)
        linspace3 = jnp.linspace(0, 2 * jnp.pi, n_samples3, endpoint=False)
        linspace2 = jnp.linspace(0, 2 * jnp.pi, n_samples2, endpoint=False)
        linspace1 = jnp.linspace(0, 2 * jnp.pi, n_samples1, endpoint=False)

        circ4_x = jnp.cos(linspace4) * 1.2
        circ4_y = jnp.sin(linspace4) * 1.2
        circ3_x = jnp.cos(linspace4) * 0.9
        circ3_y = jnp.sin(linspace3) * 0.9
        circ2_x = jnp.cos(linspace2) * 0.55
        circ2_y = jnp.sin(linspace2) * 0.55
        circ1_x = jnp.cos(linspace1) * 0.25
        circ1_y = jnp.sin(linspace1) * 0.25

        X = (
            jnp.vstack(
                [
                    jnp.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
                    jnp.hstack([circ4_y, circ3_y, circ2_y, circ1_y]),
                ]
            ).T
            * self.scale
        )
        X = sklearn.utils.shuffle(X)  # pyright: ignore

        # Add noise
        X = X + self.noise * jax.random.normal(key, shape=X.shape)

        return X.astype("float32")


class SklearnSampler(AbstractSampler):
    """A class to define toy probability 2-dimensional distributions.

    Produces rotated ``moons`` and ``s_curve`` sklearn datasets, using
    ``theta_rotation``.
    """

    name: SklearnDatasets = field(static=True)
    rotation: float = field(static=True)
    loc: Array = field(static=True)
    scale: float = field(static=True)
    noise: float = field(static=True)

    def __init__(
        self,
        name: SklearnDatasets,
        theta_rotation: float = 0.0,
        loc: Optional[ArrayLike] = None,
        scale: float = 1.0,
        noise: float = 0.01,
    ):
        assert name in get_args(
            SklearnDatasets
        ), f"SklearnSampler `{name}` not implemented."
        self.name = name
        self.rotation = theta_rotation
        self.loc = jnp.zeros(2) if loc is None else jnp.asarray(loc)
        self.scale = scale
        self.noise = noise

    def _sample_n(self, key, n):
        # define rotation matrix tp rotate samples
        rotation = jnp.array(
            [
                [jnp.cos(self.rotation), -jnp.sin(self.rotation)],
                [jnp.sin(self.rotation), jnp.cos(self.rotation)],
            ]
        )
        seed = jax.random.randint(key, [], minval=0, maxval=1e5).item()
        if self.name == "moon_upper":
            samples, _ = sklearn.datasets.make_moons(
                n_samples=(n, 0),  # pyright: ignore
                random_state=seed,
                noise=self.noise,
            )
        elif self.name == "moon_lower":
            samples, _ = sklearn.datasets.make_moons(
                n_samples=(0, n),  # pyright: ignore
                random_state=seed,
                noise=self.noise,
            )
        elif self.name == "circle_small":
            samples, _ = sklearn.datasets.make_circles(
                n_samples=(0, n),  # pyright: ignore
                random_state=seed,
                noise=self.noise,
                factor=0.5,
            )
        elif self.name == "circle_big":
            samples, _ = sklearn.datasets.make_circles(
                n_samples=(n, 0),  # pyright: ignore
                random_state=seed,
                noise=self.noise,
                factor=0.5,
            )
        elif self.name == "swiss":
            x, _ = sklearn.datasets.make_swiss_roll(
                n_samples=n,
                random_state=seed,
                noise=self.noise,
            )
            samples = x[:, [2, 0]]
        elif self.name == "s_curve":
            x, _ = sklearn.datasets.make_s_curve(
                n_samples=n,
                random_state=seed,
                noise=self.noise,
            )
            samples = x[:, [2, 0]]

        samples = jnp.asarray(samples, dtype=jnp.float32)  # pyright: ignore
        samples = jnp.squeeze(jnp.matmul(rotation[None, :], samples.T).T)
        samples = self.loc + self.scale * samples
        return samples


class Transformed(AbstractSampler):
    base: AbstractSampler
    transform: Callable[[Array], Array]

    def __init__(self, base: AbstractSampler, transform: Callable[[Array], Array]):
        self.base = base
        self.transform = transform

    def _sample_n(self, key, n):
        return self.transform(self.base._sample_n(key, n))


class Distrax(AbstractSampler):
    dist: dx.Distribution

    def _sample_n(self, key, n):
        return self.dist.sample(seed=key, sample_shape=(n,))


class GaussianMixture(AbstractSampler):
    centers: Array = field(static=True)
    loc: float = field(static=True)
    scale: float = field(static=True)

    def __init__(self, name: GaussianDists, loc: float = 5.0, scale: float = 0.25):
        assert name in get_args(
            GaussianDists
        ), f"GaussianSampler `{name}` not implemented."
        self.centers = {
            "simple": jnp.array([[0, 0]]),
            "circle": jnp.array(
                [
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)),
                    (1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)),
                    (-1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)),
                    (-1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)),
                ]
            ),
            "square_five": jnp.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]]),
            "square_four": jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
        }[name]
        self.loc = loc
        self.scale = scale

    def _sample_n(self, key, n):
        rng1, rng2 = jax.random.split(key)
        means = jax.random.choice(rng1, self.centers, (n,))
        normal_samples = jax.random.normal(rng2, (n, 2))
        samples = self.loc * means + (self.scale) * normal_samples
        return samples


def make_sampler(**kwargs) -> AbstractSampler:
    name: Samplers | dx.Distribution = kwargs.pop("name", None)

    if isinstance(name, dx.Distribution):
        return Distrax(dist=name, **kwargs)

    if name == "maf_moon":
        return MAFMoonSampler(**kwargs)
    elif name == "rings":
        return RingSampler(**kwargs)
    elif name.startswith("gauss_") and name[6:] in get_args(GaussianDists):
        return GaussianMixture(name[6:], **kwargs)  # pyright: ignore
    else:
        return SklearnSampler(name, **kwargs)  # pyright: ignore


def make_samplers(
    source_kwargs: dict[str, Any],
    target_kwargs: dict[str, Any],
    source_transform: Callable[[Array], Array] = lambda x: x,
    target_transform: Callable[[Array], Array] = lambda x: x,
) -> tuple[AbstractSampler, AbstractSampler]:
    """Samplers for the source and target distributions."""

    def make_fn(source: bool) -> AbstractSampler:
        if source:
            kwargs = source_kwargs
            transform = source_transform
        else:
            kwargs = target_kwargs
            transform = target_transform

        sampler = make_sampler(**kwargs)
        sampler = Transformed(sampler, transform)
        return sampler

    source = make_fn(source=True)
    target = make_fn(source=False)

    return source, target


def make_baseline(name: Datasets) -> tuple[AbstractSampler, AbstractSampler]:
    assert name in get_args(Datasets), f"Dataset `{name}` not implemented."

    if name == "demo":
        source_kwargs = {
            "name": "moon_upper",
            "theta_rotation": jnp.pi / 6,
            "loc": jnp.array([0.0, -0.5]),
            "noise": 0.05,
        }
        target_kwargs = {
            "name": "s_curve",
            "theta_rotation": -jnp.pi / 6,
            "loc": jnp.array([0.5, -2.0]),
            "scale": 0.6,
            "noise": 0.05,
        }
    elif name == "makkuva_1":
        source_kwargs = {"name": "gauss_simple"}
        target_kwargs = {"name": "gauss_circle"}
    elif name == "makkuva_2":
        source_kwargs = {"name": "gauss_square_five"}
        target_kwargs = {"name": "gauss_square_four"}
    elif name == "rout_1":
        source_kwargs = {"name": "circle_big"}
        target_kwargs = {"name": "circle_small"}
    elif name == "rout_2":
        source_kwargs = {"name": "moon_lower"}
        target_kwargs = {"name": "moon_upper"}
    elif name == "rout_3":
        source_kwargs = {"name": "s_curve", "scale": 5.0}
        target_kwargs = {"name": "gauss_simple", "scale": 1.0}
    elif name == "rout_4":
        source_kwargs = {"name": "swiss", "scale": 0.5}
        target_kwargs = {"name": "gauss_simple", "scale": 1.0}
    elif name == "huang_1":
        source_kwargs = {"name": "rings"}
        target_kwargs = {"name": "gauss_simple", "scale": 1.0}
    elif name == "huang_2":
        source_kwargs = {"name": "maf_moon"}
        target_kwargs = {"name": "gauss_simple", "scale": 1.0}
    elif name == "huang_3":
        source_kwargs = {"name": "rings"}
        target_kwargs = {"name": "maf_moon"}

    source = make_sampler(**source_kwargs)
    target = make_sampler(**target_kwargs)
    return source, target
