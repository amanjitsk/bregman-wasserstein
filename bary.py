import fire
from collections.abc import Sequence, Callable
from time import time
import traceback

import numpy as np
from scipy.linalg import pascal
import jax
from jax import numpy as jnp, random as jr
from jaxtyping import Array, Float, PRNGKeyArray
import distrax as dx
from ott.problems.linear import barycenter_problem
from ott.solvers.linear import continuous_barycenter, sinkhorn
from ott.geometry.costs import (
    CostFn,
    mean_and_cov_to_x,
    Bures,
    SqEuclidean,
)

import costs
import math_utils as mu
import plot_utils as pu
from plot_utils import NUM_LEVELS


means_and_covs_to_x = jax.vmap(mean_and_cov_to_x, in_axes=(0, 0, None))
plt = pu.plt

LABELS = dict(
    Bures="Bures",
    SqEuclidean=r"Euclidean",
    Euclidean="Euclidean",
    GaussianLPF="Bregman",
    EuclideanGaussianLPF=r"Euclidean ($\Theta$)",
    DualGaussianLPF="Bregman",
    SimplexKL="Negentropy (left)",
    DualSimplexKL="Negentropy (right)",
)


def _bar_weight(num_measures, unif: bool = False):
    if unif:
        bar_weight = jnp.ones(num_measures)
    else:
        bar_weight = jnp.roll(pascal(num_measures, kind="lower")[-1], num_measures // 2)
    return bar_weight / bar_weight.sum()


def cost_name(cost_fn: CostFn) -> str:
    if hasattr(cost_fn, "name"):
        return cost_fn.name  # pyright: ignore
    return cost_fn.__class__.__name__


def get_label(cost_fn: CostFn, default=None) -> str:
    default = cost_name(cost_fn) if default is None else str(default)
    return LABELS.get(cost_name(cost_fn), cost_name(cost_fn))


def sinkhorn_barycenter(
    cost_fn: CostFn,
    particles: Sequence[Array],
    weights: Sequence[Array],
    bar_init: Array,
    bar_weight: Array | None = None,
    epsilon: float = 1e-2,
) -> Array:
    bar_size = bar_init.shape[0]
    if bar_weight is None:
        bar_weight = jnp.ones_like(bar_init, shape=(bar_size,))
    bar_weight /= bar_weight.sum()
    ys = jnp.vstack(particles)
    bs = jnp.concat(weights)
    num_per_segment = tuple(w.size for w in weights)
    problem = barycenter_problem.FreeBarycenterProblem(
        y=ys,
        b=bs,
        weights=bar_weight,
        cost_fn=cost_fn,
        epsilon=epsilon,
        num_per_segment=num_per_segment,
    )

    linear_solver = sinkhorn.Sinkhorn(lse_mode=True)
    solver = continuous_barycenter.FreeWassersteinBarycenter(
        linear_solver,
        # threshold=1e-4,
        # min_iterations=5,
        # max_iterations=100,
        # store_inner_errors=False,
    )
    state = solver(problem, bar_size=bar_size, x_init=bar_init)
    return state.x  # pyright: ignore


def dual_geodesic(
    key: PRNGKeyArray,
    cost_fn: costs.AbstractPrimalDual,
    endpoints: Float[Array, "num dim"],
    num_measures: int,
    num_particles: int,
    dist_fn: Callable[[Array, PRNGKeyArray], dx.Distribution],
    remove_midpoint: bool = True,
    cutoff: slice = slice(0, None),
) -> tuple[Sequence[dx.Distribution], Sequence[Array], Sequence[Array]]:
    num_measures *= endpoints.shape[0] - 1
    measures, particles, weights = [], [], []
    # dual geodesic in prmal coordinates
    projection = jax.vmap(cost_fn.project_dual)
    line = cost_fn.to_primal(mu.line(projection(cost_fn.to_dual_(endpoints)), n=100))[
        cutoff
    ]
    num_measures += 2 if num_measures % 2 == 0 else 0
    line = line[line.shape[0] % num_measures :: line.shape[0] // num_measures]

    if remove_midpoint:
        num_measures = line.shape[0]
        if num_measures % 2 == 0:
            idx = num_measures // 2 - 1
            obj = slice(idx, idx + 2)
        else:
            idx = num_measures // 2
            obj = slice(idx, idx + 1)
        line = jnp.delete(line, obj, axis=0)
    num_measures = line.shape[0]

    for i in range(num_measures):
        subkey = jr.fold_in(key, i)
        dist = dist_fn(line[i], subkey)
        measures.append(dist)
        particles.append(dist.sample(seed=subkey, sample_shape=(num_particles,)))
        weights.append(jnp.full((num_particles,), 1.0 / num_particles))

    return measures, particles, weights


def random_measures(
    key: PRNGKeyArray,
    num_measures: int,
    num_particles: int,
    dist_fn: Callable[[PRNGKeyArray], dx.Distribution],
) -> tuple[Sequence[dx.Distribution], Sequence[Array], Sequence[Array]]:
    measures, particles, weights = [], [], []

    for i in range(num_measures):
        subkey = jr.fold_in(key, i)
        dist = dist_fn(subkey)
        measures.append(dist)
        particles.append(dist.sample(seed=subkey, sample_shape=(num_particles,)))
        weights.append(jnp.full((num_particles,), 1.0 / num_particles))

    return measures, particles, weights


def simplex(
    num_marginals: int = 8,
    num_particles: int = 1000,
    epsilon: float = 1e-2,
    a0: float = 50.0,
    dry: bool = False,
    plot_samples: bool = True,
    save: bool = False,
):
    """Barycenters on the Simplex."""

    dlb, dub = -5.0, 5.0
    dlim = np.r_[dlb, dub]

    def plot(
        axs,
        samples: Float[Array, "N 2"],
        dist: dx.Distribution | None,
        color="C0",
        **kwargs,
    ):
        dual_samples = kl.to_dual(samples)
        estimate = dist is None
        if estimate:
            P = mu.s_to_p(samples)
            mean = jnp.mean(P, axis=0)
            var = jnp.var(P, axis=0)
            precision = jnp.mean(mean * (1 - mean) / var - 1)
            concentration = precision * mean
            primal_dist = mu.MinimalSimplex(dx.Dirichlet(concentration))
            dual_dist = dx.Transformed(primal_dist, kl.bijector)
        else:
            primal_dist = dist
            dual_dist = dx.Transformed(dist, kl.bijector)

        pax, dax = axs
        scatter_kw = dict(color=color, alpha=0.2, s=7)
        pu.draw_scatter(
            pax,
            samples,
            density=jax.vmap(primal_dist.prob),
            sample=plot_samples,
            scatter_kw=scatter_kw,
            is_simplex=True,
            levels=NUM_LEVELS,
            **kwargs,
        )
        pu.draw_scatter(
            dax,
            dual_samples,
            density=jax.vmap(dual_dist.prob),
            sample=plot_samples,
            scatter_kw=scatter_kw,
            format_kw=None,  # don't format here
            is_simplex=False,
            xlim=dlim,
            ylim=dlim,
            marginals=True,
            levels=NUM_LEVELS,
            color=color,
            **kwargs,
        )

    def finalize_axes(axs):
        pax, dax = axs
        pu.setup_simplex(pax)
        pu.format_axes(
            dax,
            xticks=dlim,
            yticks=dlim,
            xlim=dlim,
            ylim=dlim,
            nbins=2,
            xlabel_kwargs=dict(x=0.9),
            ylabel_kwargs=dict(y=0.9),
        )

    kl = costs.SimplexKL()
    w2 = costs.ScaledCost(costs.Euclidean(), scale=1e2)
    cost_fns = (kl, w2)
    # cost_fns += (costs.DualCost(kl),)
    points = (
        np.r_[[[0.15, 0.85, 0], [0.1, 0, 0.9]]],
        np.r_[[[0.01, 0.69, 0.3], [0.7, 0, 0.3]]],
        np.r_[[[0.75, 0.25, 0], [0.01, 0.29, 0.7]]],
    )
    key = jr.key(0)

    ncols = len(points)
    nrows = 2
    pu.latexify()
    fig, subfigs, axes = pu.make_figure(nrows, ncols, 3)
    labels, colors = [], []
    names_map = dict(
        simplexkl="Negentropy (left)",
        simplexkl_dual="Negentropy (right)",
        euclidean=r"Quadratic",
    )

    for i, endpoints in enumerate(points):
        axs = axes[:, i]
        measures, particles, weights = dual_geodesic(
            key=key,
            cost_fn=kl,
            endpoints=mu.p_to_s(endpoints),
            num_measures=num_marginals,
            num_particles=num_particles,
            dist_fn=lambda s, _: mu.MinimalSimplex(dx.Dirichlet(mu.s_to_p(s) * a0)),
            cutoff=slice(15, -15),
            # cutoff=slice(None, None),
        )
        num_measures = len(measures)
        bar_size = num_particles
        bar_weight = _bar_weight(num_measures, unif=False)
        bar_init = mu.p_to_s(jr.dirichlet(key, jnp.ones(3), (bar_size,)))
        for samples, marginal in zip(particles, measures):
            plot(axs, samples, marginal, color="k", cmap="Greys")
        if i == 0:
            labels.append("Marginals")
            colors.append("k")

        if not dry:
            for cid, cost_fn in enumerate(cost_fns):
                bar = sinkhorn_barycenter(
                    cost_fn=cost_fn,
                    particles=particles,
                    weights=weights,
                    bar_init=bar_init,
                    bar_weight=bar_weight,
                    epsilon=epsilon,
                )
                color = f"C{cid}"
                cmap = pu.linear_cmap(colors=["white", color], name=cost_fn.name)
                plot(axs, bar, None, color=color, cmap=cmap)
                if i == 0:
                    labels.append(names_map[cost_fn.name])
                    colors.append(color)
        # finalize axes
        finalize_axes(axs)

    handles = pu.create_legend_handles(labels, colors=colors)
    subfigs[0].legend(
        handles=handles,
        loc="outside lower center",
        # bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        ncol=len(cost_fns) + 1,
    )
    if save:
        plt.savefig(f"simplex_{num_marginals}x{num_particles}.pdf")
    plt.show()


def hypercube(
    num_marginals: int = 5,
    num_particles: int = 100,
    num_barycenters: int = 1,
    beta_marginals: bool = False,
    epsilon: float = 1e-2,
    dry: bool = False,
    plot_samples: bool = True,
    save: bool = False,
):
    """Barycenters on the Hypercube [0, 1]^d."""
    eps = 1e-4
    plb, pub = 0.0, 1.0
    plim_ = np.r_[plb + eps, pub - eps]
    plim = np.r_[plb, pub]
    scatter_alpha = 0.5
    contour_alpha = 0.5
    # dlb, dub = -1.0, 2.0
    # dlim = np.r_[dlb, dub]

    def plot(
        axs,
        samples: Float[Array, "N 2"],
        dist: dx.Distribution | None,
        color="C0",
        **kwargs,
    ):
        if dist is None:
            dist = mu.KDE(samples)

        # pax, dax = axs
        pax = axs
        scatter_kw = dict(color=color, alpha=scatter_alpha, s=7)
        pu.draw_scatter(
            pax,
            samples,
            density=jax.vmap(dist.prob),
            sample=plot_samples,
            scatter_kw=scatter_kw,
            format_kw=None,  # don't format here
            is_simplex=False,
            xlim=plim_,
            ylim=plim_,
            marginals=True,
            levels=NUM_LEVELS,
            color=color,
            alpha=contour_alpha,
            **kwargs,
        )

    def finalize_axes(axs, **kwargs):
        # pax, dax = axs
        pax = axs
        pu.format_axes(
            pax,
            xticks=plim,
            yticks=plim,
            xlim=plim,
            ylim=plim,
            nbins=2,
            xlabel_kwargs=dict(x=0.9),
            ylabel_kwargs=dict(y=0.9),
            visible_spines="all",
            **kwargs,
        )

    # hnn_betas = np.logspace(-1, 1, 3)
    # hnn_betas = np.linspace(0.1, 1, 2)
    # hnn_betas = np.linspace(0.25, 1, 4)
    baseline = 1e-1
    hnn_betas = [baseline]
    duals = True
    hnn_ = costs.HNNTanh()  # baseline
    w2 = costs.ScaledCost(costs.Euclidean(), scale=1e2)
    key = jr.key(0)

    marginals_together = not beta_marginals
    pu.latexify()
    nrows = num_barycenters
    ncols = (1 if marginals_together else num_marginals) + (2 if duals else 1) + 1
    cmap = pu.get_cmap("coolwarm")
    colors = cmap(np.linspace(0, 1, len(hnn_betas)))
    fig, subfigs, axes = pu.make_figure(nrows, ncols, 3)
    names_map = dict(
        hnntanh="HNN-Sigmoid (left)",
        hnntanh_dual="HNN-Sigmoid (right)",
        euclidean="Quadratic",
    )

    def dist_fn(key):
        if beta_marginals:
            return mu.mixture_beta(key, 2, 3)
        else:
            x = jr.uniform(key, (2,), minval=0.05, maxval=0.95)
            y = hnn_.to_dual_(x)
            var = jr.uniform(key, (2,), minval=0.05, maxval=0.25)
            rho = jr.uniform(key, (), minval=-0.1, maxval=0.1)
            cov = jnp.array([[var[0], rho], [rho, var[1]]])
            mvn = dx.MultivariateNormalFullCovariance(y, cov)
            return hnn_.primal_dist(mvn)

    for i in range(num_barycenters):
        j = 0
        key = jr.fold_in(key, i)
        axs = axes[i, :]
        measures, particles, weights = random_measures(
            key=key,
            num_measures=num_marginals,
            num_particles=num_particles,
            dist_fn=dist_fn,
        )
        num_measures = len(measures)
        assert num_measures == num_marginals
        bar_size = num_particles
        bar_weight = _bar_weight(num_measures, unif=True)
        bar_init = jr.uniform(key, shape=(bar_size, 2), minval=0, maxval=1)
        for samples, marginal in zip(particles, measures):
            plot(axs[j], samples, marginal, color="k", cmap="Greys")
            if not marginals_together:
                finalize_axes(axs[j], title=rf"$\mu_{j + 1}$" if i == 0 else None)
                j += 1
        if marginals_together:
            finalize_axes(axs[j], title="Marginals" if i == 0 else None)
            j += 1

        def barycenter(cost_fn):
            return sinkhorn_barycenter(
                cost_fn=cost_fn,
                particles=particles,
                weights=weights,
                bar_init=bar_init,
                bar_weight=bar_weight,
                epsilon=epsilon,
            )

        if not dry:
            cost_fn = hnn_
            for b_idx, beta in enumerate(hnn_betas):
                cost_fn = costs.HNNTanh(beta=beta)
                bar = barycenter(cost_fn)
                if len(hnn_betas) == 1:
                    color = "C0"
                    cmap = "Blues"
                else:
                    color = colors[b_idx]
                    cmap = pu.linear_cmap(colors=["white", color], name=cost_fn.name)
                plot(axs[j], bar, None, color=color, cmap=cmap)
                if duals:
                    dual_cost_fn = costs.DualCost(cost_fn)
                    bar = barycenter(dual_cost_fn)
                    plot(axs[j + 1], bar, None, color=color, cmap=cmap)
            finalize_axes(
                axs[j],
                title=names_map[cost_fn.name] if i == 0 else None,
            )
            if duals:
                finalize_axes(
                    axs[j + 1],
                    title=names_map[costs.DualCost(cost_fn).name] if i == 0 else None,
                )

            j += 2 if duals else 1
            bar = barycenter(w2)
            plot(axs[j], bar, None, color="C0", cmap="Blues")
            finalize_axes(axs[j], title=names_map[w2.name] if i == 0 else None)
            j += 1

    if save:
        plt.savefig(f"hypercube_{num_marginals}x{num_particles}.pdf")
    plt.show()


def unidimensional(n_interp: int = 11, dry: bool = False, verbose: bool = False):
    """GMM barycenters in 1d."""
    import image_data

    key = jr.key(0)
    dim = 1
    x_dim = dim * (dim + 1)
    nbins = 1000
    # cmap = pu.get_cmap("viridis")
    cmap = pu.sns.color_palette("blend:C0,C1", as_cmap=True)

    def mixture(particles: Array) -> dx.MixtureSameFamily:
        means, covs = mu.x_to_means_and_covs(particles, dim, squeeze=True)
        means = jnp.atleast_1d(means)
        covs = jnp.atleast_1d(covs)
        return dx.MixtureSameFamily(
            dx.Categorical(jnp.ones(means.shape[0])),
            dx.Normal(means, jnp.sqrt(covs)),
        )

    def plot_particles(ax, particles: Array, **kwargs):
        pu.plot_2dpc(ax, particles, scatter=True, **kwargs)

    def plot_mixture(ax, particles: Array, **kwargs):
        gmm = mixture(particles)
        pu.draw_1ddensity(ax, jax.vmap(gmm.prob), xlim=xlim, **kwargs)

    def plot(axes, particles: Array, scatter_kw=None, plot_kw=None, **kwargs):
        if scatter_kw is None:
            scatter_kw = dict()
        if plot_kw is None:
            plot_kw = dict()

        scatter_kw = dict(alpha=0.6, s=100, edgecolors="k") | scatter_kw | kwargs
        plot_kw = dict(lw=2, gridsize=nbins) | plot_kw | kwargs
        if len(axes) < 2:
            plot_mixture(axes[0], particles, **plot_kw)
        else:
            plot_particles(axes[0], particles, **scatter_kw)
            plot_mixture(axes[1], particles, **plot_kw)

    def plot_interpolation(ax, particles: Float[Array, "{n_interp} n {x_dim}"]):
        dist = jax.vmap(mixture)(particles)
        x = jnp.linspace(xlim[0], xlim[1], nbins)
        zs = np.linspace(0, 1, n_interp)
        probs = jax.vmap(dist.prob)(x)  # (nbins, n_interp)
        verts = [jnp.vstack((x, y)).T for y in probs.T]  # pyright: ignore
        poly = pu.PolyCollection(verts, facecolors=cmap(zs))
        poly.set_alpha(0.7)
        ax.add_collection3d(poly, zs=zs, zdir="y")
        ax.set_xlabel("$x$")
        ax.set_xlim3d(*xlim)
        ax.set_ylabel("$t$")
        ax.set_ylim3d(0, 1)
        ax.set_zlabel("")
        ax.set_zlim3d(0, probs.max() * 1.01)

    xlim = (0.0, 100.0)
    particles = [
        jnp.asarray([[5.0, 3.0], [20.0, 6.0], [35, 2.0], [40, 4.0]]),
        jnp.asarray([[85.0, 8.0], [60.0, 5.0], [70, 3.0], [95, 2.0]]),
        jnp.asarray([[75.0, 50.0]]),
        jnp.asarray([[25.0, 50.0]]),
    ]
    sizes = tuple(p.shape[0] for p in particles)

    weights = tuple(jnp.ones(s) / s for s in sizes)
    num_marginals = len(particles)
    pu.latexify()
    nrows, ncols = 2, 1
    fig, axes = plt.subplots(
        nrows, ncols, sharex=True, sharey=True, figsize=(8, 6), constrained_layout=True
    )
    axes[0].set_prop_cycle(color=["r", "g", "b", "y"], linestyle=["-", "--", "-.", ":"])
    pu.color_prop_cycle("tab10", 10, ax=axes[-1])
    dist_axes = axes[:-1]
    bary_axes = axes[-1:]
    for i in range(len(particles)):
        plot(dist_axes, particles[i], label=rf"$\nu_{i + 1}$")

    if not dry:
        # barycenters
        bar_size = round(sum(sizes) / len(sizes))
        mean_init = jr.uniform(key, shape=(bar_size,), minval=0, maxval=100)
        cov_init = jr.uniform(key, shape=(bar_size,), minval=0.5, maxval=5.0)
        bar_init = jnp.stack((mean_init, cov_init), axis=-1)
        bar_fn = jax.jit(
            lambda cost_fn, bar_weight: sinkhorn_barycenter(
                cost_fn=cost_fn,
                particles=particles,
                weights=weights,
                bar_init=bar_init,
                bar_weight=bar_weight,
                epsilon=1e-3,
            )
        )
        bar_weights = jnp.full(num_marginals, 1.0 / num_marginals)
        # bar_weights = jr.dirichlet(key, jnp.ones((num_marginals,)), ())
        if verbose:
            print("Barycenter init:", bar_init)
            print("Barycenter weights:", bar_weights)
        bars = []
        cost_fns = (
            # costs.GaussianLPF(dim),
            costs.GaussianLPF(dim).reversed(),
            Bures(dim),
            SqEuclidean(),
            costs.EuclideanGaussianLPF(dim),
        )
        for cost_idx, cost_fn in enumerate(cost_fns):
            try:
                label = get_label(cost_fn)
                bar = bar_fn(cost_fn, bar_weights)
                bars.append((cost_idx, bar))
                if verbose:
                    print(f"{cost_idx + 1}) {label}: {bar.shape}")
                    print(bar)
                plot(bary_axes, bar, label=label)
            except Exception:
                if verbose:
                    print(traceback.format_exc())
                continue

        if num_marginals == 2:
            ncols = len(bars)
            fig_interp = plt.figure(figsize=(5 * ncols, 5))
            gs = fig_interp.add_gridspec(ncols=ncols)
            for i, (cost_idx, bar) in enumerate(bars):
                ax = fig_interp.add_subplot(gs[i], projection="3d")
                try:
                    plot_interpolation(ax, bar)
                    ax.set_title(
                        get_label(cost_fns[cost_idx]) + " Barycenter Interpolation"
                    )
                except Exception:
                    if verbose:
                        print(traceback.format_exc())
                    continue
            fig_interp.tight_layout()
            if not verbose:
                fig_interp.savefig("1d_interpolations.png", bbox_inches="tight")

    format_kw = dict(leg_loc="best", visible_spines="all")
    pu.format_axes(
        axes[0],
        title="Posterior support (marginals)",
        ylabel="Model density",
        **format_kw,
    )
    pu.format_axes(
        axes[1],
        title="Bayes estimator (OT barycenter)",
        xlabel=r"$x$",
        ylabel="Density",
        **format_kw,
        leg_title="Ground cost",
    )
    if not verbose:
        fig.savefig("univariate_gmm.pdf")
    plt.show()


if __name__ == "__main__":
    import os

    os.environ["PAGER"] = "cat"
    fire.Fire(
        {
            "simplex": simplex,
            "hypercube": hypercube,
            "unidimensional": unidimensional,
        }
    )
