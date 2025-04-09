import fire
from collections.abc import Callable

import numpy as np
import jax
from jax import numpy as jnp, random as jr
import distrax as dx
from jaxtyping import Array, Scalar, PRNGKeyArray, Float
from ott.geometry import pointcloud
from ott.tools.unreg import hungarian, HungarianOutput
import seaborn as sns

import math_utils as mu
import costs
import plot_utils as pu
from matplotlib.patches import ConnectionPatch
from plot_utils import NUM_LEVELS

FILLED = False
CMAP = "Greys"
# SCMAP = sns.color_palette("blend:C0,C1", as_cmap=True)
SCMAP = "Oranges"

CALPHA = 1.0
SCALPHA = 0.0

START, END = 0.35, 0.75
S, E = START, END


def transport(X: Array, Y: Array) -> Array:
    geom = pointcloud.PointCloud(X, Y, cost_fn=costs.Euclidean())
    _, out = hungarian(geom)
    assert out.paired_indices is not None
    return Y[out.paired_indices[1]]


def displacement_interpolations(
    key: PRNGKeyArray,
    source: dx.Distribution,
    target: dx.Distribution,
    cost_fn: costs.AbstractPrimalDual,
    num_t: int,
    num_samples: int,
) -> tuple[
    Float[Array, "{num_t} {num_samples} d"],
    Float[Array, "{num_t} {num_samples} d"],
]:
    source_X = source.sample(seed=key, sample_shape=(num_samples))
    source_Y = cost_fn.to_dual(source_X)
    target_X = target.sample(seed=key, sample_shape=(num_samples))
    target_Y = cost_fn.to_dual(target_X)
    interp = mu.linear_interpolation(num_t)

    # primal displacement interpolation
    pdi = jnp.einsum(
        "n2,2Nd->nNd", interp, jnp.stack((source_X, transport(source_Y, target_X)))
    )
    ddi = cost_fn.to_primal(
        jnp.einsum(
            "n2,2Nd->nNd",
            interp,
            jnp.stack((source_Y, transport(source_X, target_Y))),
        )
    )
    # quadratic (W2) displacement interpolation
    # qdi = jnp.einsum(
    #     "n2,2Nd->nNd", interp, jnp.stack((source_X, transport(source_X, target_X)))
    # )
    return pdi, ddi


def plot(
    key: PRNGKeyArray,
    source: dx.Distribution,
    target: dx.Distribution,
    cost_fn: costs.AbstractPrimalDual,
    estimator: Callable[[Array], tuple[dx.Distribution, dx.Distribution]] | None = None,
    plim: tuple[float, float] = (0, 1),
    dlim: tuple[float, float] = (-4, 4),
    is_simplex: bool = False,
    num_t: int = 5,
    num_t_power: int = 1,
    num_samples: int = 100,
    num_plot_samples: int | None = None,
    together: bool = False,
    save: bool = False,
):
    if not together:
        global FILLED
        FILLED = True
        global CMAP
        CMAP = "viridis"
        global SCMAP
        SCMAP = CMAP

    step_t = (num_t**num_t_power - 1) // (num_t - 1)
    num_t_ = num_t
    num_t **= num_t_power
    pdi, ddi = displacement_interpolations(
        key, source, target, cost_fn, num_t=num_t, num_samples=num_samples
    )

    if num_plot_samples is None or not together:
        num_plot_samples = num_samples
    assert num_plot_samples <= num_samples

    # sample_indices = jr.choice(key, num_samples, (num_plot_samples,), replace=False)
    step, start = divmod(num_samples, num_plot_samples)
    sample_indices = jnp.arange(start, num_samples, step)

    if estimator is None:

        def estimator(samples):
            dual_samples = cost_fn.to_dual_(samples)
            primal_dist = mu.KDE(samples)
            dual_dist = mu.KDE(dual_samples)
            return primal_dist, dual_dist

    def plot(axs, samples, dist):
        dual_samples = cost_fn.to_dual_(samples)
        if dist is None:
            dist, dual_dist = estimator(samples)
        else:
            dual_dist = cost_fn.dual_dist(dist)

        pax, dax = axs
        scatter_kw = dict(c=scmap(np.linspace(S, E, num_plot_samples)), alpha=SCALPHA)
        pu.draw_scatter(
            pax,
            samples,
            density=jax.vmap(dist.prob),
            kde=False,
            kde_kw={},
            sample=False,
            sample_indices=sample_indices,
            scatter_kw=scatter_kw,
            is_simplex=is_simplex,
            xlim=plim,
            ylim=None,
            cut=3,
            coverage=1.0,
            marginals=True,
            levels=NUM_LEVELS,
            filled=FILLED,
            cmap=cmap,
            alpha=CALPHA,
        )
        pu.draw_scatter(
            dax,
            dual_samples,
            density=jax.vmap(dual_dist.prob),
            kde=False,
            kde_kw={},
            sample=False,
            sample_indices=sample_indices,
            scatter_kw=scatter_kw,
            is_simplex=False,
            xlim=None,
            ylim=None,
            cut=3,
            coverage=1.0,
            marginals=True,
            levels=NUM_LEVELS,
            filled=FILLED,
            cmap=cmap,
            alpha=CALPHA,
        )

    def finalize_axes(axs):
        pax, dax = axs
        if is_simplex:
            # pu.setup_simplex(pax, p="" if not together else "p")
            pu.setup_simplex(pax)
        else:
            pax.set(xlim=plim, ylim=plim)
            pu.format_axes(pax, xticks=plim, yticks=plim, nbins=2, visible_spines="all")
        dax.set(xlim=dlim, ylim=dlim)
        pu.format_axes(dax, xticks=dlim, yticks=dlim, nbins=2, visible_spines="all")

    def add_connections(fig, axes, samples):
        """Add connections between axes denoting transport."""
        kwargs = dict(lw=2)
        paxes, daxes = axes
        assert samples.shape == (num_t, num_samples, 2)
        dual_samples = cost_fn.to_dual(samples)
        if is_simplex:
            samples = pu.bc2xy(mu.s_to_p(samples))
        colors = cmap(np.linspace(0, 1, num_plot_samples))
        for j in range(num_t - 1):
            for i, s_i in enumerate(sample_indices):
                fig.add_artist(
                    ConnectionPatch(
                        xyA=samples[j][s_i],
                        xyB=samples[j + 1][s_i],
                        coordsA="data",
                        coordsB="data",
                        axesA=paxes[j],
                        axesB=paxes[j + 1],
                        color=colors[i],
                        **kwargs,
                    )
                )
                fig.add_artist(
                    ConnectionPatch(
                        xyA=dual_samples[j][s_i],
                        xyB=dual_samples[j + 1][s_i],
                        coordsA="data",
                        coordsB="data",
                        axesA=daxes[j],
                        axesB=daxes[j + 1],
                        color=colors[i],
                        **kwargs,
                    )
                )

    def add_lines(ax, samples):
        pax, dax = ax
        assert samples.shape == (num_t, num_samples, 2)
        # shape (num_samples, num_t, 2)
        samples = jnp.transpose(samples, (1, 0, 2))
        dual_samples = cost_fn.to_dual(samples)
        if is_simplex:
            samples = mu.s_to_p(samples)
        colors = scmap(np.linspace(S, E, num_plot_samples))
        scatter_kw = dict(alpha=SCALPHA)
        time_indices = np.arange(0, num_t, step_t)
        lines = []
        for i, s_i in enumerate(sample_indices):
            color = colors[i]
            scatter_kw["color"] = color
            if is_simplex:
                lines.append(
                    pu.plot_simplex(pax, samples[s_i], scatter=False, color=color)[0]
                )
                pu.plot_simplex(
                    pax, samples[s_i][time_indices], scatter=True, **scatter_kw
                )
            else:
                lines.append(
                    pu.plot_2dpc(pax, samples[s_i], scatter=False, color=color)[0]
                )
                pu.plot_2dpc(
                    pax, samples[s_i][time_indices], scatter=True, **scatter_kw
                )
            lines.append(
                pu.plot_2dpc(dax, dual_samples[s_i], scatter=False, color=color)[0]
            )
            pu.plot_2dpc(
                dax, dual_samples[s_i][time_indices], scatter=True, **scatter_kw
            )

        for line in lines:
            pu.add_arrow(line, sample_indices=list(range(0, num_t, step_t)))

        finalize_axes(ax)

    pu.latexify()
    nrows = 2
    ncols = num_t_ if not together else 2
    cmap = pu.get_cmap(CMAP)
    scmap = pu.get_cmap(SCMAP)
    # cmap = sns.color_palette("blend:C0,C1", as_cmap=True)
    size = 3 if together else 2
    if not together:
        for i in range(2):
            interpolation = pdi if i else ddi
            fig, subfigs, axes = pu.make_figure(nrows, ncols, size)
            for j_idx, j in enumerate(range(0, num_t, step_t)):
                if j == 0:
                    dist = source
                elif j == num_t - 1:
                    dist = target
                else:
                    dist = None

                plot(axes[:, j_idx], samples=interpolation[j], dist=dist)
                finalize_axes(axes[:, j_idx])
            # add_connections(fig, axes, interpolation, cmap)
    else:
        fig, subfigs, axes = pu.make_figure(nrows, ncols, size)
        for i in range(2):
            global CALPHA
            global SCALPHA
            interpolation = pdi if i else ddi
            # roll so that initial and final distributions are plotted last
            for j in np.roll(range(0, num_t, step_t), -1):
                if j == 0:
                    dist = source
                    cmap = pu.get_cmap("Blues")
                    CALPHA = 1.0
                    SCALPHA = 1.0
                elif j == num_t - 1:
                    dist = target
                    cmap = pu.get_cmap("Greens")
                    CALPHA = 1.0
                    SCALPHA = 1.0
                else:
                    dist = None
                    cmap = pu.get_cmap(CMAP)
                    CALPHA = 0.5
                    SCALPHA = 1.0
                plot(axes[:, i], samples=interpolation[j], dist=dist)
            add_lines(axes[:, i], interpolation)

        labels = (r"$\mu_0$", r"$\mu_t$", r"$\mu_1$")
        colors = ("C0", "grey", "C2")
        handles = pu.create_legend_handles(labels, colors=colors, bar=True, alpha=0.5)
        subfigs[0].legend(
            handles=handles,
            loc="outside lower center",
            frameon=False,
            ncol=3,
        )

    if save and together:
        pu.plt.savefig(f"matching_{num_t_}x{num_plot_samples}.pdf")

    pu.plt.show()


def simplex(
    num_t: int = 5,
    num_samples: int = 100,
    num_plot_samples: int | None = 3,
    together: bool = True,
    save: bool = True,
):
    kl = costs.SimplexKL()
    # ilr = costs.SimplexILR()

    def estimator(samples):
        dual_dist = mu.mvn_from_samples(kl.to_dual(samples))
        primal_dist = kl.primal_dist(dual_dist)
        return primal_dist, dual_dist

    if together:
        source = mu.MinimalSimplex(dx.Dirichlet(30 * jnp.r_[0.45, 0.45, 0.1]))
        target = mu.MinimalSimplex(dx.Dirichlet(50 * jnp.r_[0.05, 0.45, 0.5]))
    else:
        source = kl.primal_dist(
            dx.MultivariateNormalFullCovariance(
                jnp.zeros(2), jnp.array([[1, 0.5], [0.5, 1]])
            )
        )
        target = kl.primal_dist(
            dx.MultivariateNormalFullCovariance(
                jnp.zeros(2), jnp.array([[1, -0.5], [-0.5, 1]])
            )
        )

    plot(
        key=jr.PRNGKey(0),
        source=source,
        target=target,
        cost_fn=kl,
        estimator=estimator,
        dlim=(-4.0, 4.0),
        is_simplex=True,
        num_t=num_t,
        num_t_power=3,
        num_samples=num_samples,
        num_plot_samples=num_plot_samples,
        together=together,
        save=save,
    )


if __name__ == "__main__":
    fire.Fire({"simplex": simplex})
