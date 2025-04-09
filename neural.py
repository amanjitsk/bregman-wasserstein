import fire
import traceback
from pathlib import Path
from typing import Optional, Any
from functools import partial
from pathvalidate import sanitize_filename
import numpy as np

import jax
import equinox as eqx
from jax import numpy as jnp, random as jr, nn as jnn
from jaxtyping import Array, Float, PRNGKeyArray

import optax
from flax import linen as nn

from ott import datasets
from ott.geometry import costs
from ott import utils

# from ott.problems.linear import potentials as dual_potentials
import expectile_neural_dual
from ott.neural.networks.potentials import BasePotential, MLP

# from matplotlib import pyplot as plt
import data
import math_utils as mu
import plot_utils as pu
from costs import AbstractBregman, SqEuclidean

plt = pu.plt
IN_COLOR, OUT_COLOR, MAP_COLOR = pu.IN_COLOR, pu.OUT_COLOR, pu.MAP_COLOR
PATH_COLOR = pu.PATH_COLOR

ENOTPotentials = expectile_neural_dual.ENOTPotentials
OptionalKey = Optional[PRNGKeyArray]


class MongeMap(BasePotential):
    kantorovich: BasePotential
    cost_fn: costs.CostFn
    forward: bool = True

    @property
    def is_potential(self):
        return False

    @nn.compact
    def __call__(self, x: Array) -> Array:
        if self.kantorovich.is_potential:
            z = jax.grad(self.kantorovich)(x)
        else:
            z = self.kantorovich(x)
        return self.cost_fn.twist_operator(x, z, not self.forward)


class DualCoords(eqx.Module):
    bregman: AbstractBregman
    variable: bool = eqx.field(static=True)
    primal: Array
    dual: Array

    def __init__(
        self,
        bregman: AbstractBregman,
        primal: Optional[Array] = None,
        dual: Optional[Array] = None,
    ):
        assert primal is not None or dual is not None
        # 0 (False) for dual, 1 (True) for primal
        self.variable = primal is not None
        self.bregman = bregman
        if primal is not None:
            self.primal = primal
            self.dual = self.bregman.to_dual(primal)
        elif dual is not None:
            self.dual = dual
            self.primal = self.bregman.to_primal(dual)
        else:
            raise ValueError

    @property
    def passed(self):
        if self.variable:
            return self.primal
        return self.dual

    @property
    def mirrored(self):
        if self.variable:
            return self.dual
        return self.primal

    def clone(self, samples: Optional[Array] = None):
        samples = (
            (self.primal if self.variable else self.dual).copy()
            if samples is None
            else samples
        )
        if self.variable:
            return DualCoords(self.bregman, primal=samples)
        return DualCoords(self.bregman, dual=samples)

    def get(self, primal: bool = True):
        return self.primal if primal else self.dual

    def to_dict(self):
        return {"primal": self.primal, "dual": self.dual}


class BregmanPotentials(eqx.Module):
    bregman: AbstractBregman
    dual_potentials: ENOTPotentials = eqx.field(static=True)
    ddi: bool = eqx.field(static=True)

    def __init__(
        self,
        bregman: AbstractBregman,
        dual_potentials: ENOTPotentials,
        ddi: bool = True,
    ):
        self.bregman = bregman
        self.dual_potentials = dual_potentials
        self.ddi = ddi

    @property
    def is_bidirectional(self):
        return self.dual_potentials.is_bidirectional

    def mixed_transport(self, samples: Array, forward: bool = True) -> Array:
        return self.dual_potentials.transport(samples, forward)

    def coords(self, **kwargs) -> DualCoords:
        return DualCoords(self.bregman, **kwargs)

    def source(self, samples: Array) -> DualCoords:
        if self.ddi:
            return self.coords(primal=samples)
        else:
            return self.coords(dual=samples)

    def target(self, samples: Array) -> DualCoords:
        if self.ddi:
            return self.coords(dual=samples)
        else:
            return self.coords(primal=samples)

    def samples(self, samples: Array, forward: bool = True) -> DualCoords:
        if self.ddi != forward:
            return self.coords(dual=samples)
        else:
            return self.coords(primal=samples)

    def transport(self, samples: Array, forward: bool = True) -> DualCoords:
        cross = self.mixed_transport(samples, forward)
        if self.ddi != forward:
            # samples must in dual coordinates
            # mu_1^Y -> mu_0^X OR mu_0^Y -> mu_1^X
            mapped_samples = self.coords(primal=cross)
        else:
            # samples must in primal coordinates
            # mu_0^X -> mu_1^Y OR mu_1^X -> mu_0^Y
            mapped_samples = self.coords(dual=cross)
        return mapped_samples

    def get_dict(self, source: Array, target: Array):
        D = {}
        D["source"] = self.source(source).to_dict()
        D["target"] = self.target(target).to_dict()
        D["push"] = self.transport(source, forward=True).to_dict()
        D["pull"] = self.transport(target, forward=False).to_dict()
        D["ddi"] = self.ddi
        # D["bregman"] = self.bregman
        return D

    def plot_ot_map(
        self,
        source: jnp.ndarray,
        target: jnp.ndarray,
        samples: Optional[jnp.ndarray] = None,
        forward: bool = True,
        primal: bool = True,
        ax: Optional[pu.mpa.Axes] = None,
        scatter_kwargs: Optional[dict[str, Any]] = None,
        legend_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, pu.mpa.Axes]:
        """Plot data and learned optimal transport map.

        Args:
          source: samples from the source measure
          target: samples from the target measure
          samples: extra samples to transport, either ``source`` (if ``forward``) or
            ``target`` (if not ``forward``) by default.
          forward: use the forward map from the potentials if ``True``,
            otherwise use the inverse map.
          ax: axis to add the plot to
          scatter_kwargs: additional kwargs passed into
            :meth:`~matplotlib.axes.Axes.scatter`
          legend_kwargs: additional kwargs passed into
            :meth:`~matplotlib.axes.Axes.legend`

        Returns:
          Figure and axes.
        """
        if scatter_kwargs is None:
            scatter_kwargs = {"alpha": 0.5}
        if legend_kwargs is None:
            legend_kwargs = {
                "ncol": 3,
                "loc": "upper center",
                "bbox_to_anchor": (0.5, -0.05),
                "edgecolor": "k",
            }

        if ax is None:
            fig = plt.figure(facecolor="white")
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        # plot the source and target samples
        if forward:
            label_transport = r"$\nabla f(source)$"
            source_color, target_color = IN_COLOR, OUT_COLOR
        else:
            label_transport = r"$\nabla g(target)$"
            source_color, target_color = OUT_COLOR, IN_COLOR

        source_: DualCoords = self.source(source)
        target_: DualCoords = self.target(target)

        source = source_.get(primal=primal)
        target = target_.get(primal=primal)

        ax.scatter(
            source[:, 0],
            source[:, 1],
            color=source_color,
            label="source",
            **scatter_kwargs,
        )
        ax.scatter(
            target[:, 0],
            target[:, 1],
            color=target_color,
            label="target",
            **scatter_kwargs,
        )

        # plot the transported samples
        samples_ = (
            (source_ if forward else target_)
            if samples is None
            else self.samples(samples, forward=forward)
        )
        samples = samples_.get(primal=primal)
        transported_samples = self.transport(samples_.passed, forward=forward).get(
            primal=primal
        )
        ax.scatter(
            transported_samples[:, 0],
            transported_samples[:, 1],
            color=MAP_COLOR,
            label=label_transport,
            **scatter_kwargs,
        )

        U, V = (transported_samples - samples).T
        ax.quiver(
            samples[:, 0],
            samples[:, 1],
            U,
            V,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            headwidth=10,
            headlength=10,
            color=PATH_COLOR,
        )

        ax.legend(**legend_kwargs)
        return fig, ax

    def interpolate(
        self, A: Float[Array, "N D"], B: Float[Array, "N D"], num_interp: int
    ) -> Float[Array, "{num_interp} N D"]:
        """Interpolate between A and B.
        Precondition: A and B have the same shape.
        """
        T = mu.linear_interpolation(num_interp)
        C = jnp.stack((A, B))
        return jnp.einsum("n2,2ND->nND", T, C)

    def plot_mixed(self, source: Array, target: Array):
        if self.ddi:
            kw = dict(source_label=r"$\mu_0^X$", target_label=r"$\mu_1^Y$")
        else:
            kw = dict(source_label=r"$\mu_0^Y$", target_label=r"$\mu_1^X$")
        batch = dict(source=source, target=target)
        # forward
        batch["push"] = self.mixed_transport(source, forward=True)
        ffig = pu.plot_transport(batch, forward=True, overlap=False, **kw)
        # inverse
        batch["push"] = self.mixed_transport(target, forward=False)
        ifig = pu.plot_transport(batch, forward=False, overlap=False, **kw)
        return ffig, ifig

    def plot_evolution(self, source: Array, target: Array, num_interp: int = 20):
        mu_0 = self.source(source)
        mu_1 = self.target(target)

        # source -> target (Forward)
        push = self.transport(mu_0.passed, forward=True)
        push_t = self.interpolate(mu_0.mirrored, push.passed, num_interp)
        push_t = push.clone(push_t).get()

        # target -> source (Inverse)
        pull = self.transport(mu_1.passed, forward=False)
        pull_t = self.interpolate(mu_1.mirrored, pull.passed, num_interp)
        pull_t = pull.clone(pull_t).get()

        nrows, ncols = 1, 2
        pu.latexify()
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols, 4 * nrows),
            gridspec_kw={"wspace": 0, "hspace": 0},
        )

        # compute margins and axis limits
        margin = 0.05
        inputs = jnp.concat((mu_0.get(), mu_1.get()), axis=0)
        x_lim, y_lim = pu.compute_lims(inputs, margin=margin)

        s = 5
        # forward
        ax = axes[0]
        f_sc = ax.scatter(push_t[0, :, 0], push_t[0, :, 1], s=s, color="k")
        ax.set(xlim=x_lim, ylim=y_lim)
        ax.set_axis_off()
        # inverse
        ax = axes[1]
        i_sc = ax.scatter(pull_t[0, :, 0], pull_t[0, :, 1], s=s, color="k")
        ax.set(xlim=x_lim, ylim=y_lim)
        ax.set_axis_off()

        def animate(i):
            f_sc.set_offsets(push_t[i])
            i_sc.set_offsets(pull_t[i])
            return f_sc, i_sc

        ani = pu.FuncAnimation(fig, animate, frames=num_interp, interval=50, blit=True)
        fig.tight_layout()
        return fig, axes, ani

    def plot_displacement(self, source: Array, target: Array, num_interp: int = 20):
        power = 3
        num_t = num_interp**power
        # step_t = (num_t - 1) // (num_interp - 1)

        mu_0 = self.source(source)
        mu_1 = self.target(target)
        source = mu_0.get()
        target = mu_1.get()

        # push_t shape (num_t, num_samples, 2)
        if self.ddi:
            # source -> target (Forward)
            push = self.transport(mu_0.passed, forward=True)
            push_t = self.interpolate(mu_0.mirrored, push.passed, num_t)
            push_t = push.clone(push_t).get()
            source_color, target_color = IN_COLOR, OUT_COLOR
        else:
            # target -> source (Inverse)
            push = self.transport(mu_1.passed, forward=False)
            push_t = self.interpolate(mu_1.mirrored, push.passed, num_t)
            push_t = push.clone(push_t).get()
            source_color, target_color = OUT_COLOR, IN_COLOR

        fig, ax = plt.subplots(figsize=(4, 4))

        s = 5
        ax.scatter(source[:, 0], source[:, 1], s=s, color=source_color)
        ax.scatter(target[:, 0], target[:, 1], s=s, color=target_color)
        ax.autoscale(enable=False)
        ax.scatter(push_t[-1, :, 0], push_t[-1, :, 1], s=s, color=MAP_COLOR)
        lines = pu.LineCollection(jnp.unstack(push_t, axis=1), colors=PATH_COLOR)
        ax.add_collection(lines)
        # cmap = pu.unicmap(MAP_COLOR)
        # colors = cmap(np.linspace(0, 1, num_interp))
        # for t_idx, t in enumerate(range(0, num_t, step_t)):
        #     ax.scatter(push_t[t, :, 0], push_t[t, :, 1], s=s, color=colors[t_idx])
        ax.set_axis_off()

        return fig

    def plot_interpolations(self, source: Array, target: Array, num_interp: int = 20):
        decorations = False
        mu_0 = self.source(source)
        mu_1 = self.target(target)

        source = mu_0.get()
        target = mu_1.get()

        # source -> target (Forward)
        push = self.transport(mu_0.passed, forward=True)
        push_t = self.interpolate(mu_0.mirrored, push.passed, num_interp)
        push_t = push.clone(push_t).get()

        # target -> source (Inverse)
        pull = self.transport(mu_1.passed, forward=False)
        pull_t = self.interpolate(mu_1.mirrored, pull.passed, num_interp)
        pull_t = pull.clone(pull_t).get()

        nrows, ncols = 1, 2
        pu.latexify()
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols, 4 * nrows),
            gridspec_kw={"wspace": 0, "hspace": 0},
        )
        if decorations:
            labels = [
                "Input samples",
                "Output samples",
                "Transported Samples",
                "Transport Paths",
            ]
            colors = [IN_COLOR, OUT_COLOR, MAP_COLOR, PATH_COLOR]
            handles = pu.create_legend_handles(labels, colors=colors, bar=True)
            fig.legend(
                handles,
                labels,
                loc="outside lower center",
                ncol=4,
                frameon=False,
                bbox_to_anchor=(0.5, -0.1),
            )

        def quiver(ax, A, B):
            U, V = (B - A).T
            Q = ax.quiver(
                A[:, 0],
                A[:, 1],
                U,
                V,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.003,
                headwidth=1,
                headlength=0,
                headaxislength=0,
                color=PATH_COLOR,
            )
            return Q

        s = 5
        title_kw = dict(fontsize=18, color="k")
        # forward
        ax = axes[0]
        ax.scatter(source[:, 0], source[:, 1], s=s, color=IN_COLOR)
        ax.scatter(target[:, 0], target[:, 1], s=s, color=OUT_COLOR)
        f_sc = ax.scatter(push_t[0, :, 0], push_t[0, :, 1], s=s, color=MAP_COLOR)
        fQ = quiver(ax, source, push_t[0, :, :])
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_axis_off()
        if decorations:
            ax.set_title(r"$(\grad \Omega^* \circ \grad \phi)_{\#}\alpha$", **title_kw)
        # inverse
        ax = axes[1]
        ax.scatter(source[:, 0], source[:, 1], s=s, color=OUT_COLOR)
        ax.scatter(target[:, 0], target[:, 1], s=s, color=IN_COLOR)
        i_sc = ax.scatter(pull_t[0, :, 0], pull_t[0, :, 1], s=s, color=MAP_COLOR)
        iQ = quiver(ax, target, pull_t[0, :, :])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_axis_off()
        if decorations:
            ax.set_title(r"$(\grad \phi^* \circ \grad \Omega)_{\#}\beta$", **title_kw)

        def animate(i):
            f_sc.set_offsets(push_t[i])
            i_sc.set_offsets(pull_t[i])
            U, V = (push_t[i] - source).T
            fQ.set_UVC(U, V)
            U, V = (pull_t[i] - target).T
            iQ.set_UVC(U, V)
            return f_sc, i_sc, fQ, iQ

        ani = pu.FuncAnimation(fig, animate, frames=num_interp, interval=50, blit=True)
        fig.tight_layout()
        return fig, axes, ani


def run(
    bregman: AbstractBregman,
    sampler: data.AbstractPairedSampler,
    train_batch_size: int,
    valid_batch_size: int,
    is_bidirectional: bool = True,
    dual_interpolation: bool = True,
    regularizer_strength: float = 1.0,
    num_train_iters: int = 5000,
    plot_interpolations: bool = True,
    plot_paper: bool = False,
    dim_data: int = 2,
    prefix: str = "",
    key: OptionalKey = None,
):
    neural_f = MLP(
        dim_hidden=[64, 64, 64, 64, 1 if is_bidirectional else dim_data],
        act_fn=jnn.elu,
    )
    neural_g = MLP(dim_hidden=[64, 64, 64, 64, 1], act_fn=jnn.elu)
    lr_schedule_f = optax.cosine_decay_schedule(
        init_value=5e-4, decay_steps=num_train_iters, alpha=1e-4
    )
    lr_schedule_g = optax.cosine_decay_schedule(
        init_value=5e-4, decay_steps=num_train_iters, alpha=1e-4
    )
    optimizer_f = optax.adam(learning_rate=lr_schedule_f, b1=0.9, b2=0.999)
    optimizer_g = optax.adam(learning_rate=lr_schedule_g, b1=0.9, b2=0.999)

    neural_dual_solver = expectile_neural_dual.ExpectileNeuralDual(
        dim_data,
        neural_f,
        neural_g,
        optimizer_f,
        optimizer_g,
        cost_fn=SqEuclidean(),
        is_bidirectional=is_bidirectional,
        expectile=0.99,
        expectile_loss_coef=regularizer_strength,
        num_train_iters=num_train_iters,
        logging=False,
    )

    rng = utils.default_prng_key(key)
    mirror_sampler = data.MirrorSampler(sampler, bregman, dual_interpolation)
    source, target = mirror_sampler.sample(rng, valid_batch_size)
    potential_fn = lambda w2_potentials: BregmanPotentials(
        bregman, w2_potentials, ddi=dual_interpolation
    )

    # def callback(step, potentials):
    #     if (step + 1) % 5_000 == 0:
    #         print(f"Training iteration: {step}/{num_train_iters}")

    #         neural_dual_dist = potentials.distance(source, target)
    #         print(
    #             f"Neural dual distance between source and target data: {neural_dual_dist:.2f}"
    #         )
    #         potentials = potential_fn(potentials)
    #         potentials.plot_ot_map(source, target, forward=True)
    #         plt.show()

    potentials = neural_dual_solver(
        sampler=mirror_sampler,
        train_batch_size=train_batch_size,
        valid_batch_size=valid_batch_size,
        callback=None,
        # callback=callback,
    )
    potentials = potential_fn(potentials)
    # saving
    plot_dir = Path("plot")
    plot_dir.mkdir(parents=True, exist_ok=True)
    fname = str(plot_dir.joinpath(sanitize_filename(prefix + bregman.name)))
    savefig_kw = dict(bbox_inches="tight")

    if plot_paper:
        fig = potentials.plot_displacement(source, target)
        fig.savefig(f"{fname}_disint.pdf", **savefig_kw)
        info = potentials.get_dict(source, target)
        jnp.save(f"{fname}_summary.npy", info)
    else:
        if plot_interpolations:
            fig, _, ani = potentials.plot_interpolations(source, target)
        else:
            fig, _, ani = potentials.plot_evolution(source, target)
        ani.save(f"{fname}.gif", writer=pu.PillowWriter(fps=10))
        # fig.savefig(fname.with_suffix(".png"), **savefig_kw)
        del ani
    # plt.show()
    plt.close(fig)
    if plot_paper:
        ffig, ifig = potentials.plot_mixed(source, target)
        if potentials.ddi:
            ffig.savefig(f"{fname}_forward.pdf", **savefig_kw)
        else:
            ifig.savefig(f"{fname}_inverse.pdf", **savefig_kw)
        plt.close(ffig)
        plt.close(ifig)

    return potentials


def main(
    name: data.Datasets = "demo",
    ddi: bool = True,
    paper: bool = False,
    train_iters: int = 100_000,
    hypercube: bool = False,
):
    from costs import Euclidean, HNNTanh, ExtendedKL

    source, target = data.make_baseline(name)
    if hypercube:
        sampler = data.Hypercube(source, target)
        prefix = "hypercube"
    else:
        sampler = data.Independent(source, target)
        prefix = "vanilla"
    prefix += "_" + ("ddi" if ddi else "pdi")

    run_fn = partial(
        run,
        sampler=sampler,
        train_batch_size=1024,
        valid_batch_size=1024,
        is_bidirectional=True,
        dual_interpolation=ddi,
        regularizer_strength=0.3,
        num_train_iters=train_iters,
        plot_interpolations=not name.startswith("huang"),
        plot_paper=paper,
        dim_data=2,
        prefix=f"{prefix}_{name}_",
    )

    # Bregman cost functions
    cost_fns: list[AbstractBregman] = []
    cost_fns.append(Euclidean())
    if hypercube:
        cost_fns.append(HNNTanh(beta=1.0))
        cost_fns.append(ExtendedKL(a=1.0))
    else:
        cost_fns.append(HNNTanh(beta=1.0).dualized())
        for a in [-1.0, 1.0]:
            cost_fns.append(ExtendedKL(a=a).dualized())

    for bregman in cost_fns:
        print(f"Dataset: {name}, Cost: {bregman.name}")
        run_fn(bregman=bregman)


def paper(num_iters: int = 100_000):
    """Run the experiments for the paper."""
    run_fn = partial(main, paper=True, train_iters=num_iters, hypercube=False)
    for dset in ("demo", "rout_2"):
        run_fn(name=dset, ddi=True)
        run_fn(name=dset, ddi=False)


if __name__ == "__main__":
    fire.Fire({"main": main, "paper": paper})
