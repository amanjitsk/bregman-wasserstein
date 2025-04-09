# Bregman-Wasserstein divergence

Code for [Bregman-Wasserstein divergence: geometry and applications](https://arxiv.org/abs/2302.05833)
(ArXiv preprint).

Install the requirements into a local virtual environment using:

    pip install -r requirements.txt

To run the  barycenter examples in Figures 6 and 7, run

    python bary.py simplex --num_particles=250 --num_marginals=8 --a0=100.0 --noplot_samples
    python bary.py unidimensional

Similarly, the neural OT examples can be run with `neural.py`.

    python neural.py paper

If you find this useful, please consider citing:

    @Article{rankin2023bdg,
      archiveprefix={arXiv},
      author={Rankin, Cale and Wong, Ting-Kam Leonard},
      eprint={2302.05833v1},
      month={Feb},
      primaryclass={math.PR},
      title={Bregman-Wasserstein divergence: geometry and applications},
      url={http://arxiv.org/abs/2302.05833v1},
      year={2023}
    }
