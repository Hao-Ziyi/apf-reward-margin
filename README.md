# Positive Safety Margins from Artificial Potential Field Reward Augmentation

Code, trained models, and per-seed evaluation data for the paper

> Ziyi Hao, *Positive Safety Margins from Artificial Potential Field Reward Augmentation for Size-Robust Obstacle Avoidance under Partial Observability*. Preprint: [arXiv:2611.02947](https://arxiv.org/abs/2611.02947) (under review at Robotics and Autonomous Systems).

A reinforcement-learning controller for a simulated Franka Emika Panda (PandaReach-v3, PyBullet) must
reach a goal past two cylindrical pillars whose size and exact position it does **not** observe
("hidden observation"). A horizontal artificial-potential-field (APF) term added to the reward
induces a positive clearance (47.9 mm vs. 9.6 mm without it) and keeps the safe success rate above
88 % when the pillar radius at test time is larger than the training radius (0.03 m -> 0.055 m).

## Environment

Tested with Python 3.10 on Windows.

```
pip install -r requirements.txt
```

Key versions: `panda-gym==3.0.7`, `stable-baselines3==2.9.0`, `gymnasium==1.3.0`, `torch==2.14.0` (CPU),
`numpy==2.2.6`, `scipy==1.15.3`, `matplotlib==3.9.4`.

## Repository layout

| Path | Content |
|---|---|
| `multi_obstacle_env.py` | Environment wrapper: pillars, random drift, hidden/full observation, the reward of Eq. (1) |
| `train_multi.py` | PPO training of the five reward groups (`ours`, `no_apf`, `no_progress`, `vanilla`, `no_xy_apf`) |
| `verify_radius_generalize.py` | Radius sweep 0.03-0.055 m, produces `radius_generalization.csv` and `radius_cache/` |
| `train_solver.py`, `verify_solver.py`, `solver_common.py` | SAC / TD3 experiments (`models_solver/`, `scan_cache/`, `solver_radius.csv`) |
| `verify_fullobs.py` | Full-observation experiment (`models_fullobs/`, `full_obs_radius.csv`) |
| `classic_apf_baseline.py` | Classical APF planner that reads the pillar positions (`classic_apf_radius.csv`) |
| `pilot_collpen.py`, `summarize_collpen.py` | Larger-collision-penalty control (`models_collpen/`, `collpen_cache/`) |
| `pilot_sens.py`, `summarize_sens.py` | Sensitivity to `d_safe` and `lambda_s` (`models_sens/`, `sens_cache/`, `sens_summary.csv`) |
| `check_overtop.py` | Over-the-top route check |
| `significance_test.py`, `significance_solver.py`, `ci_utils.py` | Confidence intervals and Welch tests |
| `verify_paper_numbers.py` | Recomputes the numbers quoted in the paper from the per-seed data |
| `plot_*.py`, `render_scene.py`, `figstyle.py` | Figure scripts |
| `models_*/`, `*_cache/`, `curves_*/`, `*.csv` | Trained models, per-seed evaluation results, learning curves |

`pilot_inflate.py` / `summarize_inflate.py` and `inflate_cache/` hold an exploratory control
(inflated collision radius) that is not reported in the paper.

## Reproducing the results

All experiments use 150,000 training steps per run and 50 evaluation goals per radius.

```
# PPO, all reward groups, seeds 1-10 (extend --seeds for more); models -> models_formal/
python train_multi.py --groups ours no_apf no_progress vanilla no_xy_apf --seeds 1 2 3

# Radius sweep and per-seed caches
python verify_radius_generalize.py

# SAC / TD3 (ours and no_apf), then evaluation
python train_solver.py --solvers SAC TD3 --groups ours no_apf --seeds 1 2 3
python verify_solver.py

# Full observation
python verify_fullobs.py

# Classical APF planner
python classic_apf_baseline.py

# Larger collision penalty, and (d_safe, lambda_s) sensitivity
python pilot_collpen.py --penalty 1.0 --groups no_apf --seeds 1 2
python pilot_sens.py --dsafe 0.06 --lam 1.5 --seeds 1 2 3
python summarize_sens.py

# Recompute every number quoted in the paper from the raw per-seed data
python verify_paper_numbers.py
```

The paper reports 30 seeds for PPO and 20 for SAC, TD3, and full observation; the exact seed lists are
stored in the cache file names (`radius_cache/{group}_seed{k}.npz`, etc.). Training is
stochastic, so re-running with the same seed may differ slightly across hardware and library
versions; the trained models and per-seed results used in the paper are included here.

## Citation

If you use this code, please cite the paper and the archived release.

```bibtex
@misc{hao2026positive,
  author       = {Hao, Ziyi},
  title        = {Positive Safety Margins from Artificial Potential Field Reward Augmentation
                  for Size-Robust Obstacle Avoidance under Partial Observability},
  year         = {2026},
  eprint       = {2611.02947},
  archivePrefix = {arXiv},
  url          = {https://arxiv.org/abs/2611.02947}
}

@software{hao2026code,
  author    = {Hao, Ziyi},
  title     = {Code and data for ``Positive Safety Margins from Artificial Potential Field
               Reward Augmentation for Size-Robust Obstacle Avoidance under Partial Observability''},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.23885305}
}
```

Archive: [doi:10.5281/zenodo.23885305](https://doi.org/10.5281/zenodo.23885305).

## License

MIT, see `LICENSE`.

## Contact

Ziyi Hao, 1155237839@link.cuhk.edu.hk

