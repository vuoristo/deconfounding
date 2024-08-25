Deconfounding Imitation Learning with Variational Inference
===========================================================

This repository contains the implementation of experiments for "Deconfounding Imitation Learning with Variational Inference".

## Example

Run with e.g.
```
python -m sb3based.rnnbc \
--epochs 1000 \
--folder logs \
--bc-train-steps 100 \
--bc-batch-size 16 \
--n-episodes 10 \
--n-envs 10 \
--bc-lr 1e-4 \
--env LunarLander-v2 \
--max-episode-length 600 \
--buffer-size 1000000 \
--enc-train-steps 100 \
--enc-lr 1e-4 \
--enc-batch-size 16 \
--bptt-steps 600 \
--kl-weight 0.0 \
--device cuda:0 \
--deconfounding
```

Command for training an expert for LunarLander

```
python3 -m rl_zoo3.train \
--algo ppo \
--env LunarLander-v2 \
--conf-file sb3based/lunar_hparams.py \
--save-freq 100000
```

For running vectorized environments in `ray` experiments, you may need to disable the use of `SubprocVecEnv`. Disabling worked by commenting the `SubprocVecEnv` stuff from `~/.local/lib/python3.10/site-packages/rl_zoo3/utils.py`

## Citation

```
@article{vuorio2022deconfounded,
  title={Deconfounded imitation learning},
  author={Vuorio, Risto and Brehmer, Johann and Ackermann, Hanno and Dijkman, Daniel and Cohen, Taco and de Haan, Pim},
  journal={arXiv preprint arXiv:2211.02667},
  year={2022}
}
```