# Robust Adersarial Imitation Learning via Adaptively-Selected Demonstrations
This repository contains the PyTorch code for the paper "Robust Adersarial Imitation Learning via Adaptively-Selected Demonstrations" in IJCAI 2021.

## Requirements
Experiments were run with Python 3.7 and these packages:
* pytorch == 1.4.0
* gym == 0.15.6
* mujoco-py == 2.0.2.9

## Train SAIL
The experimental results are available in the paper.
 * SAIL with soft weight
 ```
  python sail_main.py --env_id 2/4/5/7 --il_method sail --weight --soft --seed 0
 ```
 * SAIL with binary weight
 ```
  python sail_main.py --env_id 2/4/5/7 --il_method sail --weight --seed 0
 ```
 * GAIL/AIRL/VAIL/InfoGAIL
 ```
  python sail_main.py --env_id 2/4/5/7 --il_method gail/airl/vail/infogail --seed 0
 ```

For other compared methods, the implementation can be found in core/irl.py and trex_main.py for [Two-IWIL/IC-GAIL](https://github.com/kristery/Imitation-Learning-from-Imperfect-Demonstration) and [D-REX](https://github.com/dsbrown1331/CoRL2019-DREX). 

After saving the policy model during traning, you can plot the optimization trajectories in Figure 2 under the instruction of https://github.com/tomgoldstein/loss-landscape.

## Acknowledegement
The code structure is based on the source code of ICML20 [VILD](https://github.com/voot-t/vild_code). We also use expert data collected by VILD.

## Reference
[1] Generative adversarial imitation learning. NeurIPS 2016.

[2] Learning robust rewards with adversarial inverse reinforcement learning. ICLR 2018.

[3] Variational discriminator bottleneck: Improving imitation learning, inverse rl, and gans by constraining information flow. ICLR 2017.

[4] InfoGAIL: Interpretable Imitation Learning from Visual Demonstrations. NeurIPS 2017

[5] Imitation learning from imperfect demonstration. ICML 2019.

[6] Extrapolating beyond suboptimal demonstrations via inversere inforcement learning from observations. ICML 2019.

[7] Better-than-demonstrator imitation learning via automatically-ranked demonstrations. CoRL 2020.

[8] Variational Imitation Learning with Diverse-quality Demonstrations. ICML 2020.
