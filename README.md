# Robust Adversarial Imitation Learning via Adaptively-Selected Demonstrations

This repository contains the PyTorch code for the paper "Robust Adversarial Imitation Learning via Adaptively-Selected Demonstrations" in IJCAI 2021. [[Paper](https://www.ijcai.org/proceedings/2021/434)][[Appendix](https://github.com/yunke-wang/yunke-wang.github.io/blob/main/docs/IJCAI__21_Robust_Adversarial_Imitation_Learning_via_Adaptively_Selected_Demonstrations.pdf)]

## Requirements
Experiments were run with Python 3.6 and these packages:
* pytorch == 1.1.0
* gym == 0.15.7
* mujoco-py == 2.0.2.9

## Train SAIL

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

For other compared methods, the re-implementation of [2IWIL/IC-GAIL](https://github.com/kristery/Imitation-Learning-from-Imperfect-Demonstration) and [D-REX](https://github.com/dsbrown1331/CoRL2019-DREX) can be found in core/irl.py and trex_main.py. 

After saving the policy model during training, you can plot the optimization trajectories in Figure 2 under the instruction of https://github.com/tomgoldstein/loss-landscape.

## Acknowledegement
We thank the authors of [VILD](https://github.com/voot-t/vild_code). Our code structure is based on their source code and we also use expert data collected by VILD.

## Contact
For any questions, please feel free to contact me. (Email: yunke.wang@whu.edu.cn)

## Citation
```
@inproceedings{ijcai2021-0434,
  title     = {Robust Adversarial Imitation Learning via Adaptively-Selected Demonstrations},
  author    = {Wang, Yunke and Xu, Chang and Du, Bo},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Zhi-Hua Zhou},
  pages     = {3155--3161},
  year      = {2021},
  month     = {8},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2021/434},
  url       = {https://doi.org/10.24963/ijcai.2021/434},
}
```

## Reference
[1] Generative adversarial imitation learning. NeurIPS 2016.

[2] Learning robust rewards with adversarial inverse reinforcement learning. ICLR 2018.

[3] Variational discriminator bottleneck: Improving imitation learning, inverse rl, and gans by constraining information flow. ICLR 2017.

[4] InfoGAIL: Interpretable Imitation Learning from Visual Demonstrations. NeurIPS 2017

[5] Imitation learning from imperfect demonstration. ICML 2019.

[6] Extrapolating beyond suboptimal demonstrations via inversere inforcement learning from observations. ICML 2019.

[7] Better-than-demonstrator imitation learning via automatically-ranked demonstrations. CoRL 2020.

[8] Variational Imitation Learning with Diverse-quality Demonstrations. ICML 2020.

[[9]](https://github.com/yunke-wang/WGAIL) Learning to Weight Imperfect Demonstrations. ICML 2021

[[10]](https://github.com/yunke-wang/UID) Unlabeled Imperfect Demonstrations in Adversarial Imitation Learning. AAAI 2023.
