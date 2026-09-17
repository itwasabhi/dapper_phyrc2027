# Third-party components

The code written for this project is released under the [Apache License 2.0](LICENSE).
The components below are **not** covered by that license and keep their own terms.

## Components

| Component | Where it comes from | How it is distributed here |
|---|---|---|
| NVIDIA Isaac Sim 6.0.1 | Official `nvcr.io/nvidia/isaac-sim` container; the base image digest is pinned in `Dockerfile` | Downloaded by `./run.sh build`. NVIDIA's terms apply |
| Stretch4 robot | `assets/custom/robots/stretch_4/stretch_4.usd` (from RCareWorld-2.0) | Included as a single USD file |
| Female manikin | `assets/custom/human/female2_c4-c5.usd` and `textures/c4-c5.jpg` | Included, used with permission from Cornell University |
| Python packages | termcolor, scipy, matplotlib (other libraries come with Isaac Sim) | Installed at pinned versions by `./run.sh build`; not committed |

## DexGarmentLab

Part of the simulation code in `src/PhyRC_Sim` (the base environment, manikin, and garment
helpers) is adapted from [DexGarmentLab](https://github.com/wayrise/DexGarmentLab)
and ported to Isaac Sim 6.0.1 surface FEM.

DexGarmentLab assets are not committed to this repository; `./run.sh prepare` downloads them:

- Floor texture: one JPEG from `Scene.zip` in the
  [DexGarmentLab dataset on Hugging Face](https://huggingface.co/datasets/wayrise/DexGarmentLab),
  released under Apache-2.0.
- Initial cloth material: `linen_Pumpkin.usd` and its texture JPEGs from the
  DexGarmentLab GitHub repository.

If you use this environment, please also cite DexGarmentLab:

```bibtex
@article{wang2025dexgarmentlab,
  title={DexGarmentLab: Dexterous Garment Manipulation Environment with Generalizable Policy},
  author={Wang, Yuran and Wu, Ruihai and Chen, Yue and Wang, Jiarui and Liang, Jiaqi and Zhu, Ziyu and Geng, Haoran and Malik, Jitendra and Abbeel, Pieter and Dong, Hao},
  journal={arXiv preprint arXiv:2505.11032},
  year={2025}
}
```

Every downloaded file has its exact URL and SHA-256 hash recorded in
[`config/assets.lock.json`](config/assets.lock.json). `prepare` refuses files
whose hash does not match.

If you redistribute the assets under `assets/custom/` or any downloaded files,
check the terms of each component first. Questions: phyrc.challenge@gmail.com
