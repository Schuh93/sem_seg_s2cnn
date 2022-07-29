# Semantic segmentation with spherical CNNs

This repository contains the semantic segmentation code for spherical CNNs used in *Equivariance versus Augmentation for Spherical Images* [1].

The implementation is based on the original S2CNN code, available [here](https://github.com/jonkhler/s2cnn), from which this repo is forked. This repositiory adds the following to the S2CNN code:

- A new equivariant convolutional layer `SO3ToS2` which takes feature maps on SO(3) as input and outputs feature maps on S².
- Some simple example code in `examples/mnist_sem_seg/` for generating semantic masks for the spherical MNIST digits and training equivariant and non-equivariant networks on this dataset. For details see the `README.md` in that directory.
- Some minor modifications of the original S2CNN code to make it compatible with PyTorch 1.8.1.

More details on the `SO3ToS2` layer, the semantic segmentation dataset and the network architectures implemented here are given in [1].

# Reference

[1] *Equivariance versus Augmentation for Spherical Images*, Jan E. Gerken, Oscar Carlsson, Hampus Linander, Fredrik Ohlsson, Christoffer Petersson and Daniel Persson, Proceedings of the 39th International Conference on Machine Learning; PMLR, 2022; pp 7404–7421 [arXiv:2202.03990](http://arxiv.org/abs/2202.03990)

If you use this code, please cite

```
@inproceedings{gerken2022,
  title = {Equivariance versus {{Augmentation}} for {{Spherical Images}}},
  booktitle = {Proceedings of the 39th {{International Conference}} on {{Machine Learning}}},
  author = {Gerken, Jan E. and Carlsson, Oscar and Linander, Hampus and Ohlsson, Fredrik and Petersson, Christoffer and Persson, Daniel},
  date = {2022-06-28},
  eprint = {2202.03990},
  eprinttype = {arxiv},
  primaryclass = {cs},
  pages = {7404--7421},
  publisher = {{PMLR}},
  doi = {10.48550/arXiv.2202.03990},
  archiveprefix = {arXiv},
  eventtitle = {International {{Conference}} on {{Machine Learning}}},
}
```
