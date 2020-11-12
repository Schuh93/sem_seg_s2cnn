# Spherical MNIST example - semantic segmentation

This code is based on the original MNIST example and extends it to do semantic segmentation as described in detail in [1].

## Install

The required Python packages can be found in the `requirements.txt` file in the `container` subfolder. This code assumes that the `s2cnn` package is in the `PYTHONPATH`. To build a [Singularity](https://sylabs.io/singularity) container from this file which is known to work for this code, run
```
python3 run.py build-singularity
```
This was tested with Singularity version 3.7.0 and Python version 3.8.10.

Note that the requirements file contains many packages (e.g. Tensorflow) which are not needed to run the code.

## Run

The code to train a standard/spherical CNN is located in `train_cnn.py` / `train_s2cnn.py`, respectively. To run it inside the Singularity container built with the command above, run
```
python3 run.py --env singularity train-cnn <training-options>
```
or
```
python3 run.py --env singularity train-s2cnn <training-options>
```
for the available options, see `train_cnn.py` and `train_s2cnn.py`.

The datasets are generated/downloaded during first invocation and saved in `datasets`.

The 1-digit models used in [1] can be trained by running

218k CNN:
```
python3 run.py --env singularity train-cnn --learning_rate 0.001 --batch_size 32 --feature_numbers 13 15 22 31 141 --kernel_sizes 5 3 9 7 3 --strides 1 1 1 1 2 --use_skips --train_data rot --test_data rot --epochs 100 --bandwidth 50 --len_train_data 60000 --len_test_data 10000
```

1M CNN:
```
python3 run.py --env singularity train-cnn --learning_rate 0.001 --batch_size 32 --feature_numbers 12 13 16 77 96 163 --kernel_sizes 3 3 5 5 3 5 --strides 1 1 1 2 1 1 --use_skips --train_data rot --test_data rot --epochs 100 --bandwidth 50 --len_train_data 60000 --len_test_data 10000
```

5.5M CNN:
```
python3 run.py --env singularity train-cnn --learning_rate 0.001 --batch_size 32 --feature_numbers 12 15 16 85 141 191 1100 --kernel_sizes 3 5 3 7 5 3 3 --strides 1 1 1 2 1 1 2 --use_skips --train_data rot --test_data rot --epochs 100 --bandwidth 50 --len_train_data 60000 --len_test_data 10000
```

204k S2CNN:
```
python3 run.py --env singularity train-s2cnn --learning_rate 0.001 --batch_size 32 --feature_numbers 11 12 13 14 13 12 11 --bandlimit_numbers 42 35 27 20 27 35 42 --kernel_max_beta 0.1238 0.1474 0.1768 0.2292 0.3095 0.2292 0.1768 0.1474 --epochs 100 --bandwidth 50 --train_data non_rot --test_data rot --len_train_data 60000 --len_test_data 10000
```

Although these reproduce the exact models used in [1] for the 1-digit case, note that the data generation used in [1] additionally contains routines for patching several digits together on a canvas before projection. Furthermore, the code used for the experiments in [1] uses a more elaborate training procedure which e.g. includes early stopping. The results produced by these commands are therefore likely to not match exactly the ones reported in [1].

## References
[1] Equivariance versus Augmentation for Spherical Images by Jan E. Gerken, Oscar Carlsson, Hampus Linander, Fredrik Ohlsson, Christoffer Petersson and Daniel Persson, [arXiv:2202.03990](https://arxiv.org/abs/2202.03990)
