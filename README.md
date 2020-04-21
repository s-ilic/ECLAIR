# ECLAIR: Ensemble of Codes for Likelihood Analysis, Inference, and Reporting

**Author: Stéphane Ilić**

with feedback and suggestions from: Michael Kopp, Daniel B. Thomas, Constantinos Skordis, Tom G. Złośnik, Nadia Bolis

## Purpose of the Code

The ECLAIR suite of codes is meant to be used as a general inference tool, allowing to sample the posterior distribution of a set of parameters corresponding to a particular physical model, under the constraint of a number of datasets/likelihoods. As such, it interfaces together.... It also contains a set of useful
Though primarly aimed at coslologist.... it can be use more generally...


## Prerequisites and installation

### Main code

The ECLAIR suite is written in Python and thus requires a working Python 2 or 3 installation. It also requires a small number of additional Python modules, namely `numpy` (for general-purpose array manipulation), `matplotlib` (for the plotting scripts), and `emcee` (for sampling). The latter also requires the `h5py` module if one wants to use the HDF5 binary data format for the MCMC outputs. All those packages can be installed with a simple `pip` command:
```
pip install numpy matplotlib emcee h5py
```

The installation of the suite itself simply requires cloning the present git repository:
```
git clone https://github.com/s-ilic/ECLAIR.git
```

### Planck likelihoods

Using the bundled Planck 2015 and 2018 CMB likelihoods requires first the installation of the latest official [Planck likelihood code](http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Code-v3.0_R3.01.tar.gz). After a successful installation, the associated Python wrapper should be callable by your Python installation via the command ``import clik``.

Secondly, it also requires downloading the [baseline 2015](http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Data-baseline_R2.00.tar.gz) and the [baseline 2018](http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Data-baseline_R3.00.tar.gz) CMB data release, all available at the [Planck legacy archive](http://pla.esac.esa.int/pla/#cosmology). You also need to create two environment variables (adding them e.g. to your `.bashrc` file) named `PLANCK_2015_DATA` and `PLANCK_2018_DATA`, pointing respectively to the 2015 and 2018 release data folders (which contain the `low_l`, `hi_l`, etc, subfolders).

## Usage

### MCMC

To start a Monte Carlo Markov chain, ECLAIR only requires the user to prepare an `.ini` file with all the required settings, and then use the command (while in the ECLAIR folder):
```
python ECLAIR_mcmc.py input_file.ini
```

### Maximizer

### Plotting results

## Developing the code

Participation to the further development of the code is welcome. Feel free to clone the current repository, develop your own branch, and get it merged to the public distribution. You may as well [open issues](https://github.com/s-ilic/ECLAIR/issues) to discuss what new features you would like to see implemented.

## License

The ECLAIR project is licensed under the MIT License. See the [LICENSE file](https://github.com/s-ilic/ECLAIR/blob/master/LICENSE) for details.

## Code of conduct

While ECLAIR is free to use, we request that publications using the code should cite at least the paper "Dark Matter properties through cosmic history", arXiV:2004.XXXX.

## Support

To get support, please [open an issue](https://github.com/s-ilic/ECLAIR/issues) in the present repository.
