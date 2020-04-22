# ECLAIR: Ensemble of Codes for Likelihood Analysis, Inference, and Reporting

**Author: Stéphane Ilić**

with feedback and suggestions from: Michael Kopp, Daniel B. Thomas, Constantinos Skordis, Tom G. Złośnik, Nadia Bolis

## Purpose of the Code

The ECLAIR suite of codes is meant to be used as a general inference tool, allowing to sample the posterior distribution of a set of parameters corresponding to a particular physical model, under the constraint of a number of datasets/likelihoods. As such, it brings together state-of-the-art datasets (contained in the `likelihoods` directory), an efficient affine-invariant ensemble sampling algorithm (via its `emcee` Python implementation), and interfaces seamlessly to the powerful [CLASS](https://github.com/lesgourg/class_public/) Boltzmann solver or any custom modification of it. In its current iteration, ECLAIR is thus primarily aimed at cosmologists wanting to test any potential cosmological model again current data, although a generalization of the code to any type of problem is completely feasible and planned for the near future.

The ECLAIR suite also contains a robust maximizer aimed at finding the point in parameter space corresponding to the best likelihood of any considered model, using a novel technique which combines affine-invariant ensemble sampling with [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) to converge reliably towards the global maximum of the posterior.

The suite also include a plotting script allowing to conveniently diagnose and check the convergence of a chain, as well as produce summary statistics on the parameters of interest.

## Prerequisites and installation

### Main code

The ECLAIR suite is written in Python and thus requires a working Python 2 or 3 installation. It also requires a small number of additional Python modules, namely `numpy` (for general-purpose array manipulation), `matplotlib` (for the plotting scripts), and `emcee` (for sampling). The latter also requires the `h5py` module if one wants to use the HDF5 binary data format for the MCMC outputs, but plain text outputs are also available in ECLAIR. All those packages can be installed with a simple `pip` command:
```
pip install numpy matplotlib emcee h5py
```

The installation of the suite itself simply requires cloning the present git repository:
```
git clone https://github.com/s-ilic/ECLAIR.git
```
The resulting folder contains:
  * a `likelihoods` folder containing a number of sub-folders, each corresponding to a single dataset/likelihood in the form of a Python module;
  * an `inputs` folder containing a detailed `template.ini` configuration file and short example (see Usage section);
  * `ECLAIR_mcmc.py`, the main script for running Monte Carlo Markov chains;
  * `ECLAIR_maximize.py`, a script for running a search for a global likelihood maximum;
  * `ECLAIR_plots.py`, a script for analyzing and plotting the content of the ECLAIR output files;
  * `ECLAIR_parser.py`, a parser module for `.ini` files, used by the three aforementioned scripts.

### CLASS

Currently, the Boltzman code CLASS is the only "theoretical engine" interfaced to ECLAIR. Its installation is thus required to run the suite, and especially its associated CLASS Python wrapper `classy`. Please refer to the CLASS code [webpage](http://class-code.net) and [GitHub repository](https://github.com/lesgourg/class_public) for detailed installation instructions. Note however that one is not limited to the "vanilla" version of CLASS: ECLAIR is compatible with any variant or modification based on CLASS, as long as the name of the corresponding Python wrapper is correctly passed to the `which_class` option in the ECLAIR `.ini` file.

### Planck likelihoods

Using the bundled Planck 2015 and 2018 CMB likelihoods requires first the installation of the latest official Planck likelihood code ([direct download link](http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Code-v3.0_R3.01.tar.gz)). After a successful installation, the associated Python wrapper should be callable by your Python installation via the command ``import clik``.

Secondly, it also requires downloading the baseline 2015 ([direct link](http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Data-baseline_R2.00.tar.gz)) and 2018 ([direct link](http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Data-baseline_R3.00.tar.gz)) CMB data release, all available at the [Planck legacy archive](http://pla.esac.esa.int/pla/#cosmology). You also need to create two environment variables (adding them e.g. to your `.bashrc` file) named `PLANCK_2015_DATA` and `PLANCK_2018_DATA`, pointing respectively to the 2015 and 2018 release data folders (which contain the `low_l`, `hi_l`, etc, subfolders).

## Usage

### MCMC

To start a Monte Carlo Markov chain, ECLAIR only requires the user to prepare an `.ini` file with all the required settings, and then use the command (while in the ECLAIR folder):
```
python ECLAIR_mcmc.py /path/to/input_file.ini
```
The `inputs` folder contains an extensively commented `.ini` file, `template.ini`, detailing all the capabilities and possible settings of ECLAIR. Users are advised to make a copy of this file before starting to modify it. A short example `.ini` file stripped of comments, `example_short.ini`, is also provided for users wanting to quickly try out the code. One can run it with the commands (while in the ECLAIR folder):
```
mkdir outputs
python ECLAIR_mcmc.py inputs/example_short.ini
```
which will run 10 steps of a 10-walker chain on the Hubble parameter today (`H0`) and the physical density of cold dark matter (`omega_cdm`), using BAO data as constraints.

### Maximizer

Description to be added soon.

### Plotting results

Description to be added soon.

## Developing the code

Participation to the further development of the code is welcome. Feel free to clone the current repository, develop your own branch, and get it merged to the public distribution. You may as well [open issues](https://github.com/s-ilic/ECLAIR/issues) to discuss what new features you would like to see implemented.

## License

The ECLAIR project is licensed under the MIT License. See the [LICENSE file](https://github.com/s-ilic/ECLAIR/blob/master/LICENSE) for more details.

## Code of conduct

While ECLAIR is free to use, we request that publications using the code should cite at least the paper ["Dark Matter properties through cosmic history"](https://arxiv.org/abs/2004.09572) (arXiv:2004.09572).

## Support

To get support, please [open an issue](https://github.com/s-ilic/ECLAIR/issues) in the present repository.
