# ECLAIR: Ensemble of Codes for Likelihood Analysis, Inference, and Reporting

**Author: Stéphane Ilić**

with feedback and suggestions from: Michael Kopp, Louis Perenon, Daniel B. Thomas, Constantinos Skordis, Tom G. Złośnik, Nadia Bolis

## Purpose of the Code

The ECLAIR suite of codes is a general inference tool, allowing the user to:
- sample the posterior distribution of parameters from a given model, under the constraint of a number of datasets, via ensemble MCMC techniques;
- estimate the point in parameter space corresponding to the best-fit likelihood value for the model, via a robust ensemble-based minimizer;
- produce a profile likelihood for any model parameter, at a series of user-chosen points in parameter space.

The suite also include plotting scripts allowing the user to conveniently diagnose and check the convergence of a chain, as well as produce summary statistics for the model parameters of interest.

## Prerequisites, installation, and usage

Please refer to the [ECLAIR wiki](https://github.com/s-ilic/ECLAIR/wiki) for detailed instructions on how to install ECLAIR and get it running.

## Developing the code

Participation to the further development of the code is welcome. Feel free to clone the current repository, develop your own branch, and get it merged to the public distribution. You may as well [open issues](https://github.com/s-ilic/ECLAIR/issues) to discuss what new features you would like to see implemented.

## License

The ECLAIR project is licensed under the MIT License. See the [LICENSE file](https://github.com/s-ilic/ECLAIR/blob/master/LICENSE) for more details.

## Code of conduct

While ECLAIR is free to use, we request that publications using the code should cite at least the paper ["Dark Matter properties through cosmic history"](https://arxiv.org/abs/2004.09572) (arXiv:2004.09572). The authors of the `emcee` sampler kindly request [this paper](https://arxiv.org/abs/1202.3665) to be cited, while the authors of the `zeus` sampler request [this paper](https://arxiv.org/abs/2002.06212) and [this paper](https://arxiv.org/abs/2105.03468) to be cited. The published references to the various likelihoods bundled with ECLAIR are included in the [likelihood list](https://github.com/s-ilic/ECLAIR/blob/master/likelihoods/likelihoods.md).

## Support

To get support, please [open an issue](https://github.com/s-ilic/ECLAIR/issues) in the present repository.
