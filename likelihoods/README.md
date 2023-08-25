# List of available likelihoods


## Table of contents

- [1) Background measurements](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#background-measurements)
  - [1-a) BAO data](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#bao-data)
  - [1-b) H0 data](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#h0-data)
  - [1-c) SN data](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#sn-data)
- [2) Cosmic microwave background measurements](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#cosmic-microwave-background-measurements)
  - [2-a) Planck](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#planck)
  - [2-b) ACT](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#ACT)
  - [2-c) BICEP/Keck](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#bicepkeck)
  - [2-d) SPT](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#spt)

-------------

### 1) Background measurements

#### 1-a) BAO data

##### "BG.BAO.PlanckExtPR2":

BAO data used in Planck 2015 papers as "external data", contains
- 6DF from Beutler et al. ([arXiv:1106.3366](https://arxiv.org/abs/1106.3366))
- BOSS LOWZ & CMASS DR10&11 from Anderson et al. ([arXiv:1312.4877](https://arxiv.org/abs/1312.4877))
- SDSS DR7 MGS from Ross et al. ([arXiv:1409.3242](https://arxiv.org/abs/1409.3242))

##### "BG.BAO.PlanckExtPR3":

BAO data used in Planck 2018 papers as "external data", contains
- 6DF from Beutler et al. ([arXiv:1106.3366](https://arxiv.org/abs/1106.3366))
- SDSS DR7 MGS Ross et al. ([arXiv:1409.3242](https://arxiv.org/abs/1409.3242))
- SDSS DR12 Consensus BAO data from Alam et al. ([arXiv:1607.03155](https://arxiv.org/abs/1607.03155))

#### 1-b) H0 data

##### "BG.H0.R11":
H0 measurements from Riess et al. ([arXiv:1103.2976](https://arxiv.org/abs/1103.2976))

##### "BG.H0.R18":
H0 measurements from Riess et al. ([arXiv:1801.01120](https://arxiv.org/abs/1801.01120))

##### "BG.H0.R19":
H0 measurements from Riess et al. ([arXiv:1903.07603](https://arxiv.org/abs/1903.07603))

##### "BG.H0.F20":
H0 measurements from Freedman et al. ([arXiv:2002.01550](https://arxiv.org/abs/2002.01550))

#### 1-c) SN data

##### "BG.SN.JLA":
Joint Light-curve Analysis Supernovae Sample from Betoule et al. ([arXiv:1401.4064](https://arxiv.org/abs/1401.4064)), unbinned version

##### "BG.SN.Pantheon":
Combined Pantheon Supernovae Sample from Scolnic et al. ([arXiv:1710.00845](https://arxiv.org/abs/1710.00845))

-------------

### 2) Cosmic microwave background measurements

#### 2-a) Planck

##### "PR2":

Planck Public Data Release 2 ([arXiv:1507.02704](https://arxiv.org/abs/1507.02704))
- **"CMB.Planck.PR2.lowTT"**: low-ell temperature likelihood
- **"CMB.Planck.PR2.lowTEB"**: low-ell temperature and E&B polarisation likelihood
- **"CMB.Planck.PR2.highTT"**: full high-ell temperature likelihood
- **"CMB.Planck.PR2.highTTlite"**: high-ell temperature likelihood marginalised over nuisance parameters
- **"CMB.Planck.PR2.highTTTEEE"**: full high-ell temperature and E polarisation likelihood
- **"CMB.Planck.PR2.highTTTEEElite"**: high-ell temperature and E polarisation marginalised over nuisance parameters
- **"CMB.Planck.PR2.lensT"**: lensing likelihood, using T map-based lensing reconstruction
- **"CMB.Planck.PR2.lensTP"**: lensing likelihood, using T and P map-based lensing reconstruction

##### "PR3":

Planck Public Data Release 3 ([arXiv:1907.12875](https://arxiv.org/abs/1907.12875))
- **"CMB.Planck.PR3.lowTT"**: low-ell temperature likelihood
- **"CMB.Planck.PR3.lowEE"**: low-ell E polarisation likelihood
- **"CMB.Planck.PR3.lowBB"**: low-ell B polarisation likelihood
- **"CMB.Planck.PR3.lowEB"**: low-ell E&B polarisation likelihood
- **"CMB.Planck.PR3.highTT"**: full high-ell temperature likelihood
- **"CMB.Planck.PR3.highTTlite"**: high-ell temperature likelihood marginalised over nuisance parameters
- **"CMB.Planck.PR3.highTTTEEE"**: full high-ell temperature and E polarisation likelihood
- **"CMB.Planck.PR3.highTTTEEElite"**: high-ell temperature and E polarisation marginalised over nuisance parameters
- **"CMB.Planck.PR3.lensCMBdep"**: lensing likelihood, using T and P map-based lensing reconstruction, with model-dependent correction
- **"CMB.Planck.PR3.lensCMBmarg"**: lensing likelihood, using T and P map-based lensing reconstruction, marginalized over CMB power spectrum

##### "PR4":

Planck Public Data Release 4 ([lollipop](https://github.com/planck-npipe/lollipop) and [hillipop](https://github.com/planck-npipe/hillipop))
- **"CMB.Planck.PR4.lollipop_lowlE"**: low-ell E polarisation likelihood
- **"CMB.Planck.PR4.lollipop_lowlB"**: low-ell B polarisation likelihood
- **"CMB.Planck.PR4.lollipop_lowlEB"**: low-ell temperature and E&B polarisation likelihood
- **"CMB.Planck.PR4.hillipop_TT"**: full high-ell temperature likelihood
- **"CMB.Planck.PR4.hillipop_TE"**: full high-ell temperature/E polarisation correlation likelihood
- **"CMB.Planck.PR4.hillipop_EE"**: full high-ell E polarisation likelihood
- **"CMB.Planck.PR4.hillipop_TTTEEE"**: full high-ell temperature and E polarisation likelihood

#### 2-b) ACT

##### "ACTPol_DR4":

ACTPol Data release 4 ([arXiv:2007.07288](https://arxiv.org/abs/2007.07288) and [arXiv:2007.07289](https://arxiv.org/abs/2007.07289))
- **"CMB.ACT.ACTPol_DR4.lite_onlyTT"**: full ell range temperature likelihood marginalised over nuisance parameters
- **"CMB.ACT.ACTPol_DR4.lite_onlyTE"**: full ell range temperature/E polarisation correlation likelihood marginalised over nuisance parameters
- **"CMB.ACT.ACTPol_DR4.lite_onlyEE"**: full ell range E polarisation likelihood marginalised over nuisance parameters
- **"CMB.ACT.ACTPol_DR4.lite_all"**: full ell range temperature and E polarisation likelihood likelihood marginalised over nuisance parameters
- **"CMB.ACT.ACTPol_DR4.lite_for_Planck"**: temperature and E polarisation likelihood likelihood marginalised over nuisance parameters, over the restricted range of multipoles suitable for combination with the Planck CMB data

#### 2-c) BICEP/Keck

##### "CMB.BK.BK15" :
Bicep/KECK 2018 likelihood corresponding to data release named BK15 ([arXiv:1810.05216](https://arxiv.org/abs/1810.05216))

##### "CMB.BK.BK18" :
Bicep/KECK 2021 likelihood corresponding to data release named BK18 ([arXiv:2110.00483](https://arxiv.org/abs/2110.00483))

#### 2-d) SPT

##### "CMB.SPT.SPT3G_2020":
SPT 3G likelihood from Dutcher et al. ([arXiv:2101.01684](https://arxiv.org/abs/2101.01684))