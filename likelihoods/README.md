# List of available likelihoods


## Table of contents

- [1) Background measurements](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#1-background-measurements)
  - [1-a) Baryon Acoustic Oscillations (BAO) data](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#1-a-bao-data)
  - [1-b) H0 data](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#1-b-h0-data)
  - [1-c) Supernovae (SN) data](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#1-c-sn-data)
- [2) Cosmic microwave background measurements](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#2-cosmic-microwave-background-measurements)
  - [2-a) Planck](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#2-a-planck)
  - [2-b) ACT](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#2-b-act)
  - [2-c) BICEP/Keck](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#2-c-bicepkeck)
  - [2-d) SPT](https://github.com/s-ilic/ECLAIR/tree/master/likelihoods#2-d-spt)

-------------

### 1) Background measurements

#### 1-a) Baryon Acoustic Oscillations (BAO) data

##### "BG.BAO.PlanckExtPR2":

BAO data compilation used in Planck 2015 papers as "external data", contains
- 6DF from Beutler et al. ([arXiv:1106.3366](https://arxiv.org/abs/1106.3366))
- BOSS LOWZ & CMASS DR10&11 from Anderson et al. ([arXiv:1312.4877](https://arxiv.org/abs/1312.4877))
- SDSS DR7 MGS from Ross et al. ([arXiv:1409.3242](https://arxiv.org/abs/1409.3242))

##### "BG.BAO.PlanckExtPR3":

BAO data compilation used in Planck 2018 papers as "external data", contains
- 6DF from Beutler et al. ([arXiv:1106.3366](https://arxiv.org/abs/1106.3366))
- SDSS DR7 MGS Ross et al. ([arXiv:1409.3242](https://arxiv.org/abs/1409.3242))
- SDSS DR12 Consensus BAO data from Alam et al. ([arXiv:1607.03155](https://arxiv.org/abs/1607.03155))

##### "BG.BAO.SIXdF":

BAO data from the 6dF survey (Beutler et al., [arXiv:1106.3366](https://arxiv.org/abs/1106.3366))

##### "BG.BAO.SDSS":

BAO data from various data releases of SDSS (explicit from the likelihood name)
- **"BG.BAO.SDSS.DR7.MGS"**: Ross et al. ([arXiv:1409.3242](https://arxiv.org/abs/1409.3242))
- **"BG.BAO.SDSS.DR12.LRG"**: Alam et al. ([arXiv:1607.03155](https://arxiv.org/abs/1607.03155))
- **"BG.BAO.SDSS.DR16.ELG"**: Raichoor et al. ([arXiv:2007.09007](https://arxiv.org/abs/2007.09007))
- **"BG.BAO.SDSS.DR16.LRG"**: Gil-Marín et al. ([arXiv:2007.08994](https://arxiv.org/abs/2007.08994))
- **"BG.BAO.SDSS.DR16.LYAUTO"**: du Mas des Bourboux et al. ([arXiv:2007.08995](https://arxiv.org/abs/2007.08995))
- **"BG.BAO.SDSS.DR16.LYxQSO"**: du Mas des Bourboux et al. ([arXiv:2007.08995](https://arxiv.org/abs/2007.08995))
- **"BG.BAO.SDSS.DR16.QSO"**: Hou et al. ([arXiv:2007.08998](https://arxiv.org/abs/2007.08998))

##### "BG.BAO.DESI.DR1":

BAO data from the first data release of the DESI survey, from DESI Collaboration: Adame et al. ([arXiv:2404.03002](https://arxiv.org/abs/2404.03002)) and references therein
- **"BG.BAO.DESI.DR1.BGS_Z1"**: BGS, 0.1 < z < 0.4
- **"BG.BAO.DESI.DR1.ELG_Z2"**: ELG, 1.1 < z < 1.6
- **"BG.BAO.DESI.DR1.LRGplusELG_Z1"**: LRG+ELG, 0.8 < z < 1.1
- **"BG.BAO.DESI.DR1.LRG_Z1"**: LRG, 0.4 < z < 0.6
- **"BG.BAO.DESI.DR1.LRG_Z2"**: LRG, 0.6 < z < 0.8
- **"BG.BAO.DESI.DR1.LYA"**: Ly-alpha z ~ 2.33
- **"BG.BAO.DESI.DR1.QSO_Z1"**: QSO, 0.8 < z < 2.1

##### "BG.BAO.DESI.DR2":

BAO data from the second data release of the DESI survey, from DESI Collaboration: Abdul-Karim et al. ([arXiv:2503.14738](https://arxiv.org/abs/2503.14738)) and references therein
- **"BG.BAO.DESI.DR2.BGS_Z1"**: BGS, 0.1 < z < 0.4
- **"BG.BAO.DESI.DR2.ELG_Z2"**: ELG, 1.1 < z < 1.6
- **"BG.BAO.DESI.DR2.LRGplusELG_Z1"**: LRG+ELG, 0.8 < z < 1.1
- **"BG.BAO.DESI.DR2.LRG_Z1"**: LRG, 0.4 < z < 0.6
- **"BG.BAO.DESI.DR2.LRG_Z2"**: LRG, 0.6 < z < 0.8
- **"BG.BAO.DESI.DR2.LYA"**: Ly-alpha z ~ 2.33
- **"BG.BAO.DESI.DR2.QSO_Z1"**: QSO, 0.8 < z < 2.1

#### 1-b) H0 data

##### "BG.H0.R11":
H0 measurements from Riess et al. ([arXiv:1103.2976](https://arxiv.org/abs/1103.2976))

##### "BG.H0.R18":
H0 measurements from Riess et al. ([arXiv:1801.01120](https://arxiv.org/abs/1801.01120))

##### "BG.H0.R19":
H0 measurements from Riess et al. ([arXiv:1903.07603](https://arxiv.org/abs/1903.07603))

##### "BG.H0.F20":
H0 measurements from Freedman et al. ([arXiv:2002.01550](https://arxiv.org/abs/2002.01550))

#### 1-c) Supernovae (SN) data

##### "BG.SN.JLA":
Joint Light-curve Analysis Supernovae Sample from Betoule et al. ([arXiv:1401.4064](https://arxiv.org/abs/1401.4064)), unbinned version

##### "BG.SN.Pantheon":
Combined Pantheon Supernovae Sample from Scolnic et al. ([arXiv:1710.00845](https://arxiv.org/abs/1710.00845))

##### "BG.SN.DESY5":
Dark Energy Survey Year 5 (DES-Y5) type Ia supernovae sample from DES Collaboration ([arXiv:2401.02929](https://arxiv.org/abs/2401.02929))

##### "BG.SN.Union3":
Union3 & UNITY1.5 type Ia supernovae sample from Rubin et al. ([arXiv:2311.12098](https://arxiv.org/abs/2311.12098))

##### "BG.SN.PantheonPlus":
Pantheon+ (without SH0ES) type Ia supernovae sample from Brout et al. ([arXiv:
2202.04077](https://arxiv.org/abs/2202.04077))

-------------

### 2) Cosmic microwave background measurements

#### 2-a) Planck

##### "PR2":

Planck Public Data Release 2 ([arXiv:1507.02704](https://arxiv.org/abs/1507.02704))
- **"CMB.Planck.PR2.lowTT"**: low-ell temperature likelihood
- **"CMB.Planck.PR2.lowTEB"**: low-ell temperature, E-mode and B-mode polarisation likelihood
- **"CMB.Planck.PR2.highTT"**: full high-ell temperature likelihood
- **"CMB.Planck.PR2.highTTlite"**: high-ell temperature likelihood marginalised over nuisance parameters
- **"CMB.Planck.PR2.highTTTEEE"**: full high-ell temperature and E-mode polarisation likelihood
- **"CMB.Planck.PR2.highTTTEEElite"**: high-ell temperature and E-mode polarisation marginalised over nuisance parameters
- **"CMB.Planck.PR2.lensT"**: lensing likelihood, using T map-based lensing reconstruction
- **"CMB.Planck.PR2.lensTP"**: lensing likelihood, using T and P map-based lensing reconstruction

##### "PR3":

Planck Public Data Release 3 ([arXiv:1907.12875](https://arxiv.org/abs/1907.12875))
- **"CMB.Planck.PR3.lowTT"**: low-ell temperature likelihood
- **"CMB.Planck.PR3.lowEE"**: low-ell E-mode polarisation likelihood
- **"CMB.Planck.PR3.lowBB"**: low-ell B-mode polarisation likelihood
- **"CMB.Planck.PR3.lowEB"**: low-ell E-mode and B-mode polarisation likelihood
- **"CMB.Planck.PR3.highTT"**: full high-ell temperature likelihood
- **"CMB.Planck.PR3.highTTlite"**: high-ell temperature likelihood marginalised over nuisance parameters
- **"CMB.Planck.PR3.highTTTEEE"**: full high-ell temperature and E-mode polarisation likelihood
- **"CMB.Planck.PR3.highTTTEEElite"**: high-ell temperature and E-mode polarisation marginalised over nuisance parameters
- **"CMB.Planck.PR3.lensCMBdep"**: lensing likelihood, using T and P map-based lensing reconstruction, with model-dependent correction
- **"CMB.Planck.PR3.lensCMBmarg"**: lensing likelihood, using T and P map-based lensing reconstruction, marginalized over CMB power spectrum

##### "PR4":

Likelihoods derived from Planck Public Data Release 4

From [lollipop](https://github.com/planck-npipe/lollipop):
- **"CMB.Planck.PR4.lollipop_lowlE"**: low-ell E-mode polarisation likelihood
- **"CMB.Planck.PR4.lollipop_lowlB"**: low-ell B-mode polarisation likelihood
- **"CMB.Planck.PR4.lollipop_lowlEB"**: low-ell temperature and E-mode and B-mode polarisation likelihood

From [hillipop](https://github.com/planck-npipe/hillipop):
- **"CMB.Planck.PR4.hillipop_TT"**: full high-ell temperature likelihood
- **"CMB.Planck.PR4.hillipop_TE"**: full high-ell temperature/E-mode polarisation correlation likelihood
- **"CMB.Planck.PR4.hillipop_EE"**: full high-ell E-mode polarisation likelihood
- **"CMB.Planck.PR4.hillipop_TTTEEE"**: full high-ell temperature and E-mode polarisation likelihood
- **"CMB.Planck.PR4.hillipop_TT_lite"**: binned version of the high-ell temperature likelihood
- **"CMB.Planck.PR4.hillipop_TTTEEE_lite"**: binned version of the high-ell temperature and E-mode polarisation likelihood

From [new lensing likelihood](https://github.com/carronj/planck_PR4_lensing):
- **"CMB.Planck.PR4.lensing"**: lensing likelihood
- **"CMB.Planck.PR4.lensing_marged"**: lensing likelihood marginalized over CMB power spectrum

#### 2-b) ACT

##### "ACTPol_DR4":

ACTPol Data release 4 ([arXiv:2007.07288](https://arxiv.org/abs/2007.07288) and [arXiv:2007.07289](https://arxiv.org/abs/2007.07289))
- **"CMB.ACT.ACTPol_DR4.lite_onlyTT"**: full ell range temperature likelihood marginalised over nuisance parameters
- **"CMB.ACT.ACTPol_DR4.lite_onlyTE"**: full ell range temperature/E-mode polarisation correlation likelihood marginalised over nuisance parameters
- **"CMB.ACT.ACTPol_DR4.lite_onlyEE"**: full ell range E-mode polarisation likelihood marginalised over nuisance parameters
- **"CMB.ACT.ACTPol_DR4.lite_all"**: full ell range temperature and E-mode polarisation likelihood marginalised over nuisance parameters
- **"CMB.ACT.ACTPol_DR4.lite_for_Planck"**: temperature and E-mode polarisation likelihood marginalised over nuisance parameters, over the restricted range of multipoles suitable for combination with the Planck CMB data

##### "DR6_lensing":

ACT DR6 CMB lensing likelihood ([arXiv:2304.05203](https://arxiv.org/abs/2304.05203) and [arXiv:2304.05202](https://arxiv.org/abs/2304.05202)), including variants combining ACT and Planck lensing measurements ([arXiv:2206.07773](https://arxiv.org/abs/2206.07773))
- **"CMB.ACT.DR6_lensing.act_baseline**: for the ACT-only lensing power spectrum with the baseline multipole range
- **"CMB.ACT.DR6_lensing.act_extended**: for the ACT-only lensing power spectrum with the extended multipole range (L<1250)
- **"CMB.ACT.DR6_lensing.actplanck_baseline**: for the ACT+Planck lensing power spectrum with the baseline multipole range
- **"CMB.ACT.DR6_lensing.actplanck_extended**: for the ACT+Planck lensing power spectrum with the extended multipole range (L<1250)

All of the above likelihoods are meant to be combined with primary CMB measurements; lensing-only versions are also available with the same name plus the suffix **"_lens_only"**.

##### "DR6_lite":

ACT DR6 primary CMB temperature and E-mode polarisation likelihood ([arXiv:2503.14452](https://arxiv.org/abs/2503.14452)) in its "lite" version, i.e. foreground-marginalized

#### 2-c) BICEP/Keck

##### "CMB.BK.BK15" :
Bicep/KECK 2018 likelihood corresponding to data release named BK15 ([arXiv:1810.05216](https://arxiv.org/abs/1810.05216))

##### "CMB.BK.BK18" :
Bicep/KECK 2021 likelihood corresponding to data release named BK18 ([arXiv:2110.00483](https://arxiv.org/abs/2110.00483))

#### 2-d) SPT

##### "CMB.SPT.SPT3G_2020":
SPT 3G likelihood from Dutcher et al. ([arXiv:2101.01684](https://arxiv.org/abs/2101.01684))