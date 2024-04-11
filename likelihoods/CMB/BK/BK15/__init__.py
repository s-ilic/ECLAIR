import numpy as np
# import pandas as pd
import scipy.linalg as la
import os, sys
from functools import reduce
from .BK15_vars import *

#################
# Initialistion #
#################
# Some variables
path_to_data = os.path.dirname(os.path.realpath(sys.argv[0]))
path_to_data += '/likelihoods/CMB/BK/BK15/data/'
T_CMB = 2.72548        # CMB temperature
h = 6.62606957e-34     # Planck's constant
kB = 1.3806488e-23     # Boltzmann constant
Ghz_Kelvin = h/kB*1e9  # GHz Kelvin conversion
map_names_used = [
    'BK15_95_B',
    'BK15_150_B',
    'BK15_220_B',
    'W023_B',
    'P030_B',
    'W033_B',
    'P044_B',
    'P070_B',
    'P100_B',
    'P143_B',
    'P217_B',
    'P353_B',
]
map_fields_used = [map_fields[map_names.index(m)] for m in map_names_used]
nmaps = len(map_names_used)
ncrossmaps = nmaps * (nmaps + 1) // 2
flat_to_diag = np.tril_indices(nmaps)
diag_to_flat = np.zeros((nmaps,nmaps),dtype='int')
diag_to_flat[flat_to_diag] = list(range(ncrossmaps))

# Read bandpasses
bandpasses = {}
for key in map_names_used:
    bandpasses[key] = {
        'field': map_fields[map_names.index(key)],
        'filename': 'bandpass_%s.txt' % key[:-2],
    }

for key, valdict in bandpasses.items():
    tmp = np.loadtxt(path_to_data + valdict['filename'])
    #Frequency nu, response resp:
    valdict['nu'] = tmp[:,0]
    valdict['resp'] = tmp[:,1]
    valdict['dnu'] = np.gradient(valdict['nu'])
    # Calculate thermodynamic temperature conversion between this bandpass
    # and pivot frequencies 353 GHz (used for dust) and 23 GHz (used for
    # sync).
    th_int = np.sum(valdict['dnu']*valdict['resp']*valdict['nu']**4*np.exp(Ghz_Kelvin*valdict['nu']/T_CMB)/(np.exp(Ghz_Kelvin*valdict['nu']/T_CMB)-1.)**2)
    nu0=353.
    th0 = nu0**4*np.exp(Ghz_Kelvin*nu0/T_CMB) / (np.exp(Ghz_Kelvin*nu0/T_CMB) - 1.)**2
    valdict['th353'] = th_int / th0
    nu0=23.
    th0 = nu0**4*np.exp(Ghz_Kelvin*nu0/T_CMB) / (np.exp(Ghz_Kelvin*nu0/T_CMB) - 1.)**2
    valdict['th023'] = th_int / th0
    #print 'th353:', valdict['th353'], 'th023:', valdict['th023']

# Useful function
def GetIndicesAndMask(crossmaplist):
    usedmaps = map_names_used
    # nmaps = len(usedmaps)
    mask = np.array([False for i in range(len(crossmaplist))])
    flatindex = []
    for i, crossmap in enumerate(crossmaplist):
        map1, map2 = crossmap.split('x')
        if map1 in usedmaps and map2 in usedmaps:
            index1 = usedmaps.index(map1)
            index2 = usedmaps.index(map2)
            if index1 > index2:
                flatindex.append(index1*(index1+1)//2+index2)
            else:
                flatindex.append(index2*(index2+1)//2+index1)
            mask[i] = True
    return flatindex, mask

# Read window bins
window_data = np.zeros((nbins, cl_lmax, ncrossmaps))
# Retrieve mask and index permutation of windows:
indices, mask = GetIndicesAndMask(bin_window_in_order)
for k in range(nbins):
    windowfile = path_to_data + bin_window_files.replace('???',str(k+1))
    # tmp = pd.read_table(windowfile,comment='#',sep=' ',header=None, index_col=0).to_numpy()
    tmp = np.loadtxt(windowfile)[:cl_lmax, 1:]
    # Apply mask
    tmp = tmp[:,mask]
    # Permute columns and store this bin
    window_data[k][:,indices] = tmp

#Read covmat fiducial
# Retrieve mask and index permutation for a single bin.
indices, mask = GetIndicesAndMask(covmat_cl)
# Extend mask and indices. Mask just need to be copied, indices needs to be increased:
superindices = []
supermask = []
for k in range(nbins):
    superindices += [idx+k*ncrossmaps for idx in indices]
    supermask += list(mask)
supermask = np.array(supermask)

# tmp = pd.read_table(
#     path_to_data + covmat_fiducial,
#     comment='#', sep=' ', header=None, skipinitialspace=True).to_numpy()
tmp = np.loadtxt(path_to_data + covmat_fiducial)
# Apply mask:
tmp = tmp[:,supermask][supermask,:]
# print('Covmat read with shape {}'.format(tmp.shape))
# Store covmat in correct order
covmat = np.zeros((nbins*ncrossmaps,nbins*ncrossmaps))
for index_tmp, index_covmat in enumerate(superindices):
    covmat[index_covmat,superindices] = tmp[index_tmp,:]

#Compute inverse and store
covmat_inverse = la.inv(covmat)

# Useful function
def ReadMatrix(filename, crossmaps):
    usedmaps = map_names_used
    nmaps = len(usedmaps)
    # Get mask and indices
    indices, mask = GetIndicesAndMask(crossmaps)
    # Read matrix in packed format
    # A = pd.read_table(
    #     path_to_data + filename,
    #     comment='#',sep=' ',header=None, index_col=0).to_numpy()
    A = np.loadtxt(path_to_data + filename)[:, 1:]
    # Apply mask
    A = A[:,mask]

    # Create matrix for each bin and unpack A:
    Mlist = []
    # Loop over bins:
    for k in range(nbins):
        M = np.zeros((nmaps,nmaps))
        Mflat = np.zeros((nmaps*(nmaps+1)//2))
        Mflat[indices] = A[k,:]
        M[flat_to_diag] = Mflat
        # Symmetrise M and append to list:
        Mlist.append(M+M.T-np.diag(M.diagonal()))
    return Mlist


# Read noise:
cl_noise_matrix = ReadMatrix(cl_noise_file,cl_noise_order)

# Read Chat and perhaps add noise:
cl_hat_matrix = ReadMatrix(cl_hat_file,cl_hat_order)
if not cl_hat_includes_noise:
    for k in range(nbins):
        cl_hat_matrix[k] += cl_noise_matrix[k]

# Read cl_fiducial and perhaps add noise:
cl_fiducial_sqrt_matrix = ReadMatrix(cl_fiducial_file,cl_fiducial_order)
if not cl_fiducial_includes_noise:
    for k in range(nbins):
        cl_fiducial_sqrt_matrix[k] += cl_noise_matrix[k]
# Now take matrix square root:
for k in range(nbins):
    cl_fiducial_sqrt_matrix[k] = la.sqrtm(cl_fiducial_sqrt_matrix[k])


class likelihood:
  def __init__(self, lkl_input):
    pass

  def get_loglike(self, class_input, likes_input, class_run):
      """
      Compute negative log-likelihood using the Hamimeche-Lewis formalism, see
      http://arxiv.org/abs/arXiv:0801.0554
      """
      # Define the matrix transform
      def MatrixTransform(C, Chat, CfHalf):
          # C is real and symmetric, so we can use eigh()
          D, U = la.eigh(C)
          D = np.abs(D)
          S = np.sqrt(D)
          # Now form B = C^{-1/2} Chat C^{-1/2}. I am using broadcasting to divide rows and columns
          # by the eigenvalues, not sure if it is faster to form the matmul(S.T, S) matrix.
          # B = U S^{-1} V^T Chat U S^{-1} U^T
          B = np.dot(np.dot(U,np.dot(np.dot(U.T,Chat),U)/S[:,None]/S[None,:]),U.T)
          # Now evaluate the matrix function g[B]:
          D, U = la.eigh(B)
          gD = np.sign(D-1.)*np.sqrt(2.*np.maximum(0.,D-np.log(D)-1.))
          # Final transformation. U*gD = U*gD[None,:] done by broadcasting. Collect chain matrix multiplication using reduce.
          M = reduce(np.dot, [CfHalf,U*gD[None,:],U.T,CfHalf.T])
          #M = np.dot(np.dot(np.dot(CfHalf,U*gD[None,:]),U.T),Cfhalf.T)
          return M

      # Recover Cl_s from CLASS
      fl = class_run.lensed_cl()['ell'][1:cl_lmax+1] * (class_run.lensed_cl()['ell'][1:cl_lmax+1] + 1.) / (2. * np.pi)
      DlEE = fl * class_run.lensed_cl()['ee'][1:cl_lmax+1] * 1e12 * class_run.T_cmb()**2.
      DlBB = fl * class_run.lensed_cl()['bb'][1:cl_lmax+1] * 1e12 * class_run.T_cmb()**2.

      #################################
      # BEGIN Update foreground model #
      #################################
      # UpdateForegroundModel(cosmo, data)
      # Function to compute f_dust
      def DustScaling(beta, Tdust, bandpass):
          # Calculates greybody scaling of dust signal defined at 353 GHz to specified bandpass.
          nu0 = 353 #Pivot frequency for dust (353 GHz).
          # Integrate greybody scaling and thermodynamic temperature conversion across experimental bandpass.
          gb_int = np.sum(bandpass['dnu']*bandpass['resp']*bandpass['nu']**(3+beta)/(np.exp(Ghz_Kelvin*bandpass['nu']/Tdust) - 1))
          # Calculate values at pivot frequency.
          gb0 = nu0**(3+beta) / (np.exp(Ghz_Kelvin*nu0/Tdust) - 1)
          # Calculate and return dust scaling fdust.
          return ((gb_int / gb0) / bandpass['th353'])

      # Function to compute f_sync
      def SyncScaling(beta, bandpass):
          #Calculates power-law scaling of synchrotron signal defined at 150 GHz to specified bandpass.
          nu0 = 23.0 # Pivot frequency for sync (23 GHz).
          # Integrate power-law scaling and thermodynamic temperature conversion across experimental bandpass.
          pl_int = np.sum( bandpass['dnu']*bandpass['resp']*bandpass['nu']**(2+beta))
          # Calculate values at pivot frequency.
          pl0 = nu0**(2+beta)
          # Calculate and return dust scaling fsync.
          return ((pl_int / pl0) / bandpass['th023'])

      ellpivot = 80.
      ell = np.arange(1,cl_lmax+1)

      # Convenience variables: store the nuisance parameters in short named variables
      # for parname in self.use_nuisance:
      #     evalstring = parname+" = data.mcmc_parameters['"+parname+"']['current']*data.mcmc_parameters['"+parname+"']['scale']"
      #     print evalstring
      BBdust = likes_input['BBdust']
      BBsync = likes_input['BBsync']
      BBalphadust = likes_input['BBalphadust']
      BBbetadust = likes_input['BBbetadust']
      BBTdust = likes_input['BBTdust']
      BBalphasync = likes_input['BBalphasync']
      BBbetasync = likes_input['BBbetasync']
      BBdustsynccorr = likes_input['BBdustsynccorr']

      # Store current EEtoBB conversion parameters.
      EEtoBB_dust = likes_input['EEtoBB_dust']
      EEtoBB_sync = likes_input['EEtoBB_sync']

      # Compute fdust and fsync for each bandpass
      fdust = {}
      fsync = {}
      for key, bandpass in bandpasses.items():
          fdust[key] = DustScaling(BBbetadust, BBTdust, bandpass)
          fsync[key] = SyncScaling(BBbetasync, bandpass)

      # Computes coefficients such that the foreground model is simply
      # dust*self.dustcoeff+sync*self.synccoeff+dustsync*self.dustsynccoeff
      # These coefficients are independent of the map used,
      # so we save some time by computing them here.
      dustcoeff = BBdust*(ell/ellpivot)**BBalphadust
      synccoeff = BBsync*(ell/ellpivot)**BBalphasync
      dustsynccoeff = BBdustsynccorr*np.sqrt(BBdust*BBsync)*(ell/ellpivot)**(0.5*(BBalphadust+BBalphasync))
      ###############################
      # END Update foreground model #
      ###############################


      # Initialise Cls matrix to zero:
      Cls = np.zeros((nbins,nmaps,nmaps))
      # Initialise the X vector:
      X = np.zeros((nbins*ncrossmaps))
      for i in range(nmaps):
          for j in range(i+1):
              #If EE or BB, add theoretical prediction including foreground:
              if map_fields_used[i]==map_fields_used[j]=='E' or map_fields_used[i]==map_fields_used[j]=='B':
                  map1 = map_names_used[i]
                  map2 = map_names_used[j]
                  dust = fdust[map1]*fdust[map2]
                  sync = fsync[map1]*fsync[map2]
                  dustsync = fdust[map1]*fsync[map2] + fdust[map2]*fsync[map1]
                  # if EE spectrum, multiply foregrounds by the EE/BB ratio:
                  if map_fields_used[i]=='E':
                      dust = dust * EEtoBB_dust
                      sync = sync * EEtoBB_sync
                      dustsync = dustsync * np.sqrt(EEtoBB_dust*EEtoBB_sync)
                      # Deep copy is important here, since we want to reuse DlXX for each map.
                      DlXXwithforegound = np.copy(DlEE)
                  else:
                      DlXXwithforegound = np.copy(DlBB)
                  # Finally add the foreground model:
                  DlXXwithforegound += (dust*dustcoeff+sync*synccoeff+dustsync*dustsynccoeff)
                  # Apply the binning using the window function:
                  for k in range(nbins):
                      Cls[k,i,j] = Cls[k,j,i] = np.dot(DlXXwithforegound,window_data[k,:,diag_to_flat[i,j]])
      # Add noise contribution:
      for k in range(nbins):
          Cls[k,:,:] += cl_noise_matrix[k]
          # Compute entries in X vector using the matrix transform
          T = MatrixTransform(Cls[k,:,:], cl_hat_matrix[k], cl_fiducial_sqrt_matrix[k])
          # Add flat version of T to the X vector
          X[k*ncrossmaps:(k+1)*ncrossmaps] = T[flat_to_diag]
      # Compute chi squared
      chi2 = np.dot(X.T,np.dot(covmat_inverse,X))
      return -0.5*chi2
