from numpy import linalg as LA
import numpy as np


# rotating electronic representation
def AtoD(cP, nR, nState, Up):
  N = len(cP)
  cD = np.zeros((N),dtype= np.complex64)
  for ri in range(nR):
    U = Up[ri,:,:]
    for i in range(nState):
      for j in range(nState):
        k = i * nR + ri
        l = j * nR + ri
        cD[k] += U[i, j] * cP[l]
  return cD

# rotating electronic representation
def DtoA(cD, nR, nState, Up):
  N = len(cD)
  cP = np.zeros((N),dtype= np.complex64)
  for ri in range(nR):
    U = Up[ri,:,:]
    for i in range(nState):
      for j in range(nState):
        k = i * nR + ri
        l = j * nR + ri
        cP[k] += U[j, i] * cD[l]
  return cP

# computing populations
def population(Ci, nState) :
  p = np.zeros(nState, dtype=np.float32) 
  nR = int(len(Ci) / nState)
  for i in range(nState):
    p[i] = Ci[i * nR: (i + 1) * nR].conjugate().dot( Ci[ i * nR: (i + 1) * nR]).real 
  return p

#----------------------------------------
# MATRIX DIAGONALIZATION

def Diag(H):
    E,V = LA.eigh(H) # E corresponds to the eigenvalues and V corresponds to the eigenvectors
    return E,V
#----------------------------------------


# sparse matrix size 	
def size(m):
	return m.data.nbytes + m.indptr.nbytes + m.indices.nbytes