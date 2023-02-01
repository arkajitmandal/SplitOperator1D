import numpy as np
from tools import Diag 
from numpy import kron as ꕕ
#-------------------------------------
def adiabatic(R, Hel):
  Hpl = Vii(R, Hel) 
  nR  = len(R)
  nState = Hpl.shape[1]
  vectors = np.zeros( (nR, nState, nState), dtype=np.complex64) 
  Ep = np.zeros( (nR, nState), dtype=np.complex64) 
  # Interpolation of data
  for ri in range(nR):
    E,V = Diag(Hpl[ ri, :, :] )  
    #--- Phase Fix -------------
    if ri>0:
      for ei in range(nState):
        sign = np.zeros((nState))
        for ej in range(nState):
          sign[ej] = np.dot(Vold[:,ej],V[:,ei])
        signId = np.argmax(np.abs(sign))
        sign[signId] = sign[signId]/abs(sign[signId])
        V[:,ei] = V[:,ei] * sign[signId]
    #---------------------------
    Vold = V 
    vectors[ri,:,:] = V
    Ep[ ri, :] = E  
  Ep = Ep.T.reshape((nState * nR))
  return Ep, vectors
    
#----------------------------------------
# Data of the diabatic states

def Vii(R, Hel):
  """
  This code only works for 1D 
  Args:
      R (numpy array): R is a numerical grid over which the potential term is evaluated. 
      Hel (function): Hel(Ri) gives a N x N matrix which are Hel's matrix elements at nuclear position R
  """
  nState = len(Hel(R[0]))
  Vij = np.zeros((len(R),nState,nState))

  for Ri in range(len(R)):
     Vij[Ri, :, :] =  Hel(R[Ri])
  return Vij 
#--------------------------------------------------------

