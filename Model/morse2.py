import numpy as np
from tools import Diag 
from numpy import kron as ꕕ


class parameters():
    Rmin  = 1
    Rmax  = 12
    nR    = 512
    dt    = 1.0
    steps = 3000
    aniskip = 250
    M = 20000
#--------------------------------------------------------
#-------------------   Model        ---------------------
#--------------------------------------------------------

def Hel(R):
  """Hel for Morse 2

  Args:
      R (float): nuclear position

  Returns:
      N x N matrix: Matrix elements of electronic part of the Hamiltonian
  """
  D =  np.array([0.020,  0.010,  0.003])
  b =  np.array([0.650,  0.400,  0.650])
  Re = np.array([4.500,  4.000,  4.400])
  c =  np.array([0.000,  0.010,  0.020])

  Aij =  np.array([0.005,  0.005])
  Rij =  np.array([3.660,  3.340])
  a   =  np.array([32.00,  32.00])

  Vij = np.zeros((3,3))
  # Diagonal
  Vij[np.diag_indices(3)] = D * ( 1 - np.exp( -b * (R - Re) ) )**2 + c

  # Off-Diagonal
  Vij[0,1] = Aij[0] * np.exp( -a[0] * (R - Rij[0])**2 )
  Vij[0,2] = Aij[1] * np.exp( -a[1] * (R - Rij[1])**2 )

  # Symmetric Vij
  Vij[2,0], Vij[1,0] = Vij[0,2], Vij[0,1]

  return Vij

#--------------------------------------------------------
#--------------------------------------------------------
#-------------------   initial Ψ    ---------------------
#--------------------------------------------------------
def psi(R, nState= 3):
  """Initial wavefunction

  Args:
      R (numpy array): R is a numerical grid over which the nuclear part of
      the wavefuntion is evaluated. 

  Returns:
      Ψ: wavefunction in the nuclear ⊗ electronic wavefunction. 
      I have used a initial state: Ψ(R) = χ(R) ⊗ |i><i| = χ(R) ⊗ φ
      can be easily modified to have a entangled state--> Ψ(R) = ∑ χi(R) ⊗ |i><i|
  """
  
  # Nuclear Part
  M = 20000
  ω = 5*10**(-3.0)
  R0 = 3.300
  χ = np.exp(- 0.5 * M * ω * (R - R0)**2.0 )
  χ =  χ/(np.sum(χ**2)**0.5)
  
  # Electronic Part
  Φ = np.array([1, 0, 0])
  return  ꕕ(Φ, χ)
