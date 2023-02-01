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
    """Hel for Morse 1

    Args:
        R (float): nuclear position

    Returns:
        N x N matrix: Matrix elements of electronic part of the Hamiltonian
    """
    A = np.array 
    Dii = np.diag_indices

    D =  A([0.003,  0.004,  0.003])
    b =  A([0.650,  0.600,  0.650])
    Re = A([5.000,  4.000,  6.000])
    c =  A([0.000,  0.010,  0.006])

    Aij =  A([0.002,  0.002])
    Rij =  A([3.400,  4.800])
    a   =  A([16.00,  16.00])

    Vij = np.zeros((3,3))
    # Diagonal
    Vij[Dii(3)] = D * ( 1 - np.exp( -b * (R - Re) ) )**2 + c

    # Off-Diagonal
    Vij[0,1] = Aij[0] * np.exp( -a[0] * (R - Rij[0])**2 )
    Vij[1,2] = Aij[1] * np.exp( -a[1] * (R - Rij[1])**2 )

    # Symmetric Vij
    Vij[2,1], Vij[1,0] = Vij[1,2], Vij[0,1]

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
  R0 = 2.9 
  χ = np.exp(- 0.5 * M * ω * (R - R0)**2.0 )
  χ =  χ/(np.sum(χ**2)**0.5)
  
  # Electronic Part
  Φ = np.array([1, 0, 0])
  return  ꕕ(Φ, χ)
