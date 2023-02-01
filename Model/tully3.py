import numpy as np
from tools import Diag 
from numpy import kron as ꕕ


class parameters():
    Rmin  = -13
    Rmax  = 13
    nR    = 512
    dt    = 1.0
    steps = 1200
    aniskip = 250
    M = 2000
#--------------------------------------------------------
#-------------------   Model        ---------------------
#--------------------------------------------------------

def Hel(R):
    """Hel for Tully 1

    Args:
        R (float): nuclear position

    Returns:
        N x N matrix: Matrix elements of electronic part of the Hamiltonian
    """


    Vij = np.zeros((2,2))
    A = 6*10**-4
    B = 0.1 
    C = 0.9 
    Vij[0,0] = A

    if ( R < 0 ):
        Vij[1,0] = B * np.exp( C*R )
    else:
        Vij[1,0] = B * ( 2 - np.exp( -C*R ) )

    Vij[0,1] = Vij[1,0]
    Vij[1,1] = -A

    return Vij


 

#--------------------------------------------------------
#--------------------------------------------------------
#-------------------   initial Ψ    ---------------------
#--------------------------------------------------------
def psi(R):
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
  α = 1.0 
  R0 =  -9.0
  P0 = 30
  χ = np.exp(- 0.5 * α * (R - R0)**2.0 ) * np.exp( 1j * P0 * (R - R0))
  χ =  χ/(np.sum(χ*χ.conjugate())**0.5)
  
  # Electronic Part
  Φ = np.array([1, 0])
  return  ꕕ(Φ, χ)
