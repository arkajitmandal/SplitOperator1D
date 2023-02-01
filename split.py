import math
import time
import numpy as np
import os
import Potential as Pt
from numpy import linalg as LA
import sys
from tools import *
sys.path.append(os.popen("pwd").read().replace("\n","")+"/Model")




modelName = sys.argv[1]
exec(f"import {modelName} as model")
param = model.parameters()
#--------------------------------
dt = param.dt
tsteps = param.steps
Time = range(tsteps)

# Intial parameters
Rmin = param.Rmin
Rmax = param.Rmax
nR = param.nR
aniskip = param.aniskip 
#---------------------------------
dR = float((Rmax-Rmin)/nR)
R = np.arange(Rmin,Rmax,dR)
#---------------------------------

#---------------------------------
# exponential of V in adiabatic eigenrepresentation
def expV(ei, dt):
  return np.exp(-1j * dt * ei, dtype=np.complex64) 
# exponential of T in momentum (FFT) representation
def expT(dR, dt, nR, mass):
  p = np.fft.fftfreq(nR) * (2.0 * np.pi/ dR)
  return np.exp(-1j * dt * (p * p)/(2 * mass), dtype=np.complex64) 


#---------------------------------
#    MAIN CODE
#---------------------------------

#ve = Pt.electronic(Rmin,Rmax,n_steps,nf)
Ep, Up = Pt.adiabatic(R, model.Hel)
nState = Up.shape[1]
UV = expV(Ep, dt/2)
UT = expT(dR, dt, nR, param.M)

# Initial Wf
cD0 = model.psi(R)
#---------------------------------

# files
popA = open(f"Results/popA-{modelName}.txt","w+") # Adiabatic Population
popD = open(f"Results/popD-{modelName}.txt","w+") # Diabatic Population
wf =   open("Results/psi.txt","w+")
cDt = cD0
#-----------------------------

for t in Time:
  print (t)
  # population in diabatic representation
  rhoD = population(cDt, nState)
  popD.write(str(t*dt) + " " + " ".join(rhoD.astype(str)) + "\n" )
  
  # population in polaritonic representation
  cPt = DtoA(cDt, nR, nState, Up)  
  rhoP = population(cPt, nState) 
  popA.write(str(t*dt) + " " + " ".join(rhoP.astype(str)) + "\n" ) 
  
  # write wavefunction
  if (t%aniskip == 0):
    for i in range(nR):
      density = np.zeros(nState,dtype=np.float32) 
      for j in range(nState):
        density[j] =  (cPt[j*nR + i].conjugate() * cPt[j*nR + i]).real
      wf.write( str( Rmin +  dR * i)  + " "  + " ".join(density.astype(str)) )
      for j in range(nState):
        wf.write( " " + str(Ep[j * nR + i].real) )
      wf.write("\n")
    wf.write("\n\n")	  
  #--------------------

  # evolution 1st step 
  cPt = UV * cPt 
  cDt = AtoD(cPt, nR, nState, Up) 

  # evolution 2nd step
  fDt = np.zeros(len(cDt), dtype=np.complex64) 
  for i in range(nState):
    # FFT
    fDt[ nR * i: nR * (i + 1) ]  = np.fft.fft(cDt[ nR * i: nR * (i + 1) ], norm = 'ortho')
    # evolution in FFT
    fDt[ nR * i: nR * (i + 1) ] = UT * fDt[ nR * i: nR * (i + 1) ] 

  # iFFT
  for i in range(nState):
    cDt[ nR * i: nR * (i + 1) ]  = np.fft.ifft(fDt[ nR * i: nR * (i + 1) ], norm = 'ortho')
 
  # evolution 3rd step
  cPt = DtoA(cDt, nR, nState, Up)  
  cPt = UV * cPt 
  cDt = AtoD(cPt, nR, nState, Up) 

   
wf.close()
popD.close()       
popA.close()            
