import numpy as np
from constants import *
from exc_basis import get_u_Omega_uv

def coth(x):
    return (1 + np.exp(-2 * x)) / (1 - np.exp(-2 * x))

def bathParam(λ, ωc, ndof):     # discretizing the Drude-Lorentz spectral density. Again, could be problematic...

    c = np.zeros(( ndof ))
    ω = np.zeros(( ndof ))
    for d in range(ndof):
        ω[d] =  ωc * np.tan( np.pi * (1 - (d + 1)/(ndof + 1)) / 2)
        c[d] =  np.sqrt(2 * λ / (ndof + 1)) * ω[d]

    return c, ω

def get_Hs(NStates):

    hams = np.zeros(( NStates, NStates ), dtype = complex)
    
    # numbers in cm^-1
    hams[0,:] = np.array([12410.0, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9]) * cm2au
    hams[1,:] = np.array([-87.7, 12530.0, 30.8, 8.2, 0.7, 11.8, 4.3]) * cm2au
    hams[2,:] = np.array([5.5, 30.8, 12210.0, -53.5, -2.2, -9.6, 6.0]) * cm2au
    hams[3,:] = np.array([-5.9, 8.2, -53.5, 12320.0, -70.7, -17.0, -63.3]) * cm2au
    hams[4,:] = np.array([6.7, 0.7, -2.2, -70.7, 12480.0, 81.1, -1.3]) * cm2au
    hams[5,:] = np.array([-13.7, 11.8, -9.6, -17.0, 81.1, 12630.0, 39.7]) * cm2au
    hams[6,:] = np.array([-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 12440.0]) * cm2au

    return hams

def get_rho0(NStates):

    rho0 = np.zeros((NStates, NStates), dtype = complex)
    rho0[0, 0] = 1.0 + 0.0j

    return rho0

class parameters():

    # propagation
    t = 10000 * fs2au            # total propagation time: 1 ps or 10 ps
    dt = 0.1 * fs2au             # time step
    NSteps = int(t / dt)            # number of steps
    nskip = 10

    # MODEL-SPECIFIC ITEMS
    NStates = 7                     # number of states
    ndof = 300                      # number of bath modes per site

# ======== specify temperature, usually 77 K or 300 K ========

#    temp = 77 * K2au
    temp = 300 * K2au

# ============================================================

    β = 1 / temp
    λ = 35 * cm2au               # reorganization energy
    ωc = 106.14 * cm2au          # characteristic frequency

    # produce the Hamiltonian, initial RDM
    hams = get_Hs(NStates)
    rho0 = get_rho0(NStates)
    
    # bath and system-bath coupling parameters
    nmod = 7         # number of dissipation modes
    coeff = np.zeros((nmod, ndof), dtype = complex)
    C_ab = np.zeros((nmod, NSteps), dtype = complex)    # bare-bath TCFs
    qmds = np.zeros((nmod, NStates, NStates), dtype = complex)
    for n in range(nmod):
        coeff[n, :], ω  = bathParam(λ, ωc, ndof)
        for i in range(NSteps):
            C_ab[n, i] = 0.25 * np.sum((coeff[n, :]**2 / (2 * ω)) * (coth(β * ω / 2) * np.cos(ω * i * dt) - 1.0j * np.sin(ω * i * dt)))

    for n in range(nmod):
        coeff[n, :], ω  = bathParam(λ, ωc, ndof)
        qmds[n, n, n] = 1

    # featured for Redfield
    U, En, w_uv = get_u_Omega_uv(hams, NStates)
    qmds_ad = np.zeros((nmod, NStates, NStates), dtype = complex)
    for i in range(nmod):           
        qmds_ad[i, :, :] = U.T.conjugate() @ qmds[i, :, :] @ U


