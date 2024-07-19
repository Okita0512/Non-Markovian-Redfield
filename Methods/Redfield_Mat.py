import numpy as np

"""
This implementation construct the Redfield tensor using np.kron, calculate tensor-matrix product using np.einsum,
and update the Redfield tensor on-the-fly, which is very fast (no need to use 4-fold cycles). 
One can potentially further accelerate this algorithm using numba, but need to take care of the data type. 

Nevertheless, it would be inconvenient to implement the secular approximation (compared to using 4-fold cycles). 
Here we didn't include it. Need to further work it out if desired. 

"""

# ====== Redfield tensor plus the coherent term ======
def func(tn, rho_ad, par): 
    
    NStates = par.NStates
    nmod = par.nmod
    qmds_ad = par.qmds_ad
    En = par.En
    dt = par.dt
    istep = int(tn / dt)
    C_ab = par.C_ab
    R_ten_1 = par.R_ten_1
    R_ten_2 = par.R_ten_2
    R_ten_3 = par.R_ten_3
    R_ten_4 = par.R_ten_4

    # initialize with coherent commutator
    R_rho = - 1.0j * (np.diag(En) @ rho_ad - rho_ad @ np.diag(En))

    # further add the non-Markovian relaxation terms, the positive terms
    R_rho += np.einsum('ijkl, kj', R_ten_2, rho_ad) + np.einsum('ijkl, kj', R_ten_3, rho_ad)
    
    # further add the non-Markovian relaxation terms, the negative terms
    R_rho += - np.einsum('ijjk, kl', R_ten_1, rho_ad) - np.einsum('ij, jkkl', rho_ad, R_ten_4)

    # update the Redfield tensors
    for n in range(nmod):
        Sn = qmds_ad[n, :, :]
        Sn_tau = np.diag(np.exp(- 1.0j * En * tn)) @ Sn @ np.diag(np.exp(1.0j * En * tn))
        TCF_n = C_ab[n, istep]
        R_ten_1 += TCF_n * dt * np.kron(Sn, Sn_tau).reshape(NStates, NStates, NStates, NStates)
        R_ten_2 += TCF_n * dt * np.kron(Sn_tau, Sn).reshape(NStates, NStates, NStates, NStates)
        R_ten_3 += TCF_n.conjugate() * dt * np.kron(Sn, Sn_tau).reshape(NStates, NStates, NStates, NStates)
        R_ten_4 += TCF_n.conjugate() * dt * np.kron(Sn_tau, Sn).reshape(NStates, NStates, NStates, NStates)

    par.R_ten_1 = R_ten_1
    par.R_ten_2 = R_ten_2
    par.R_ten_3 = R_ten_3
    par.R_ten_4 = R_ten_4
    
    return R_rho
