import numpy as np

"""
Directly solve the integro-differential equation. This implementation is correct but too slow, which is actually stupid. 
One can actually take advantage of the integration outcome of the previous steps, and update the Redfield tensor on-the-fly, which will be much faster.
See the Redfield_Mat.py and Redfield_Mat_2.py for details. 

"""

# ====== Redfield tensor plus the coherent term ======
def func(tn, rho_ad, par): 
    
    nmod = par.nmod
    qmds_ad = par.qmds_ad
    En = par.En
    dt = par.dt
    C_ab = par.C_ab
    NSteps_tn = int(tn / dt)

    # initialize with coherent commutator
    R_rho = - 1.0j * (np.diag(En) @ rho_ad - rho_ad @ np.diag(En))

    # further add the non-Markovian relaxation terms
    for n in range(nmod):
        Sn = qmds_ad[n, :, :]
        for i in range(NSteps_tn):
            Sn_tau = np.diag(np.exp(- 1.0j * En * tn)) @ Sn @ np.diag(np.exp(1.0j * En * tn))
            R_rho += - C_ab[n, i] * (Sn @ Sn_tau @ rho_ad - Sn_tau @ rho_ad @ Sn) * dt
            R_rho += C_ab[n, i].conjugate() * (Sn @ rho_ad @ Sn_tau - rho_ad @ Sn_tau @ Sn) * dt

    return R_rho
