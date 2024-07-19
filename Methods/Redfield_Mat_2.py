import numpy as np

"""
This implementation construct the Redfield tensor using 4-fold cycle, calculate tensor-matrix product using 4-fold cycle again,
and update the Redfield tensor on-the-fly, which is still very slow. 

"""

# ====== Redfield tensor plus the coherent term ======
def func(tn, rho_ad, par): 
    
    NStates = par.NStates
    nmod = par.nmod
    qmds_ad = par.qmds_ad
    En = par.En
    w_uv = par.w_uv
    dt = par.dt
    istep = int(tn / dt)
    C_ab = par.C_ab
    SA = par.SA
    R_ten = par.R_ten

    # initialize with coherent commutator
    R_rho = - 1.0j * (np.diag(En) @ rho_ad - rho_ad @ np.diag(En))

    # further add the relaxation terms
    for a in range(NStates):
        for b in range(NStates):
            for c in range(NStates):
                for d in range(NStates):

                    R_rho[a, b] += (rho_ad[c,d] * R_ten[d,b,a,c]                      # + Γ_dbac (t)
                                  + rho_ad[c,d] * R_ten[c,a,b,d].conjugate()          # + Γ*_cabd (t)
                                  - rho_ad[d,b] * R_ten[a,c,c,d]                      # - δ_bd Σ_m Γ_ammc (t)
                                  - rho_ad[a,c] * R_ten[b,d,d,c].conjugate())         # - δ_ac Σ_m Γ*_bmmd (t)
                    
    # update Redfield tensor and save
    C_ten = np.zeros((NStates, NStates, NStates, NStates),dtype=complex)
    for a in range(NStates):
        for b in range(NStates):
            for c in range(NStates):
                for d in range(NStates):

                    # applying secular approximation or not
                    if SA:
                        filter = (a == c) * (b == d) + (a == d) * (b == c) + (a == b) * (c == d)
                    else:
                        filter = 1
            
                    for n in range(nmod):
                        Sn = qmds_ad[n, :, :]
                        C_ten[a,b,c,d] += filter * C_ab[n,istep] * Sn[a,b] * Sn[c,d] * np.exp(- 1.0j * w_uv[c,d] * tn)

    par.R_ten += C_ten * dt
    
    return R_rho
