import numpy as np

"""
This implementation takes advantage of the analytic integration result of the time-dependent Redfield tensors for Caldeira-Leggett model 
(with discretized bath spectral density), which is clear to understand and can serve as benchmark for other types of implementations. 

However, the limitations are 
(1) one still needs to re-construct the Redfield tensor every step, especially for using 4-fold cycles to build the Redfield tensor 
and calculate the tensor-matrix product, which is very slow. 
(2) complicated bath TCFs might not be analytically integrable, for example, under the polaron-transformed representation. 
(3) suffers from Poincare recurrence issue if unconverged (same for other implementations with discretized bath).

"""

# ====== Auxiliary functions ======
def Bose(beta, w):                    # Bose-Einstein distribution function
    return 1 / (np.exp(beta * w) - 1)

def wn_minus(ω, wi, tcorr):        # absorption

    tt = np.empty_like(ω, dtype = complex)
    tt = (1 - np.exp(1j * tcorr * (wi - ω))) / (wi - ω)

    return tt

def wn_plus(ω, wi, tcorr):         # emission

    tt = np.empty_like(ω, dtype = complex)
    tt = (1 - np.exp(1j * tcorr * (ω + wi))) / (ω + wi)

    return tt

# ====== Redfield tensor plus the coherent term ======
def func(tn, rho_ad, par): 
    
    NStates, w_uv, ω, coeff, qmds, beta, SA = par.NStates, par.w_uv, par.ω, par.coeff, par.qmds, par.β, par.SA
    nmod = par.nmod

    """
    NStates:    number of system states;
    w_uv:       energy gaps under the exciton basis;
    rho_ad:     RDM under adiabatic representation;
    ω:          bath phonon frequencies;
    coeff:      electron-phonon coupling strength;
    qmds:       system dissipation modes;
    beta:       inverse temperature;
    tn:         time slice;
    SA:         secular approximation, true or false;
    nmod:       number of system / bath dissipation modes.

    """

    # constructing the Redfield tensor (flattened)
    R_ten=np.zeros((NStates**2, NStates**2), dtype = complex)
    γ_tn = np.zeros((nmod), dtype = complex)

    # construct the Redfield tensor
    for x in range(NStates**2):
        for y in range(NStates**2):

            a = x // NStates
            b = x % NStates
            c = y // NStates
            d = y % NStates

            # applying secular approximation or not
            if SA == True:
                filter = (a == c) * (b == d) + (a == d) * (b == c) + (a == b) * (c == d)
            else:
                filter = 1

            for n in range(nmod):

                # Here different bath modes are independent, J_mn(w) = J_mn(w) * delta_mn
                prefactor = coeff[n, :]**2 / (2 * ω)
                gamma_t = (Bose(beta, ω) + 1.) * wn_minus(ω, w_uv[d, c], tn) + Bose(beta, ω) * wn_plus(ω, w_uv[d, c], tn)

                # summation performed on ndof
                γ_tn[n] = 1.0j * np.sum(prefactor * gamma_t)
                R_ten[x, y] += qmds[n, a, b] * qmds[n, c, d] * γ_tn[n] * filter

    # multiply the Redfield tensor by the RDM elements
    R_rho = np.zeros((NStates, NStates), dtype = complex)

    for a in range(NStates):
        for b in range(NStates):
            for c in range(NStates):
                for d in range(NStates):

                    R_rho[a, b] += (rho_ad[c, d] * R_ten[d * NStates + b, a * NStates + c]                      # + Γ_dbac (t)
                                  + rho_ad[c, d] * R_ten[c * NStates + a, b * NStates + d].conjugate()          # + Γ*_cabd (t)
                                  - rho_ad[d, b] * R_ten[a * NStates + c, c * NStates + d]                      # - δ_bd Σ_m Γ_ammc (t)
                                  - rho_ad[a, c] * R_ten[b * NStates + d, d * NStates + c].conjugate())         # - δ_ac Σ_m Γ*_bmmd (t)
            
            # further including the coherent term - i w_ab [ρs]_ab (t)
            R_rho[a, b] += - 1.0j * w_uv[a, b] * rho_ad[a, b]

    return R_rho
