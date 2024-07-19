import sys
import time
from Models.constants import *
sys.path.append('./Models')
sys.path.append('./Methods')

start_time = time.time()

# ====== Specify the method ======
# from Methods.Redfield_stupid import func
# from Methods.Redfield_analytic import func
from Methods.Redfield_Mat import func             # the fastest one, use this
# from Methods.Redfield_Mat_2 import func

# ====== Specify the model ======
from Models.SpinBoson_Ohmic import parameters  
# from Models.SpinBoson_Debye import parameters
# from Models.FMO import parameters

# ==============================================================================================
#                                       Parameters passing     
# ==============================================================================================

# ======== Passing the model-specific system and bath parameters ========
NStates = parameters.NStates    # number of states in the reduced system part
beta = parameters.β             # inverse temperature
coeff = parameters.coeff        # bath oscillator coupling coefficients
ω = parameters.ω                # bath oscillator frequencies
ndof = parameters.ndof          # number of bath DOFs
nmod = parameters.nmod          # number of dissipation modes
hams = parameters.hams          # Hs
qmds = parameters.qmds          # {n = 1, ..., nmods | Qn}
rho0 = parameters.rho0          # initial density matrix under the diabatic representation
U = parameters.U                # unitary transformation matrix between the diabatic and adiabatic representations
w_uv = parameters.w_uv          # energy gaps under the adiabatic representation

# Secular approximation, True or False
SA = False
parameters.SA = SA

print(declaration)
print("Hs:")
print(hams)
print("qmds:")
print(qmds[i, :, :] for i in range(nmod))
print("rho0:")
print(rho0)

# ======== Dynamics control parameters ========
t = parameters.t                # total propagation time
dt = parameters.dt              # integration time step & bare-bath TCF discretization spacing
NSteps = int(t / dt)            # number of steps
nskip = parameters.nskip        # step intervals to print data
dd = 0 if (NSteps % nskip == 0) else 1
N_out = NSteps//nskip + dd      # number of data points to record

# ==============================================================================================
#                                       Initialization     
# ==============================================================================================

# ======== switching rho0 to the adiabatic representation ======== 
rho0_ad = U.T.conjugate() @ rho0 @ U

# ======== initial density matrix ======== 
rhot_ad = rho0_ad
iskip = 0
rhoij_File = open("rhot.txt","w")

# ======== Runge-Kutta-4 integrator ========
def rk4(f, y0, tt, h, par):

    k1 = h * f(tt, y0, par)
    k2 = h * f(tt + h / 2, y0 + k1 / 2, par)
    k3 = h * f(tt + h / 2, y0 + k2 / 2, par)
    k4 = h * f(tt + h, y0 + k3, par)
    
    return y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# ==============================================================================================
#                                 Propagation and data recording     
# ==============================================================================================
# Redfield tensors initialization (for different implementations)
parameters.R_ten = np.zeros((NStates, NStates, NStates, NStates), dtype=complex)
parameters.R_ten_1 = np.zeros((NStates, NStates, NStates, NStates), dtype=complex)
parameters.R_ten_2 = np.zeros((NStates, NStates, NStates, NStates), dtype=complex)
parameters.R_ten_3 = np.zeros((NStates, NStates, NStates, NStates), dtype=complex)
parameters.R_ten_4 = np.zeros((NStates, NStates, NStates, NStates), dtype=complex)

for i in range(NSteps - 1):
    
    if (i % nskip == 0):

        # get RDM under the diabatic representation
        rhot_d = U @ rhot_ad @ U.T.conjugate()

        # print time
        rhoij_File.write(f"{round(iskip * nskip * dt, 3)} \t")

        # print real and imaginary parts of the RDM under the diabatic representation
        for m in range(NStates):
            for n in range(NStates):

                rhoij_File.write(str(rhot_d[m, n].real) + "\t")
                rhoij_File.write(str(rhot_d[m, n].imag) + "\t")

        rhoij_File.write("\n")
        iskip += 1
        print("Propagation: \t %.1f %%" %(100 * i / NSteps))

    # dynamics propagation under the adiabatic representation
    rhot_ad = rk4(func, rhot_ad, i * dt, dt, parameters)

    # test trace preserving property
    # print("trace preserving:", np.trace(rhot_ad))

rhoij_File.close()
end_time = time.time()
run_time = end_time - start_time

print("time consumption", run_time, "seconds")
