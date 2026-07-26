# Non-Markovian Redfield Equation for Open Quantum System Dynamics

First released on 07/19/2024

Authors: 

Wenxiang Ying, wying3@ur.rochester.edu

Kaiyue Peng, kaiyue_peng@berkeley.edu

This is a Non-Markovian Redfield equation code composed in python, a bunch of example model systems and results are also attached. 
To run the code, simply modify the "run.py" file with a method and a model desired (which can also be modified accordingly), and type "python3 run.py" at your terminal (can be your local machine). 

There are several different ways of numerical implementation for this method. 
Usually, one suffers from the computational cost of constructing the Redfield tensor and calculating the tensor-matrix product, which are usually achieved using 4-fold cycles, making the algorithm complexity at least O(N^4).
Here instead of using 4-fold cycles, we construct the Redfield tensor using np.kron, calculate tensor-matrix product using np.einsum, and update the Redfield tensor on-the-fly, which is very fast (see the "Redfield_Mat.py" file).

Update on 07/20/2024 with a tutorial note pdf file.

## Constant-bias time-dependent polaron NZ2

The repository now also contains a constant-bias implementation of the
identity-initialized time-dependent polaron Nakajima-Zwanzig equation for the
longitudinal spin-boson model.  It can run the existing classic Ohmic and Debye
benchmark parameter sets (models 3-19) without introducing a Landau-Zener
sweep.

A quick comparison of the time-dependent and static polaron frames is:

```bash
python run_tdpolaron.py --model 8 --method both --dt 0.01 --tmax 5 --plot
```

A production-resolution run using the time step stored with model 8 is:

```bash
python run_tdpolaron.py --model 8 --method td
```

The direct nonstationary memory integral scales as O(N_t^2), so production
runs can be substantially more expensive than the original TCL2/Redfield
script.  When Numba is installed, `--backend auto` (the default) uses a JIT-
compiled kernel; `--backend numpy` is always available as a dependency-free
fallback.  Outputs and numerical diagnostics are written under
`Results/TDPolaron/`.

See [`TD_POLARON.md`](TD_POLARON.md) for equations, Hamiltonian conventions,
model mapping, validation tests, and physical limitations.
