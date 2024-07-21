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
