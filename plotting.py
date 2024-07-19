import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()

# ==============================================================================================
#                                         data reading     
# ==============================================================================================

# plot population
data2 = np.loadtxt("rhot.txt", dtype = float)
plt.plot(data2[:, 0], data2[:, 1] - data2[:, 7], "-", linewidth = 2.0, color = 'red', label = "sigmaz")
#plt.plot(data2[:, 0], data2[:, 7], "-", linewidth = 2.0, color = 'navy', label = "P2")

plt.legend(frameon = False)
# plt.xlim(0, 12)
# plt.ylim(-2, 2)
plt.show()

# plot coherence
plt.plot(data2[:, 0], data2[:, 3], "-", linewidth = 2.0, color = 'red', label = "coherence_real")

plt.legend(frameon = False)
# plt.xlim(0, 12)
# plt.ylim(-2, 2)
# plt.show()

plt.show()