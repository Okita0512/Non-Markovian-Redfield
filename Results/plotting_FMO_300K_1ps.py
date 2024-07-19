import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, tick_params
import numpy as np
fig, ax = plt.subplots()

font = {'family':'Times New Roman', 'weight': 'roman', 'size':12}
fig = plt.figure(figsize=(7, 7), dpi = 128)

# ==============================================================================================
#                                         data reading     
# ==============================================================================================
fs_to_au = 41.341   
lw = 2

data2 = np.loadtxt("FMO_300K_Redfield.txt", dtype = float)
x = 1
plt.plot(data2[:,0] / (1000 * fs_to_au), data2[:,x], "--", linewidth = lw, color = 'blue', label = "Redfield, 1")
plt.plot(data2[:,0] / (1000 * fs_to_au), data2[:,x + 16], "--", linewidth = lw, color = 'orange', label = "Redfield, 2")
plt.plot(data2[:,0] / (1000 * fs_to_au), data2[:,x + 16 * 2], "--", linewidth = lw, color = 'green', label = "Redfield, 3")
plt.plot(data2[:,0] / (1000 * fs_to_au), data2[:,x + 16 * 3], "--", linewidth = lw, color = 'red', label = "Redfield, 4")
plt.plot(data2[:,0] / (1000 * fs_to_au), data2[:,x + 16 * 4], "--", linewidth = lw, color = 'purple', label = "Redfield, 5")
plt.plot(data2[:,0] / (1000 * fs_to_au), data2[:,x + 16 * 5], "--", linewidth = lw, color = 'peru', label = "Redfield, 6")
plt.plot(data2[:,0] / (1000 * fs_to_au), data2[:,x + 16 * 6], "--", linewidth = lw, color = 'pink', label = "Redfield, 7")

data2 = np.loadtxt("FMO_300K.dat", dtype = float)
plt.plot(data2[:, 0] / (1000 * fs_to_au), data2[:, 1], "-", linewidth = lw, color = 'blue', label = "HEOM, 1")
plt.plot(data2[:, 0] / (1000 * fs_to_au), data2[:, 2], "-", linewidth = lw, color = 'orange', label = "HEOM, 2")
plt.plot(data2[:, 0] / (1000 * fs_to_au), data2[:, 3], "-", linewidth = lw, color = 'green', label = "HEOM, 3")
plt.plot(data2[:, 0] / (1000 * fs_to_au), data2[:, 4], "-", linewidth = lw, color = 'red', label = "HEOM, 4")
plt.plot(data2[:, 0] / (1000 * fs_to_au), data2[:, 5], "-", linewidth = lw, color = 'purple', label = "HEOM, 5")
plt.plot(data2[:, 0] / (1000 * fs_to_au), data2[:, 6], "-", linewidth = lw, color = 'peru', label = "HEOM, 6")
plt.plot(data2[:, 0] / (1000 * fs_to_au), data2[:, 7], "-", linewidth = lw, color = 'pink', label = "HEOM, 7")

# ==============================================================================================
#                                      plotting set up     
# ==============================================================================================

# x and y range of plotting 

# x-axis range: (0, time)
time = 1               # for short time behaviors, 1 ps
# time = 10              # for long time behaviors, 10 ps

# population range
y1, y2 = 0.0, 1.0     # y-axis range: (y1, y2)

# scale for major and minor locator
x_major_locator = MultipleLocator(0.2)
x_minor_locator = MultipleLocator(0.1)
y_major_locator = MultipleLocator(0.5)
y_minor_locator = MultipleLocator(0.1)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 8, labelsize = 10)
ax.tick_params(which = 'minor', length = 4)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params(labelsize = 20, which = 'both', direction = 'in', pad = 10)
plt.xlim(0.0, time)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 8)
ax2.tick_params(which = 'minor', length = 4)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

# name of x, y axis and the panel
ax.set_xlabel('time / ps', font = 'Times New Roman', size = 20)
ax.set_ylabel(r'Population', font = 'Times New Roman', size = 20)
ax.set_title('The Fenna-Matthews-Olson (FMO) pigment-protein complex @ 300K', font = 'Times New Roman', size = 15, pad = 20)

# legend location, font & markersize
ax.legend(loc = 'upper right', prop = font, markerscale = 1, ncol = 2)
plt.legend(frameon = False)

# plt.show()

plt.savefig("FMO 300K 1ps.pdf", bbox_inches='tight')