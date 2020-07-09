import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
import sys

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

name = sys.argv[1]
xs, ys = np.loadtxt(name, delimiter=",").T
plt.plot(xs, ys, color='blue')
smooth_ys = smooth(ys, 100)
print(smooth_ys.shape, smooth_ys)
plt.plot(xs, smooth_ys, color='orange')
plt.ylim(0,600)

# plt.scatter(n_ys, ys)
plt.xlabel("Steps")
plt.ylabel("Episodic Reward per Step")
run_name = name[:-4]
plt.title("Reward for run {}".format(run_name))
plt.savefig("reward_{}.png".format(run_name))
print("Saved figure to reward_{}.png".format(run_name))
