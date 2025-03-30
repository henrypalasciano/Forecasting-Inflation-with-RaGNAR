import numpy as np
from math import comb
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

# Probability of a node being in the stage 1 neighbour set
pi = 0.03
probs = np.zeros((2, 115))
# Iterate over all possible number of nodes in the first stage
for i in range (115):
    p1 = comb(114, i) * pi ** i * (1 - pi) ** (114 - i)
    probs[0, i] = p1
    # Probability of a node being in the stage 2 neighbour set
    pi_2 = 1 - (1 - pi) ** i
    # Iterate over all possible number of nodes in the second stage
    for j in range(114 - i):
        p2 = comb(114 - i, j) * pi_2 ** j * (1 - pi_2) ** (114 - i - j)
        probs[1, j] += p1 * p2

fig, axs = plt.subplots(1, 2, figsize=(15, 3.5))
axs[0].bar(np.arange(15), probs[0,:15])
axs[1].bar(np.arange(35), probs[1,:35])
for ax in axs:
    ax.grid(True)
    ax.set_xlabel("Number of Nodes", fontsize=12)
    ax.tick_params(axis='both', which='both', bottom=True, top=False, left=True, right=False, labelsize=12)
axs[0].set_ylabel("Probability", fontsize=12)
axs[0].set_title("Stage 1 Neighbour Set", fontsize=14)
axs[1].set_title("Stage 2 Neighbour Set", fontsize=14)

plt.subplots_adjust(wspace=0.125) 
plt.savefig("figures/figure_2.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()