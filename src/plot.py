import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint

file = "../results/data.npz"

data = np.load(file, allow_pickle=True)

data = data["results"].item()

ct_s = list(data.keys())
ct_s.remove("gt")
errs = {ct: data[ct]["err"] for ct in ct_s}

pprint(list(enumerate(zip(errs["rtvs"], errs["ours"]))))

for ct in ct_s:
    plt.plot(errs[ct], label=ct)
plt.legend()
plt.savefig("err.png")
