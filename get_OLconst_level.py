import sys

import numpy as np

npz = np.load(sys.argv[1])
u = npz['Irr0_mW_per_mm2']
print(np.mean(u[101:301]))