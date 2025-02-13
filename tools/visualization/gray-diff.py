from imageio import v3 as io
import numpy as np
from sys import argv

A, B = (io.imread(argv[x]).astype(np.float32) for x in (1, 2))

nd = (((A - B)/(A + B)) + 1)/2

nd = (nd * 65535).astype(np.uint16)

io.imwrite(argv[3], nd)