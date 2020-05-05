import heterocl as hcl
import hlib
import numpy as np
from bnn_main import *

batch_size = 100

f = build_bnn_inf(batch_size,"vhls")

with open("bnn.cpp","w") as outfile:
    outfile.write(f)