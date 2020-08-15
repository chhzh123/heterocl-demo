import heterocl as hcl
import hlib
import numpy as np
import sys
from resnet_main import *

if len(sys.argv) == 1:
    resnet20 = build_resnet20_inf(params)
else:
    resnet20 = build_resnet20_opt_inf(params)
print("Finish building function.")

images, labels = next(iter(test_loader))
np_image = images.numpy()
hcl_image = hcl.asarray(np_image, dtype=qtype_float)
resnet20(hcl_image, *hcl_array, hcl_out)
print("Done synthesis.")