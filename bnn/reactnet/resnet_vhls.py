import heterocl as hcl
import hlib
import numpy as np
import sys
from resnet_main import *

batch_size = 1
target = hcl.platform.zc706
target.config(compile="vivado_hls", mode="csyn")

resnet20 = build_resnet20_inf(params,target=target)
print("Finish building function.")
images, labels = next(iter(test_loader))
np_image = images.numpy()
hcl_image = hcl.asarray(np_image, dtype=qtype_float)
resnet20(hcl_image, *hcl_array, hcl_out)
print("Done synthesis.")