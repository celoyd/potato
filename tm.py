from skimage import io
from pytorch_wavelets import DTCWTForward, DTCWTInverse
import torch
import numpy as np
from sys import argv

print("setup")
dec = DTCWTForward(J=8, biort='near_sym_b', qshift='qshift_b')
rec = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')

src_path, dst_path = argv[1:]

src = io.imread(src_path).astype(np.float32)
src = src.swapaxes(0, 2)
src = torch.tensor(src).unsqueeze(0) / 65535

print("fwd")
lr, hr = dec(src)

for n in range(len(hr)):
	weight = [2.0, 1, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
	hr[n] *= weight[n]

print("rev")
dst = rec((lr, hr)).squeeze(0)

print("breakdown")
dst = (np.clip(dst.numpy(), 0, 1) * 65535).astype(np.uint16)

dst = dst.swapaxes(0, 2)

io.imsave(dst_path, dst)