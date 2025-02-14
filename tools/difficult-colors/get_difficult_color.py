from sys import argv

import rasterio
from rasterio import windows

stub_path = argv[1]
x = int(argv[2])
y = int(argv[3])

pan_path = stub_path + "-pan.tif"
mul_path = stub_path + "-ms.tif"

pan_window = windows.Window((x // 4) * 4, (y // 4) * 4, 4, 4)
mul_window = windows.Window(x // 4, y // 4, 1, 1)

with rasterio.open(pan_path) as pan:
    pan_patch = pan.read(window=pan_window)

with rasterio.open(mul_path) as mul:
    mul_patch = mul.read(window=mul_window)

reflectances = [pan_patch.mean().item(), *list(int(x) for x in mul_patch.flatten())]

print(",".join(str(r) for r in reflectances))
