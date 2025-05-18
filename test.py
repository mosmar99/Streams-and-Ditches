import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt

rec_data = np.load("logs/m1/test_gat_pca4/reconstruction/1.npz")
image = rec_data["image"]
unet_pred = rec_data["unet_pred"]
image_name = rec_data["image_name"]

gt_path = os.path.join("./data/05m_chips/labels/", f"{image_name}.tif")
gt = tifffile.imread(gt_path)

vmax = 2
vmin = 0


fig, ax = plt.subplots(1,3)

ax[0].imshow(gt, vmax=vmax, vmin=vmin)
ax[1].imshow(image)
ax[2].imshow(unet_pred, vmax=vmax, vmin=vmin)
plt.show()