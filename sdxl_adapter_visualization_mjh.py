# %%
import os.path

import numpy as np
import torch
from tqdm import tqdm
from library import model_util, sdxl_model_util

adapter_ckpt_path = "/mnt/nfs/file_server/public/mingjiahui/data/debug/result/at-step00030000.ckpt"
sd_adapter, _ = sdxl_model_util.load_adapters_from_sdxl_checkpoint(sdxl_model_util.MODEL_VERSION_SDXL_BASE_V0_9, adapter_ckpt_path, "cpu")


# %%
from PIL import Image
img_path = '/mnt/nfs/file_server/public/mingjiahui/data/LAION12M-highreso/depth/00000/000000011-midas_v21_small_256.png'
x = torch.FloatTensor(np.array(Image.open(img_path).convert('L').resize((1024, 1024)))).unsqueeze(0).unsqueeze(0)
print(x.shape)
y = sd_adapter(x)

# # %%
# for name, param in sd_adapter.named_parameters():
#     print(name, param.shape
# # %%
for feature in y:
    print(f'{feature.shape} {feature.dtype} {feature.device}')
# %%
# plt the features
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2

def plot_feature(feature, title):
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(feature, cmap='gray')
    ax.set_title(title)
    plt.show()

print(f'feature:{feature[0].shape}')
features = [i.squeeze(0).detach().numpy() for i in y]
for i in range(len(features)):
    res = np.mean(features[i], axis=0)
    res = 255 * (res - np.min(res)) / (np.max(res) - np.min(res))
    res = res.astype(np.uint8)
    print(res)

    save_name = os.path.splitext(os.path.basename(adapter_ckpt_path))[0]+'.png'
    save_path = fr'/home/mingjiahui/data/debug/{i}_{save_name}'
    cv2.imwrite(save_path, res)
    print(f"done! img has writed in >>{save_path}<<")


# for i in range(features[0].shape[0]):
#     print(features[0][i].shape)
#     res = np.mean(features[0][i])
#     print(f'res：{res.shape}')
#     plot_feature(res, "feature")
#     print(np.mean(features[0][i]), "feature")
# %%
