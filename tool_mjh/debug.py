# import sys
# import os
# current_dir = os.path.dirname(__file__)
# sys.path.append(os.path.dirname(current_dir))
# import torch
#
# from library.sdxl_t2i_adapter import SdxlT2IAdapter
# path = r'/mnt/nfs/file_server/public/mingjiahui/data/debug/result/at-step00050000.ckpt'
# sdxl_adapter = SdxlT2IAdapter(sk=True, use_conv=False)
# checkpoint = torch.load(path, map_location='cuda')
# # test
# state_dict = checkpoint["state_dict"]
# epoch = checkpoint.get("epoch", 0)
# global_step = checkpoint.get("global_step", 0)
#
# print(type(state_dict))
# print(state_dict.keys())
# print(epoch)
# print(global_step)
#
# state_dict = sdxl_adapter.state_dict()
# print(state_dict.keys())

import json
path = "/mnt/nfs/file_server/public/mingjiahui/data/LAION12M-highreso/meta_v1_cache.json"
save_path = r'/mnt/nfs/file_server/public/mingjiahui/data/LAION12M-highreso/meta_v1_cache_debug.json'
res = {}
print('loading json......')
with open(path, 'r')as file:
    data = json.load(file)
    print(f'There are {len(data)} data!')
    for key in list(data.keys())[:10]:
        res[key] = data[key]
print('writing json......')
with open(save_path, 'w')as file:
    json.dump(res, file)

    print(f'Test:{len(res)}')



