import sys
import os
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_dir))
import sys
import os
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_dir))
import torch
from library.sdxl_t2i_adapter import SdxlT2IAdapter
from library import train_util


def _load_adapter_model(ckpt, device):
    load_stable_diffusion_format = os.path.isfile(ckpt)
    check_point = torch.load(ckpt, map_location=device)
    state_dict = check_point['state_dict']
    epoch = check_point.get('epoch', 0)
    global_step = check_point.get('global_step', 0)

    sdxl_adapter = SdxlT2IAdapter(sk=True, use_conv=False)

    adapter_sd = {}
    for k in list(state_dict.keys()):
        if k.startswith("model.sdxl_adapter."):
            adapter_sd[k.replace("model.sdxl_adapter.", "")] = state_dict.pop(k)
    sdxl_adapter.load_state_dict(adapter_sd)
    del adapter_sd

    return load_stable_diffusion_format, sdxl_adapter, (epoch, global_step)


def load_adapter(accelerator, ckpt,):
    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            print(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}")

            (load_stable_diffusion_format, sd_adapter, ckpt_info, ) = \
                _load_adapter_model(ckpt, accelerator.device)

            import gc
            gc.collect()
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    # sd_adapter = train_util.transform_models_if_DDP([sd_adapter])
    return load_stable_diffusion_format, sd_adapter, ckpt_info
