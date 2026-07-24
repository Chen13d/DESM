import os
import torch
import natsort
import math
import cv2
import yaml
import matplotlib.pyplot as plt
import numpy as np
import random as pyrandom


from PIL import Image
from skimage.metrics import mean_squared_error
from skimage import transform, measure
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from openpyxl import Workbook
from random import random,  randint
from torch.functional import Tensor
from torch.utils.data import DataLoader, Dataset


def check_existence(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        remove_list = os.listdir(dir)
        for i in range(len(remove_list)):
            remove_dir = os.path.join(dir, remove_list[i])
            os.remove(remove_dir)


def resize(input, size):
    m1 = np.max(input)
    resized = transform.resize(input, size)
    m2 = np.max(resized)
    output = resized * m1 / m2
    return np.uint16(output)


def to_cpu(input):
    input = Tensor.cpu(input)
    input = input.detach().numpy()
    return input


def getStat(cal_dataset, device):
    """Compute mean and variance for training data :param train_data: ClassDataset(ImageFolder) :return: (mean, std)"""
    print('Compute mean and variance for training data.')
    print(len(cal_dataset))
    cal_loader = DataLoader(
        cal_dataset, batch_size=1, shuffle=False, num_workers=0,
        )
    mean = torch.zeros(1)
    std = torch.zeros(1)
    for X, _ in cal_loader:
        for d in range(1):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(cal_dataset))
    std.div_(len(cal_dataset))
    #return list(mean.numpy()), list(std.numpy())
    return list(mean), list(std)


# =============================================================================
# Checkpoint and pretrained weight helpers
# =============================================================================

def path_is_set(path):
    return path is not None and str(path).strip() not in ("", "None")


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_state_dict(model):
    return unwrap_model(model).state_dict()


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def looks_like_state_dict(obj):
    return isinstance(obj, dict) and obj and all(isinstance(k, str) for k in obj.keys()) and all(torch.is_tensor(v) for v in obj.values())


def extract_state_dict(obj, preferred_key=None):
    if hasattr(obj, "state_dict"):
        return get_state_dict(obj)
    if isinstance(obj, dict):
        if preferred_key is not None and preferred_key in obj:
            if obj[preferred_key] is None:
                raise ValueError(f"Cannot find {preferred_key} in the loaded checkpoint.")
            return extract_state_dict(obj[preferred_key])
        if preferred_key is not None and any(key in obj for key in ("net_G", "net_D", "optim_G", "optim_D", "epoch")):
            raise ValueError(f"Cannot find {preferred_key} in the loaded checkpoint.")
        keys = ("state_dict", "model_state_dict", "model", "net")
        if preferred_key is None:
            keys += ("net_G", "net_D")
        for key in keys:
            if key in obj and obj[key] is not None:
                return extract_state_dict(obj[key])
    if looks_like_state_dict(obj):
        return obj
    raise ValueError("Cannot find a model state_dict in the loaded file.")


def load_state_dict(model, state_dict, strict=False):
    target = unwrap_model(model)
    try:
        return target.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        fixed = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        return target.load_state_dict(fixed, strict=strict)


def load_pretrained_weights(path, model, device, preferred_key=None, strict=False, name="model"):
    loaded = torch_load(path, map_location=device)
    state_dict = extract_state_dict(loaded, preferred_key=preferred_key)
    result = load_state_dict(model, state_dict, strict=strict)
    print(f"[Pretrain] Loaded {name}: {path}")
    if not strict and hasattr(result, "missing_keys"):
        print(f"[Pretrain] missing_keys={len(result.missing_keys)}, unexpected_keys={len(result.unexpected_keys)}")


def save_checkpoint(path, net_main, epoch, global_step,
                    best_score_pixel=None, best_score_pear=None, opt_cfg=None):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "net_G": get_state_dict(net_main.net_G),
        "optim_G": net_main.optim_G.state_dict(),
        "scheduler_G": net_main.scheduler_G.state_dict()
            if hasattr(net_main, "scheduler_G") and net_main.scheduler_G is not None else None,
        "net_D": get_state_dict(net_main.net_D)
            if hasattr(net_main, "net_D") else None,
        "optim_D": net_main.optim_D.state_dict()
            if hasattr(net_main, "optim_D") and net_main.optim_D is not None else None,
        "scheduler_D": net_main.scheduler_D.state_dict()
            if hasattr(net_main, "scheduler_D") and net_main.scheduler_D is not None else None,
        "best_score_pixel": best_score_pixel,
        "best_score_pear": best_score_pear,
        "opt": opt_cfg,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_numpy": np.random.get_state(),
        "rng_python": pyrandom.getstate(),
    }
    torch.save(ckpt, path)


def load_checkpoint(path, net_main, device, strict=True):
    ckpt = torch_load(path, map_location=device)

    load_state_dict(net_main.net_G, ckpt["net_G"], strict=strict)
    net_main.optim_G.load_state_dict(ckpt["optim_G"])
    if ckpt.get("scheduler_G") is not None and hasattr(net_main, "scheduler_G") and net_main.scheduler_G is not None:
        net_main.scheduler_G.load_state_dict(ckpt["scheduler_G"])

    if ckpt.get("net_D") is not None and hasattr(net_main, "net_D"):
        load_state_dict(net_main.net_D, ckpt["net_D"], strict=strict)
    if ckpt.get("optim_D") is not None and hasattr(net_main, "optim_D") and net_main.optim_D is not None:
        net_main.optim_D.load_state_dict(ckpt["optim_D"])
    if ckpt.get("scheduler_D") is not None and hasattr(net_main, "scheduler_D") and net_main.scheduler_D is not None:
        net_main.scheduler_D.load_state_dict(ckpt["scheduler_D"])

    st = ckpt.get("rng_torch", None)
    if st is not None:
        try:
            if torch.is_tensor(st):
                st = st.detach().cpu()
                if st.dtype != torch.uint8:
                    st = st.to(torch.uint8)
            else:
                st = torch.tensor(st, dtype=torch.uint8)
            torch.set_rng_state(st)
        except Exception as e:
            print(f"[Resume] Skip rng_torch restore due to: {e}")

    stc = ckpt.get("rng_cuda", None)
    if torch.cuda.is_available() and stc is not None:
        try:
            n_dev = torch.cuda.device_count()
            fixed = []
            for s in stc[:n_dev]:
                if torch.is_tensor(s):
                    s = s.detach().cpu()
                    if s.dtype != torch.uint8:
                        s = s.to(torch.uint8)
                    fixed.append(s)
                else:
                    fixed.append(torch.tensor(s, dtype=torch.uint8))
            torch.cuda.set_rng_state_all(fixed)
        except Exception as e:
            print(f"[Resume] Skip rng_cuda restore due to: {e}")

    if ckpt.get("rng_numpy", None) is not None:
        try:
            np.random.set_state(ckpt["rng_numpy"])
        except Exception as e:
            print(f"[Resume] Skip rng_numpy restore due to: {e}")
    if ckpt.get("rng_python", None) is not None:
        try:
            pyrandom.setstate(ckpt["rng_python"])
        except Exception as e:
            print(f"[Resume] Skip rng_python restore due to: {e}")

    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt.get("global_step", 0)
    best_score_pixel = ckpt.get("best_score_pixel", None)
    best_score_pear = ckpt.get("best_score_pear", None)

    return start_epoch, global_step, best_score_pixel, best_score_pear




def write2Yaml(data, save_path="test.yaml"):
    with open(save_path, "w") as f:
        yaml.dump(data, f)
