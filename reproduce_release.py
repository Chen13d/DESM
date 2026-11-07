"""Reproduce released reconstructions from acquired intensity and ASC files.

Run from the DESM repository root after extracting both resource archives.
IL reruns lifetime clustering and IO-prior preprocessing. This entry point does not run error heads.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import torch
import torch.nn.functional as F
import tifffile


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_model(path, device):
    model = torch.load(path, map_location='cpu', weights_only=False)
    model = model.to(device).eval().requires_grad_(False)
    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            layer.train()
            layer.track_running_stats = False
    return model


@torch.inference_mode()
def predict(model, image, lifetime=None, multiplier=1.):
    device = next(model.parameters()).device
    # Match the published CPU normalization before moving input to the GPU.
    x = torch.from_numpy(np.asarray(image, np.float32)[None, None] * multiplier)
    std = x.std()
    if not torch.isfinite(std) or std <= 0:
        raise ValueError('Input must have nonzero finite intensity variation')
    x = x / std
    if lifetime is not None:
        lf = torch.from_numpy(np.asarray(lifetime, np.float32)[None, None])
        x = torch.cat([x, lf / lf.std().clamp_min(1e-8)], 1)
    h, w = x.shape[-2:]
    x = F.pad(x.to(device), (0, (-w) % 16, 0, (-h) % 16), mode='reflect')
    return (model(x)[0, :, :h, :w] * std).cpu().numpy() / multiplier


def run(dataset=None, root='.', output=None, limit=None, device=None, check=True):
    root = Path(root).resolve()
    data = root / 'visualization/Experimental data'
    records = json.loads((data / 'dataset_manifest.json').read_text(encoding='utf-8'))
    models = {m['id']: m for m in json.loads((root / 'Trained_models/model_manifest.json').read_text(encoding='utf-8'))}
    if dataset:
        records = [r for r in records if r['folder'] == dataset or r['id'] == dataset]
    if not records:
        raise ValueError('No matching dataset. Use --list to show names.')
    if limit:
        records = records[:limit]
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    output = Path(output) if output else root / 'reproduced'
    model, last, report = None, None, []
    raw_il = {}
    for r in records:
        if r['model_id'] != last:
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
            path = root / r['model_file']
            if check and digest(path) != r['model_sha256']:
                raise ValueError('Wrong model version: ' + str(path))
            model = None if 'lifetime_labels' in r else load_model(path, device)
            last = r['model_id']
        if 'lifetime_labels' in r:
            from visualization_DESM_IL import DESM_IL_inference
            folder = r['folder']
            raw_dest = output / folder / 'raw_pipeline'
            if folder not in raw_il:
                prior = models[r['prior_model_id']]
                if check and digest(root / prior['file']) != prior['sha256']:
                    raise ValueError('Wrong IO prior model')
                resampling = 'resize_ratio' in r.get('preprocessing', {})
                steps = DESM_IL_inference(str(data / folder / 'Intensity'), str(data / folder / 'ASC'),
                    str(root / r['model_file']), str(root / prior['file']), device=device,
                    save_dir=str(raw_dest), tm_max=5500, org_list=r['channels'], full_field=True,
                    normalize_to_255=True, resample_input=resampling, crop_to_multiple=not resampling, save_float=True)
                raw_il[folder] = {step['source']: step['index'] for step in steps}
            i = raw_il[folder][Path(r['raw_source']).name]
            x = tifffile.imread(raw_dest / f'{i}_input_float.tif')
            pred = np.stack([tifffile.imread(raw_dest / f'{i}_{c}_SR_float.tif') for c in range(len(r['channels']))])
        else:
            x = tifffile.imread(data / r['raw_source'])
            if 'references' in r:
                from utils import resize
                x = x[1]
                x = resize(x, (int(x.shape[0] * 22.7 / 20), int(x.shape[1] * 22.7 / 20)))
            pred = predict(model, x, multiplier=r.get('input_multiplier', 1.))
        dest = output / r['folder'] / r['sample']
        dest.mkdir(parents=True, exist_ok=True)

        def save_channels(values, references, suffix):
            checks = []
            for c, ref in enumerate(references):
                tifffile.imwrite(dest / f'{c}_{suffix}.tif', values[c].astype(np.float32), imagej=True,
                                 metadata={'axes': 'YX', 'unit': 'um'}, resolution=(1000 / r['pixel_nm'],) * 2)
                if check:
                    truth = tifffile.imread(data / ref)
                    diff = np.abs(values[c] - truth)
                    checks.append({'channel': r['channels'][c], 'max_abs': float(diff.max()),
                                   'relative_max': float(diff.max() / max(float(np.abs(truth).max()), 1e-8))})
            return checks

        errors = save_channels(pred, r['outputs'], 'SR')
        comparison = []
        if 'io_comparison_outputs' in r:
            meta = models[r['io_comparison_model_id']]
            if check and digest(root / meta['file']) != meta['sha256']:
                raise ValueError('Wrong IO comparison model')
            io_model = load_model(root / meta['file'], device)
            comparison = save_channels(predict(io_model, x), r['io_comparison_outputs'], 'IO_comparison')
            del io_model
        report.append({'sample': r['id'], 'reconstruction_checks': errors, 'io_comparison_checks': comparison})
        print(r['id'], max([v['relative_max'] for v in errors], default=0), flush=True)
    (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset')
    ap.add_argument('--root', default='.')
    ap.add_argument('--output')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--device')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list:
        rows = json.loads((Path(a.root) / 'visualization/Experimental data/dataset_manifest.json').read_text(encoding='utf-8'))
        print('\n'.join(sorted({r['folder'] for r in rows})))
    else:
        run(a.dataset, a.root, a.output, a.limit, a.device)
