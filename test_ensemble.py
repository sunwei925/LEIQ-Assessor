import argparse
import csv
import json
import os
import re
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
from open_clip import create_model
from PIL import Image
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class SigLIP2_384_Image(nn.Module):
    def __init__(self):
        super().__init__()
        model = create_model('ViT-B-16-SigLIP2-384', pretrained=False)
        self.feature_extraction = model.visual
        self.quality = nn.Sequential(
            nn.Linear(768, 128),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = torch.flatten(x, 1)
        x = self.quality(x)
        return x.squeeze(1)


class SigLIP2_ViTG_384_Image_MT(nn.Module):
    TASK_NAMES = ['mos', 'light', 'color', 'noise',
                  'exposure', 'nature', 'content_recovery']

    def __init__(self):
        super().__init__()
        model = create_model('ViT-SO400M-14-SigLIP2-378', pretrained=False)
        self.feature_extraction = model.visual
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(1152, 128),
                nn.Linear(128, 1),
            )
            for name in self.TASK_NAMES
        })

    def forward(self, x):
        feat = self.feature_extraction(x)
        feat = torch.flatten(feat, 1)
        return {name: head(feat).squeeze(1) for name, head in self.heads.items()}


class MLETestDataset(Dataset):
    def __init__(self, test_dir, metadata_csv, transform):
        self.test_dir = test_dir
        self.transform = transform
        items = []
        if metadata_csv and os.path.exists(metadata_csv):
            with open(metadata_csv, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sample_id = (row.get('id') or '').strip()
                    file_name = (row.get('file_name') or '').strip()
                    if not sample_id or not file_name:
                        continue
                    items.append((sample_id, file_name))
        else:
            for name in sorted(os.listdir(test_dir)):
                if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    items.append((os.path.splitext(name)[0], name))
        if not items:
            raise RuntimeError('No test images found in: ' + test_dir)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample_id, file_name = self.items[idx]
        path = os.path.join(self.test_dir, file_name)
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image, sample_id


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logistic_part = 1 + np.exp(-(X - bayta3) / np.abs(bayta4))
    return bayta2 + (bayta1 - bayta2) / logistic_part


def fit_mapping_params_st(mat_path):
    data = scio.loadmat(mat_path)
    y_output = data['y_output'].flatten().astype(np.float64)
    label = data['label'].flatten().astype(np.float64)
    beta0 = [np.max(label), np.min(label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, label, p0=beta0, maxfev=100_000_000)
    return popt


def fit_mapping_params_mt(mat_path):
    data = scio.loadmat(mat_path)
    y_output = data['mos_pred'].flatten().astype(np.float64)
    y_label = data['mos_label'].flatten().astype(np.float64)
    beta0 = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta0, maxfev=100_000_000)
    return popt


def load_checkpoint_state_dict(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    return {k.replace('module.', '', 1): v for k, v in state.items()}


def discover_splits(ckpt_dir, model_name, n_splits=10):
    pairs = {}
    pat = re.compile(
        r'^' + re.escape(model_name) + r'_MLE_v(\d+)_ep\d+_SRCC[\d.]+\.pth$')
    for fname in os.listdir(ckpt_dir):
        m = pat.match(fname)
        if m:
            split_idx = int(m.group(1))
            pth_path = os.path.join(ckpt_dir, fname)
            mat_path = pth_path.replace('.pth', '.mat')
            if os.path.exists(mat_path):
                pairs[split_idx] = (pth_path, mat_path)
    found = sorted(pairs.keys())
    print('[Discover] found %d splits in %s: %s' % (len(found), ckpt_dir, found))
    for i in sorted(pairs):
        print('  v%d: %s' % (i, os.path.basename(pairs[i][0])))
    missing = [i for i in range(n_splits) if i not in pairs]
    if missing:
        print('[Warning] missing splits: %s' % missing)
    return pairs


@torch.no_grad()
def run_st_split(pth_path, mat_path, loader, device):
    model = SigLIP2_384_Image()
    sd = load_checkpoint_state_dict(pth_path, device)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    mapping_popt = fit_mapping_params_st(mat_path)
    preds = {}
    for images, sample_ids in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images).detach().cpu().tolist()
        for sid, raw in zip(sample_ids, outputs):
            preds[sid] = float(logistic_func(np.array([raw]), *mapping_popt)[0])
    del model
    torch.cuda.empty_cache()
    return preds


@torch.no_grad()
def run_mt_split(pth_path, mat_path, loader, device):
    model = SigLIP2_ViTG_384_Image_MT()
    sd = load_checkpoint_state_dict(pth_path, device)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    mapping_popt = fit_mapping_params_mt(mat_path)
    preds = {}
    for images, sample_ids in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        mos_raw = outputs['mos'].detach().cpu().tolist()
        for sid, raw in zip(sample_ids, mos_raw):
            preds[sid] = float(logistic_func(np.array([raw]), *mapping_popt)[0])
    del model
    torch.cuda.empty_cache()
    return preds


def run_model_splits(name, ckpt_dir, ckpt_name, loader, device, run_fn, n_splits):
    print('\n' + '=' * 60)
    print('  ' + name)
    print('  ckpt_dir: %s  ckpt_name: %s' % (ckpt_dir, ckpt_name))
    print('=' * 60)

    split_pairs = discover_splits(ckpt_dir, ckpt_name, n_splits)
    if not split_pairs:
        raise RuntimeError(
            'No checkpoint pairs found for %s in %s' % (ckpt_name, ckpt_dir))

    all_preds = {}
    for si in sorted(split_pairs):
        pth, mat = split_pairs[si]
        print('\n' + '-' * 60)
        print('Split %d: %s' % (si, os.path.basename(pth)))
        print('-' * 60)
        all_preds[si] = run_fn(pth, mat, loader, device)

    sample_ids = list(next(iter(all_preds.values())).keys())
    print('\n  Averaging %d splits for %d samples' % (len(all_preds), len(sample_ids)))

    averaged = {}
    for sid in sample_ids:
        vals = [all_preds[i][sid] for i in sorted(all_preds) if sid in all_preds[i]]
        averaged[sid] = float(np.mean(vals))
    return averaged


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]

    tf_st = transforms.Compose([
        transforms.Resize(432),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    tf_mt = transforms.Compose([
        transforms.Resize(432),
        transforms.CenterCrop(378),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    ds_st = MLETestDataset(args.test_dir, args.metadata_csv, tf_st)
    ds_mt = MLETestDataset(args.test_dir, args.metadata_csv, tf_mt)
    loader_st = DataLoader(ds_st, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    loader_mt = DataLoader(ds_mt, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    canonical_ids = [sid for sid, _ in ds_st.items]
    print('Test images: %d' % len(ds_st))

    st_avg = run_model_splits(
        'SigLIP2_384_Image (single-task)',
        args.st_ckpt_dir, args.st_ckpt_name,
        loader_st, device, run_st_split, args.n_splits)

    mt_avg = run_model_splits(
        'SigLIP2_ViTG_384_Image_MT (multi-task)',
        args.mt_ckpt_dir, args.mt_ckpt_name,
        loader_mt, device, run_mt_split, args.n_splits)

    print('\n' + '=' * 60)
    print('  Ensemble: averaging single-task and multi-task MOS')
    print('  Samples: %d' % len(canonical_ids))
    print('=' * 60)

    results = []
    for sid in canonical_ids:
        v_st = st_avg.get(sid, float('nan'))
        v_mt = mt_avg.get(sid, float('nan'))
        avg_mos = float(np.nanmean([v_st, v_mt]))
        if args.clamp_to_range:
            avg_mos = max(args.min_mos, min(args.max_mos, avg_mos))
        if args.round is not None:
            avg_mos = round(avg_mos, args.round)
        results.append({'id': sid, 'mos': avg_mos})

    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('\n[Output] wrote %d samples to %s' % (len(results), args.output_json))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLE ensemble test')

    parser.add_argument('--st_ckpt_dir', type=str, default='SigLIP2_384_Image')
    parser.add_argument('--st_ckpt_name', type=str, default='SigLIP2_384_Image')
    parser.add_argument('--mt_ckpt_dir', type=str,
                        default='SigLIP2_ViTG_384_Image_10splits')
    parser.add_argument('--mt_ckpt_name', type=str,
                        default='SigLIP2_ViTG_384_Image_MT')
    parser.add_argument('--n_splits', type=int, default=10)

    parser.add_argument('--test_dir', type=str, default='test_images')
    parser.add_argument('--metadata_csv', type=str,
                        default='test_images/metadata.csv')
    parser.add_argument('--output_json', type=str, default='result.json')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--round', type=int, default=4)
    parser.add_argument('--clamp_to_range', action='store_true')
    parser.add_argument('--min_mos', type=float, default=0.0)
    parser.add_argument('--max_mos', type=float, default=10.0)

    main(parser.parse_args())
