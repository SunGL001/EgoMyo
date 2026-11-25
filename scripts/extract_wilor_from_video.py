#!/usr/bin/env python3
"""Extract cached WiLoR features from raw videos using official WiLoR inference crops.

The frame mapping is intentionally explicit: this script never aligns video to
labels or uses pose annotations to locate a hand.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True,
                        help='Model-ready data root with membership.json')
    parser.add_argument('--frame-map', type=Path, required=True,
                        help='Reviewed JSON mapping each trial to video frames; see docs')
    parser.add_argument('--wilor-root', type=Path, required=True,
                        help='Official WiLoR repository checkout')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Official WiLoR checkpoint')
    parser.add_argument('--config', type=Path, required=True,
                        help='Official WiLoR model_config.yaml')
    parser.add_argument('--detector', type=Path, required=True,
                        help='Official WiLoR detector.pt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--confidence', type=float, default=0.3)
    parser.add_argument('--rescale-factor', type=float, default=2.0,
                        help='Match official WiLoR demo default')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--export-baseline-crops', action='store_true',
                        help='Also save native RGB batch fields and DINO-ready crops (additional disk space)')
    args = parser.parse_args()

    wilor_root = args.wilor_root.resolve()
    sys.path.insert(0, str(wilor_root))
    from ultralytics import YOLO
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.models import load_wilor

    device = torch.device(args.device)
    model, model_cfg = load_wilor(str(args.checkpoint), str(args.config))
    model = model.to(device).eval()
    model.requires_grad_(False)
    detector = YOLO(str(args.detector)).to(device)

    membership = json.loads((args.data / 'membership.json').read_text())
    frame_map = json.loads(args.frame_map.read_text())
    trial_map = frame_map.get('trials')
    if not isinstance(trial_map, dict):
        raise ValueError('frame-map JSON must contain a trials object')

    output_root = args.data / 'features' / 'wilor'
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        'method': 'official WiLoR detector + ViTDetDataset crop + frozen WiLoR backbone',
        'frame_map': str(args.frame_map.resolve()),
        'wilor_checkpoint': str(args.checkpoint.resolve()),
        'wilor_checkpoint_sha256': sha256(args.checkpoint),
        'wilor_config': str(args.config.resolve()),
        'detector_checkpoint': str(args.detector.resolve()),
        'detector_checkpoint_sha256': sha256(args.detector),
        'confidence': args.confidence,
        'rescale_factor': args.rescale_factor,
        'feature_shape_per_frame': [13, 1280],
        'pooling': 'adaptive average pool 4x3 spatial tokens + global spatial mean',
        'frame_indices': 'zero-based decoded video frame indices supplied explicitly',
        'baseline_crops': args.export_baseline_crops,
    }
    trial_rows = {row['trial_id']: row for row in membership['trials']}
    if set(trial_map) != set(trial_rows):
        raise ValueError('frame-map trial IDs must exactly match membership.json')

    for trial_id, member in trial_rows.items():
        spec = trial_map[trial_id]
        if not {'video', 'frame_indices', 'side'}.issubset(spec):
            raise ValueError(f'{trial_id}: requires video, frame_indices and side')
        if spec['side'] not in ('L', 'R'):
            raise ValueError(f'{trial_id}: side must be L or R')
        indices = spec['frame_indices']
        if (not isinstance(indices, list) or len(indices) != member['frames'] or
                any(not isinstance(index, int) or index < 0 for index in indices)):
            raise ValueError(f'{trial_id}: frame_indices must contain exactly membership.frames nonnegative integers')
        if len(set(indices)) != len(indices):
            raise ValueError(f'{trial_id}: duplicate frame indices are not allowed')

        destination = output_root / f'{trial_id}.npy'
        if destination.exists() and not args.overwrite:
            print(f'Skip existing {destination}; use --overwrite to replace', flush=True)
            continue
        video_path = Path(spec['video'])
        if not video_path.is_absolute():
            video_path = args.frame_map.parent / video_path
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f'{trial_id}: cannot open video {video_path}')

        maps = []
        crop_batch = []
        stored_crops = []
        right_class = 1 if spec['side'] == 'R' else 0
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f'{trial_id}: cannot decode frame {frame_index} from {video_path}')
            detections = detector(image, conf=args.confidence, verbose=False)[0]
            boxes, handed = [], []
            for detection in detections:
                values = detection.boxes.data.detach().cpu().numpy().reshape(-1, 6)
                for value in values:
                    if int(value[5]) == right_class:
                        boxes.append(value[:4])
                        handed.append(right_class)
            if len(boxes) != 1:
                raise RuntimeError(
                    f'{trial_id} frame {frame_index}: expected exactly one {spec["side"]}-hand '
                    f'detection above confidence {args.confidence}, found {len(boxes)}; '
                    'review the video/frame mapping or use a reviewed single-hand crop manifest')
            crop_data = ViTDetDataset(
                model_cfg, image, np.asarray(boxes, dtype=np.float32),
                np.asarray(handed, dtype=np.float32),
                rescale_factor=args.rescale_factor,
            )[0]
            crop_batch.append(crop_data['img'])
            if args.export_baseline_crops:
                stored_crops.append({name: np.asarray(crop_data[name]) for name in
                                     ('img', 'right', 'box_center', 'box_size', 'img_size')})
            if len(crop_batch) == args.batch_size:
                maps.extend(_backbone_features(model, crop_batch, device))
                crop_batch = []
        capture.release()
        if crop_batch:
            maps.extend(_backbone_features(model, crop_batch, device))

        feature_rows = []
        for feature in maps:
            tokens = torch.cat((
                F.adaptive_avg_pool2d(feature, (4, 3)).flatten(2),
                feature.mean((2, 3))[:, :, None],
            ), dim=2).transpose(1, 2)
            feature_rows.append(tokens[0].half().cpu().numpy())
        values = np.stack(feature_rows)
        if values.shape != (member['frames'], 13, 1280) or not np.isfinite(values).all():
            raise ValueError(f'{trial_id}: invalid extracted feature array {values.shape}')
        temporary = destination.with_suffix('.npy.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, values, allow_pickle=False)
        temporary.replace(destination)
        if args.export_baseline_crops:
            crop_root = args.data / 'crops' / trial_id
            crop_root.mkdir(parents=True, exist_ok=True)
            fields = {name: np.stack([row[name] for row in stored_crops]) for name in stored_crops[0]}
            np.savez_compressed(crop_root / 'rgb.npz', **fields)
            mean = torch.tensor(model_cfg.MODEL.IMAGE_MEAN).reshape(1, 3, 1, 1)
            std = torch.tensor(model_cfg.MODEL.IMAGE_STD).reshape(1, 3, 1, 1)
            images = torch.from_numpy(fields['img']).float() * std + mean
            images = F.interpolate(images, (224, 224), mode='bilinear', align_corners=False)
            dino_mean = torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1)
            dino_std = torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1)
            np.save(crop_root / 'dino.npy', ((images - dino_mean) / dino_std).numpy().astype(np.float16))
        print(f'{trial_id}: {values.shape} -> {destination}', flush=True)

    (output_root / 'extraction.json').write_text(json.dumps(metadata, indent=2) + '\n')


def _backbone_features(model, crops, device):
    images = torch.stack(crops).to(device)
    with torch.inference_mode():
        _, _, _, feature = model.backbone(images[:, :, :, 32:-32])
    if feature.ndim != 4 or feature.shape[1] != 1280:
        raise ValueError(f'Unexpected WiLoR backbone feature shape: {tuple(feature.shape)}')
    return list(feature.float().cpu().split(1, dim=0))


if __name__ == '__main__':
    main()
