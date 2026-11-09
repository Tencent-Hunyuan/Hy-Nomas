#!/usr/bin/env python3
"""Infer Hy-Nomas IDM camera action probabilities for an MP4 or directory of MP4s."""
import argparse
from pathlib import Path

from idm.clip_head import ClipHead, DEFAULT_CLIP_HEAD
from idm.postprocessing import DEFAULT_SHAKE_THRESHOLD, DEFAULT_ROTATION_THRESHOLD
from idm.runtime import ROOT, infer_video, load_model, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output JSON file')
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights/idm.pt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument('--sampling', choices=('windows', 'uniform'), default='windows',
                        help='windows: 6-second windows at 24 FPS; uniform: 144 frames over the entire video')
    parser.add_argument('--clip-head', type=Path, nargs='?', const=DEFAULT_CLIP_HEAD,
                        help='Use the bundled or specified clip-level head; requires --sampling uniform')
    parser.add_argument('--hud', action='store_true', help='Also export a prediction HUD MP4 for each video')
    parser.add_argument('--shake-threshold', type=float, default=DEFAULT_SHAKE_THRESHOLD,
                        help='SHAKE decision threshold for JSON and HUD (default: 0.7)')
    parser.add_argument('--rotation-threshold', type=float, default=DEFAULT_ROTATION_THRESHOLD,
                        help='Exclusive rotation decision threshold for JSON and HUD (default: 0.5)')
    parser.add_argument('--hud-threshold', type=float, default=0.5,
                        help='HUD highlight threshold for other keys and zoom (default: 0.5)')
    args = parser.parse_args()
    files = sorted(p for p in args.video.iterdir() if p.suffix.lower() == '.mp4') \
        if args.video.is_dir() else [args.video]
    if not files or any(not p.is_file() for p in files):
        parser.error('No readable input videos')
    if args.output.resolve() in {path.resolve() for path in files}:
        parser.error('JSON output must differ from every input video')
    if args.device == 'cpu' and args.precision != 'fp32':
        parser.error('--device cpu requires --precision fp32')
    if args.clip_head is not None and args.sampling != 'uniform':
        parser.error('--clip-head requires --sampling uniform')
    if not 0 <= args.hud_threshold <= 1:
        parser.error('--hud-threshold must be between 0 and 1')
    if not 0 <= args.shake_threshold <= 1:
        parser.error('--shake-threshold must be between 0 and 1')
    if not 0 <= args.rotation_threshold <= 1:
        parser.error('--rotation-threshold must be between 0 and 1')
    if args.hud:
        from idm.hud import ffmpeg_executable, render_video
        ffmpeg_executable()
        outputs = [args.output.parent / f'{path.stem}_hud.mp4' for path in files]
        if any(out.resolve() in {p.resolve() for p in files} for out in outputs):
            parser.error('HUD output would overwrite an input video; use another output directory')
        if len({p.resolve() for p in outputs}) != len(outputs):
            parser.error('Input stems produce duplicate HUD filenames')
        if args.output.resolve() in {p.resolve() for p in outputs}:
            parser.error('JSON output must differ from HUD video outputs')
    head = ClipHead(args.clip_head) if args.clip_head is not None else None
    if head is not None:
        head.verify_backbone(args.weights)
    model, metadata = load_model(args.weights, args.device)
    if head is not None:
        metadata['clip_head'] = dict(head.metadata)
    result = {'model': metadata, 'samples': []}
    for path in files:
        sample = infer_video(model, path, precision=args.precision, sampling=args.sampling,
                             shake_threshold=args.shake_threshold,
                             rotation_threshold=args.rotation_threshold)
        if head is not None:
            sample = head.apply(sample)
        result['samples'].append(sample)
        write_json(args.output, result)
        if args.hud:
            sample['hud'] = render_video(path, args.output.parent / f'{path.stem}_hud.mp4',
                                         sample, threshold=args.hud_threshold)
            write_json(args.output, result)
        print(f"{len(result['samples'])}/{len(files)} {path.name}", flush=True)


if __name__ == '__main__':
    main()
