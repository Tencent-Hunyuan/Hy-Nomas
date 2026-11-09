#!/usr/bin/env python3
"""Apply the bundled clip-level head to saved uniform144 predictions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from idm.clip_head import ClipHead, DEFAULT_CLIP_HEAD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=DEFAULT_CLIP_HEAD)
    args = parser.parse_args()
    if args.output.resolve() in {args.predictions.resolve(), args.weights.resolve()}:
        parser.error('--output must differ from the input predictions and head weights')
    head = ClipHead(args.weights)
    payload = json.loads(args.predictions.read_text())
    payload['samples'] = head.apply_many(payload['samples'])
    payload['model'] = {**payload.get('model', {}), 'clip_head': head.metadata}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(f"Applied clip-level head to {len(payload['samples'])} videos")


if __name__ == '__main__':
    main()
