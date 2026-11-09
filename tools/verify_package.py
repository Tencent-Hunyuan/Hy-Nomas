#!/usr/bin/env python3
"""Verify every file listed in the package manifest."""
import hashlib
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--github-readme', action='store_true',
                        help='Also require six accessible inline GitHub video attachments')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.github_readme:
        from set_video_urls import validate_readmes
        validate_readmes(root, online=True)
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    for entry in manifest['files']:
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Manifest path escapes the package')
        if path.stat().st_size != entry['bytes']:
            raise ValueError(f"Size mismatch: {entry['path']}")
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != entry['sha256']:
            raise ValueError(f"Checksum mismatch: {entry['path']}")
    print(f"Verified {len(manifest['files'])} files.")


if __name__ == '__main__':
    main()
