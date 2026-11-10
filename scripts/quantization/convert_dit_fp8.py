"""Stream a floating DiT safetensors checkpoint into offline block-scaled FP8 shards."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hymm.models.dit.modules.fp8_linear import FORMAT, SELECT, quantize_weight
from hymm.models.dit.modules.int8_stored_bf16_linear import _sha256
from scripts.quantization.convert_dit_w8_bf16 import normalize_key


def convert(source, destination, device='cuda:0'):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name+'.tmp-', dir=destination.parent))
    stat = source.stat()
    manifest = dict(format=FORMAT, layers={}, shards=[], source=dict(path=str(source), bytes=stat.st_size,
                    sha256=_sha256(source)), activation='dynamic-per-token-128', weight_block=[128,128], scale='power-of-two-fp32')
    try:
        batch, size, seen = {}, 0, set()
        def flush():
            nonlocal batch, size
            if not batch:
                return
            path = temporary / f'model-{len(manifest["shards"])+1:05d}.safetensors'
            save_file(batch, str(path))
            manifest['shards'].append(dict(file=path.name, bytes=path.stat().st_size, sha256=_sha256(path)))
            batch, size = {}, 0
        with safe_open(str(source), framework='pt', device='cpu') as reader:
            for original in reader.keys():
                key = normalize_key(original)
                if key in seen:
                    raise ValueError(f'Duplicate normalized key: {key}')
                seen.add(key)
                w = reader.get_tensor(original)
                if not w.is_floating_point() or not torch.isfinite(w).all():
                    raise ValueError(f'Expected finite floating checkpoint tensor: {key}')
                name = key.removesuffix('.weight')
                if key.endswith('.weight') and SELECT.fullmatch(name):
                    q, s = quantize_weight(w.to(device))
                    batch[name+'.weight_q'], batch[name+'.weight_scale'] = q.cpu(), s.cpu()
                    manifest['layers'][name] = list(w.shape)
                else:
                    batch[key] = w.to(torch.bfloat16).contiguous()
                size += w.numel() * w.element_size()
                if size >= 512 << 20:
                    flush()
            flush()
        if not manifest['layers']:
            raise ValueError('No supported DiT projection weights found')
        if source.stat().st_mtime_ns != stat.st_mtime_ns or source.stat().st_size != stat.st_size:
            raise RuntimeError('Source changed during conversion')
        (temporary/'fp8_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = convert(args.source, args.output_dir, args.device)
    print(json.dumps(dict(layers=len(result['layers']), bytes=sum(s['bytes'] for s in result['shards']))))
