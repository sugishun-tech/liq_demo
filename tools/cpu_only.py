"""Check the active training environment without modifying it."""
from __future__ import annotations
import importlib.metadata
from bundle import BundleError

def require_cpu():
    import torch
    if torch.version.cuda is not None or getattr(torch.version, 'hip', None) is not None:
        raise BundleError('CPU版Torchのvenvを有効にしてください。このツールはGPU版を使用・削除しません。')
    forbidden = []
    for dist in importlib.metadata.distributions():
        name = (dist.metadata.get('Name') or '').lower().replace('_', '-')
        if name.startswith(('nvidia-', 'cupy', 'tensorrt')) or name in {'onnxruntime-gpu', 'triton', 'pytorch-triton', 'onnxruntime-directml'}:
            forbidden.append(name)
    if forbidden:
        raise BundleError('CPU専用ではないパッケージが環境にあります: ' + ', '.join(sorted(set(forbidden))) + '。CPU専用venvを選んでください。自動削除はしません。')
    return torch.__version__
