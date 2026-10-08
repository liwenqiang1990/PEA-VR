from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def derived_seed(seed: int, *parts) -> int:
    return int(canonical_hash([seed, *map(str, parts)])[:8], 16)


def implementation_hash() -> str:
    root = Path(__file__).parent
    return canonical_hash([(path.relative_to(root).as_posix(), sha256(path)) for path in sorted(root.rglob('*.py'))])


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def seed_all(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state() -> dict:
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available():
            raise ValueError('A CUDA training state requires CUDA for exact resume')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def device_for(name: str) -> torch.device:
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: cpu, cuda, auto')
    return device


def environment() -> dict:
    import scipy
    import sklearn
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'numpy': np.__version__, 'torch': torch.__version__, 'scipy': scipy.__version__,
            'sklearn': sklearn.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None}
