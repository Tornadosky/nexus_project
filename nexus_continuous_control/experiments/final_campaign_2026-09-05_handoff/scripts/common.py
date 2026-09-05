"""Non-destructive utilities. Only load pickle checkpoints you trust."""
from __future__ import annotations
import hashlib, json, os, pickle, sys, tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
def atomic(path: Path, data: bytes, replace: bool = False) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f'Refusing to overwrite {path}')
    fd, temp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        if replace: os.replace(temp, path)
        else: os.link(temp, path); os.unlink(temp)
    finally:
        if os.path.exists(temp): os.unlink(temp)

def jsonable(x: Any) -> Any:
    if isinstance(x, dict): return {str(k): jsonable(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [jsonable(v) for v in x]
    if hasattr(x, 'tolist'): return x.tolist()
    if isinstance(x, Path): return str(x)
    return x

def write_json(path: Path, x: Any, replace: bool = False) -> None:
    atomic(path, json.dumps(jsonable(x), indent=2, allow_nan=False).encode(), replace)

def write_pickle(path: Path, x: Any) -> None:
    atomic(path, pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL))

def add_repo(repo: str) -> Path:
    if (ROOT/'INSTALLING').exists():
        raise RuntimeError('Installation incomplete. Do not run worker scripts until verify_installation.py succeeds.')
    p=Path(repo).resolve()
    if not (p/'nexus_continuous').is_dir():
        raise FileNotFoundError(f'{p} must be the nexus_continuous_control package root')
    sys.path.insert(0,str(p)); sys.path.insert(0,str(p/'tools'))
    return p

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old)!=1:
        raise RuntimeError('Source snapshot differs: instrumentation anchor is not unique. '
                           'Stop before spending GPU time; do not guess a replacement.')
    return source.replace(old,new,1)
