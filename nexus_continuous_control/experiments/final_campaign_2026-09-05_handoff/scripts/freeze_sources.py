"""Create new, LF-normalized source snapshots without checking out or editing Git.

The core snapshot is the actual working files, including uncommitted fixes. RGB
is exported from the audited origin/main object. No checkpoints are copied.
"""
from __future__ import annotations
import argparse, hashlib, io, json, subprocess, tarfile
from pathlib import Path
from common import ROOT, write_json, atomic

DIRECTORIES = {'nexus_continuous', 'configs', 'tests', 'tools'}
SUFFIXES = {'.py', '.yaml', '.yml', '.toml'}
RGB_COMMIT = '7557d5d9b9c75fbe93091ead6ae525a1c377cdf6'

def selected(name: str) -> bool:
    p = Path(name)
    return (p.name == 'pyproject.toml' or
            (p.parts[0] in DIRECTORIES and p.suffix in SUFFIXES)) and '__pycache__' not in p.parts

def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(['git', *args], cwd=repo, text=True).strip()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--package', default=str(ROOT))
    a = ap.parse_args()
    repo = Path(a.repo).resolve(); package = Path(a.package).resolve()
    source = repo / 'nexus_continuous_control'; target = package / 'sources'
    if target.exists(): raise FileExistsError(f'Already frozen: {target}')
    if git(repo, 'rev-parse', 'origin/main') != RGB_COMMIT:
        raise RuntimeError('origin/main changed; inspect changes before updating the frozen RGB source')
    contract=json.loads((package/'audit/source_contract.json').read_text())
    if git(repo,'rev-parse','HEAD')!=contract['core_base_head']:
        raise RuntimeError('Working branch HEAD changed since the live audit. Review before freezing a revised campaign.')
    for rel,expected in contract['sha256_lf'].items():
        q=source/rel
        if not q.is_file() or hashlib.sha256(q.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:
            raise RuntimeError(f'Critical working source differs from the audit: {rel}. Preserve it; review the diff, not a blind pull.')
    manifest = {}; target.mkdir(parents=True)
    def put(kind: str, name: str, data: bytes) -> None:
        data = data.replace(b'\r\n', b'\n')
        rel = f'{kind}/{name}'
        atomic(target / rel, data)
        manifest[rel] = hashlib.sha256(data).hexdigest()
    for folder in sorted(DIRECTORIES):
        for p in sorted((source / folder).rglob('*')):
            name = p.relative_to(source).as_posix()
            if p.is_file() and selected(name): put('core', name, p.read_bytes())
    put('core', 'pyproject.toml', (source / 'pyproject.toml').read_bytes())
    data = subprocess.check_output(['git', 'archive', f'{RGB_COMMIT}:nexus_continuous_control'], cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for member in tf:
            if member.isfile() and selected(member.name):
                if Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                    raise ValueError('Unsafe archive path')
                if member.size > 5_000_000: raise ValueError('Unexpectedly large source file')
                put('rgb', member.name, tf.extractfile(member).read())
    write_json(package / 'source_manifest.json', manifest)
    write_json(package / 'audit/source_origins.json', {
        'core': {'kind': 'working-tree snapshot, NOT a clean commit', 'repo': str(repo),
                 'base_head': git(repo, 'rev-parse', 'HEAD'),
                 'semantic_diff': git(repo, 'diff', '--ignore-space-at-eol', '--stat')},
        'rgb': {'kind': 'Git object export', 'commit': RGB_COMMIT},
        'normalization': 'CRLF -> LF only; original files unchanged',
        'files': len(manifest), 'checkpoints_copied': 0})
    print(json.dumps({'files': len(manifest), 'path': str(target)}, indent=2))

if __name__ == '__main__': main()
