"""Exact source/environment snapshots, independent of credentials and data files."""
from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
import platform
import sys
import zipfile

from src.experiments.artifacts import atomic_json, file_hash, fingerprint

FILES = ('source.zip', 'source_manifest.json', 'environment.freeze.json')


def source_files(root: Path):
    paths = [p for folder in ('src', 'scripts') for p in (root / folder).rglob('*.py')]
    paths += [p for p in (root / 'configs').rglob('*') if p.suffix in ('.yaml', '.json')]
    paths += [root / 'docs/evaluation_protocol.md', root / 'docs/annotation_guidelines.md']
    paths += list(root.glob('requirements*.txt'))
    paths += list((root / 'notebooks').glob('05*.ipynb'))
    return sorted(p for p in paths if p.is_file())


def source_hashes(root: Path):
    paths = source_files(root)
    if any(p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) for p in paths):
        raise ValueError('Generating source must not contain symlinks/outside files')
    return {str(p.relative_to(root)): file_hash(p) for p in paths}


def freeze_environment():
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'installed_distributions': sorted([
                {'name': d.metadata['Name'], 'version': d.version} for d in metadata.distributions()
                if d.metadata['Name']], key=lambda d: (d['name'].casefold(), d['version']))}


def save_reproduction(root: Path, output: Path, identity: dict):
    hashes = source_hashes(root)
    if fingerprint(hashes) != identity['code']['source_sha256']:
        raise ValueError('Source changed before it could be frozen')
    environment = freeze_environment()
    if fingerprint(environment) != identity['reproduction_environment_sha256']:
        raise ValueError('Environment changed before it could be frozen')
    atomic_json(output / 'source_manifest.json', {'source_sha256': fingerprint(hashes), 'files': hashes})
    atomic_json(output / 'environment.freeze.json', environment)
    with zipfile.ZipFile(output / 'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            data = (root / name).read_bytes()
            import hashlib
            if hashlib.sha256(data).hexdigest() != hashes[name]:
                raise ValueError('Source changed during snapshot')
            entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    return {name: file_hash(output / name) for name in FILES}


def validate_reproduction(directory: Path, manifest: dict):
    hashes = manifest.get('reproduction_files_sha256')
    if hashes is None:  # Historical artifacts predate source snapshots.
        return
    if set(hashes) != set(FILES) or any((directory / name).is_symlink() or
            file_hash(directory / name) != value for name, value in hashes.items()):
        raise ValueError('Generating source/environment snapshot changed or is missing')
    source = json.loads((directory / 'source_manifest.json').read_text())
    if (source['source_sha256'] != fingerprint(source['files']) or
            source['source_sha256'] != manifest['identity']['code']['source_sha256']):
        raise ValueError('Source snapshot identity mismatch')
    with zipfile.ZipFile(directory / 'source.zip') as archive:
        if len(archive.namelist()) != len(source['files']) or set(archive.namelist()) != set(source['files']):
            raise ValueError('Unexpected source archive contents')
        import hashlib
        for name, expected in source['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts or hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('Source archive hash/path mismatch')
    environment = json.loads((directory / 'environment.freeze.json').read_text())
    if fingerprint(environment) != manifest['identity']['reproduction_environment_sha256']:
        raise ValueError('Frozen environment identity mismatch')
