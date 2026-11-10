"""Export one pilot's allowlisted artifacts, excluding indexes and secret files."""
from __future__ import annotations
import argparse
from pathlib import Path
import zipfile


def export_results(root: Path, run_name: str, output: Path):
    if not run_name or Path(run_name).name != run_name or run_name in ('.', '..'):
        raise ValueError('run_name must be a single directory name')
    roots = [root / run_name] + [root / f'{run_name}_{suffix}' for suffix in
              ('reference_metrics.json', 'retrieval_review.json', 'review', 'progress', 'diagnostics.json')]
    if not (root / run_name / 'manifest.json').is_file():
        raise ValueError('Run manifest missing')
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in roots:
            paths = sorted(source.rglob('*')) if source.is_dir() else [source]
            if source.is_symlink():
                raise ValueError('Export must not follow symlinks')
            for path in paths:
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError('Export must not follow symlinks/outside files')
                if (not path.is_file() or path.suffix in ('.sqlite', '.db', '.part')
                        or path.name.startswith('.') or path.resolve() == output.resolve()):
                    continue
                archive.write(path, path.relative_to(root))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(export_results(a.root, a.run_name, a.output))


if __name__ == '__main__':
    main()
