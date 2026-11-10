"""Package the allowlisted generating source for a Colab update without Git push."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.provenance import source_hashes
from src.experiments.artifacts import fingerprint, file_hash


def package(root, output):
    hashes = source_hashes(root)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, expected in hashes.items():
            if file_hash(root / name) != expected:
                raise ValueError('Source changed during packaging')
            z.write(root / name, name)
        z.writestr('source_manifest.json', json.dumps({'source_sha256': fingerprint(hashes), 'files': hashes}))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(package(Path(__file__).resolve().parents[1], a.output))
