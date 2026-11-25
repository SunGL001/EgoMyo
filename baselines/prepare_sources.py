"""Download byte-pinned external source files, never checkpoints or MANO assets."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

EGOEMG_COMMIT = 'e2d1fa5ed793c67abc23dbfa2d8fe309aaf61d7b'
META_COMMIT = '5f6f62b1a0a08426adffe55900842e75a8adb38c'
META_SHA256 = '420ad7eab0ff9aaa695fea02e2371467150f38e3dedab0c62436eb02ebe814d0'
FILES = {
    'tds.py': ('egoemg/models/featurizers/tds.py', '560b0cb7f17024e55e22eecb258cd56655ed3581b2a2f66889476aaeaaf8b37d'),
    'transformer.py': ('egoemg/models/decoders/transformer.py', '404cdd80951869517bebc7996f9c42a77214c0bf33762fe4f405abe7c1c4dcaa'),
    'pooling.py': ('egoemg/models/modules/_pooling.py', 'f5ff23e6878b4f50e6eabc586b77df33c844aa4377dc815aaee8e4f65d61cb06'),
    'neuropose.py': ('egoemg/models/featurizers/neuropose.py', '8ffacb25a68d2693e41934a4fcc1d5f2c8cd27bcf9452ed5a6d69daaf861eac4'),
}


def download(url, destination, expected=None):
    with urlopen(url, timeout=60) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if expected and digest != expected:
        raise ValueError('External source hash mismatch: ' + url)
    destination.write_bytes(content)
    return dict(url=url, sha256=digest, file=destination.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--accept-upstream-licenses', action='store_true', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    base = f'https://raw.githubusercontent.com/zhenqis123/EgoEMG/{EGOEMG_COMMIT}/'
    for name, (relative, expected) in FILES.items():
        records.append(download(base + relative, args.out / name, expected))
    for name in ('LICENSE', 'THIRD_PARTY_NOTICES.md'):
        records.append(download(base + name, args.out / ('EgoEMG-' + name)))
    base = f'https://raw.githubusercontent.com/facebookresearch/emg2pose/{META_COMMIT}/'
    for relative, name in (('emg2pose/networks.py', 'meta_networks.py'), ('LICENSE', 'Meta-LICENSE')):
        records.append(download(base + relative, args.out / name,
                                META_SHA256 if name == 'meta_networks.py' else None))
    (args.out / 'receipt.json').write_text(json.dumps(dict(
        egoemg_commit=EGOEMG_COMMIT, meta_commit=META_COMMIT, files=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
