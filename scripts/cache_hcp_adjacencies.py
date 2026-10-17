"""Materialize validated averaged HCP connectomes for fast experiments."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import HCPMatlabConnectomeDataset

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--include-features', action='store_true')
    args = p.parse_args()
    ds = HCPMatlabConnectomeDataset(args.archive, positive_only=True)
    adjs = np.stack([ds[i]['adj'].numpy() for i in range(len(ds))]).astype(np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / 'averaged_positive_adjacencies.npy', adjs)
    if args.include_features:
        features = np.lib.format.open_memmap(args.output_dir / 'standardized_bold_features.npy', mode='w+', dtype=np.float32, shape=(len(ds), 264, 4800))
        for i in range(len(ds)):
            features[i] = ds[i]['x'].numpy()
        del features
    (args.output_dir / 'subjects.json').write_text(json.dumps(ds.subjects, indent=2))
    print(json.dumps({'subjects': len(ds), 'shape': list(adjs.shape), 'output_dir': str(args.output_dir)}))

if __name__ == '__main__':
    main()
