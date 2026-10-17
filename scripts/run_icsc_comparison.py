"""Run consensus community analysis on clean and frozen-blend connectomes."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.noise import add_connectome_noise
from src.clustering.modularity import iterative_consensus_spectral_clustering, compute_newman_modularity

def main():
    p=argparse.ArgumentParser(); p.add_argument('--clean',type=Path,required=True); p.add_argument('--denoised',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--alpha',type=float,default=.75); p.add_argument('--noise-fraction',type=float,default=.30); p.add_argument('--modules',type=int,default=7); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    clean=np.load(a.clean); den=np.load(a.denoised); noisy=np.stack([add_connectome_noise(clean[i],noise_fraction=a.noise_fraction,mode='positive',rng=np.random.default_rng(a.seed+50000+i)) for i in range(len(clean))]); blend=(1-a.alpha)*noisy+a.alpha*den; a.output_dir.mkdir(parents=True,exist_ok=True); results={}; partitions={}
    for name,graphs in [('clean',clean),('noisy',noisy),('blend',blend)]:
        labels,consensus=iterative_consensus_spectral_clustering(list(graphs),n_clusters=a.modules,n_iterations=1,threshold=.4,random_state=a.seed); partitions[name]=labels; results[name]={'modularity_q':float(compute_newman_modularity(consensus,labels)),'module_sizes':{str(int(k)):int(np.sum(labels==k)) for k in np.unique(labels)}}; np.save(a.output_dir/f'{name}_module_labels.npy',labels); np.save(a.output_dir/f'{name}_consensus.npy',consensus)
    results['agreement']={'clean_vs_noisy_ari':float(adjusted_rand_score(partitions['clean'],partitions['noisy'])),'clean_vs_blend_ari':float(adjusted_rand_score(partitions['clean'],partitions['blend']))}; (a.output_dir/'report.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
