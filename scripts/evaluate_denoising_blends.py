"""Evaluate residual blends of noisy inputs and saved VGAE outputs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.noise import add_connectome_noise

def labels(m, k, seed):
    i,j=np.triu_indices(m.shape[1],1); v=m[:,i,j]; v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8); s=np.clip(v@v.T,0,1); np.fill_diagonal(s,0); return SpectralClustering(n_clusters=k,affinity='precomputed',assign_labels='kmeans',random_state=seed).fit_predict(s)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--clean',type=Path,required=True); p.add_argument('--denoised',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--noise-fraction',type=float,default=.30); p.add_argument('--seed',type=int,default=42); p.add_argument('--clusters',type=int,default=4); a=p.parse_args()
    clean=np.load(a.clean); den=np.load(a.denoised); rng=np.random.default_rng(a.seed+50000); noisy=np.stack([add_connectome_noise(x,noise_fraction=a.noise_fraction,mode='positive',rng=np.random.default_rng(a.seed+50000+i)) for i,x in enumerate(clean)]); target=labels(clean,a.clusters,a.seed); out={}
    for alpha in [0.0,.25,.5,.75,1.0]:
        blend=(1-alpha)*noisy+alpha*den; tri=np.triu_indices(blend.shape[1],1); pred=labels(blend,a.clusters,a.seed); out[str(alpha)]={'mse':float(np.mean((blend[:,tri[0],tri[1]]-clean[:,tri[0],tri[1]])**2)),'ari':float(adjusted_rand_score(target,pred)),'nmi':float(normalized_mutual_info_score(target,pred)),'cluster_sizes':{str(int(k)):int(np.sum(pred==k)) for k in np.unique(pred)}}
    a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/'report.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
