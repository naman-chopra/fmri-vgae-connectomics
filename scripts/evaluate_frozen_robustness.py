"""Evaluate the frozen bounded VGAE over noise levels and random seeds."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch, sys
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.noise import add_connectome_noise
from src.models.vgae import VariationalGraphAutoEncoder

def cluster(m, seed):
    i,j=np.triu_indices(m.shape[1],1); v=m[:,i,j]; v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8); s=np.clip(v@v.T,0,1); np.fill_diagonal(s,0); return SpectralClustering(n_clusters=4,affinity='precomputed',assign_labels='kmeans',random_state=42).fit_predict(s)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--blend-alpha',type=float,default=.75); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    adjs=np.load(a.cache/'averaged_positive_adjacencies.npy').astype(np.float32); features=np.load(a.cache/'standardized_bold_features.npy',mmap_mode='r'); model=VariationalGraphAutoEncoder(input_dim=4800,hidden_dim=128,latent_dim=64,decoder_activation='bounded').cuda(); model.load_state_dict(torch.load(a.model,map_location='cuda',weights_only=True)); model.eval(); clean_labels=cluster(adjs,42); results=[]
    with torch.no_grad():
        for frac in [.05,.15,.30]:
            for seed in [42,123,2026]:
                noisy=[]; pred=[]
                for i in range(len(adjs)):
                    noisy_i=add_connectome_noise(adjs[i],noise_fraction=frac,mode='positive',rng=np.random.default_rng(seed+i)); out=model(torch.from_numpy(np.array(features[i],copy=True)).cuda(),torch.from_numpy(noisy_i).cuda())[0].cpu().numpy(); noisy.append(noisy_i); pred.append(out)
                noisy=np.asarray(noisy); pred=np.asarray(pred); blend=(1-a.blend_alpha)*noisy+a.blend_alpha*pred; tri=np.triu_indices(264,1); bl=cluster(blend,seed); results.append({'noise_fraction':frac,'seed':seed,'baseline_mse':float(np.mean((noisy[:,tri[0],tri[1]]-adjs[:,tri[0],tri[1]])**2)),'vgae_mse':float(np.mean((pred[:,tri[0],tri[1]]-adjs[:,tri[0],tri[1]])**2)),'blend_mse':float(np.mean((blend[:,tri[0],tri[1]]-adjs[:,tri[0],tri[1]])**2)),'blend_ari':float(adjusted_rand_score(clean_labels,bl))})
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({'blend_alpha':a.blend_alpha,'results':results},indent=2)); print(json.dumps({'blend_alpha':a.blend_alpha,'results':results},indent=2))
if __name__=='__main__': main()
