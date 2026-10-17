"""Compare fixed-k subject clustering before and after denoising."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def vectors(m):
    i,j=np.triu_indices(m.shape[1],1); return m[:,i,j].astype(np.float32)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--clean',type=Path,required=True); p.add_argument('--denoised',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--clusters',type=int,default=4); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    clean=np.load(a.clean); denoised=np.load(a.denoised); a.output_dir.mkdir(parents=True,exist_ok=True); reports={}
    for name,matrix in [('clean',clean),('denoised',denoised)]:
        v=vectors(matrix); v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8); sim=np.clip(v@v.T,-1,1); np.fill_diagonal(sim,0); labels=SpectralClustering(n_clusters=a.clusters,affinity='precomputed',assign_labels='kmeans',random_state=a.seed).fit_predict(np.maximum(sim,0)); reports[name]={'cluster_sizes':{str(int(k)):int(np.sum(labels==k)) for k in np.unique(labels)},'silhouette_on_vectors':float(silhouette_score(v,labels)),'labels':labels.tolist()}; np.save(a.output_dir/f'{name}_similarity.npy',sim); np.save(a.output_dir/f'{name}_labels.npy',labels)
    reports['agreement']={'adjusted_rand_index':float(adjusted_rand_score(reports['clean']['labels'],reports['denoised']['labels'])),'normalized_mutual_information':float(normalized_mutual_info_score(reports['clean']['labels'],reports['denoised']['labels']))}; reports['clusters']=a.clusters; (a.output_dir/'report.json').write_text(json.dumps(reports,indent=2)); print(json.dumps(reports,indent=2))
if __name__=='__main__': main()
