"""Held-out dense matrix denoising baseline for diagnosing VGAE behavior."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import HCPMatlabConnectomeDataset
from src.data.noise import add_connectome_noise

def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path,default=Path('original-data/HCP_rs_power.zip')); p.add_argument('--cache',type=Path); p.add_argument('--subjects',type=int,default=32); p.add_argument('--epochs',type=int,default=2); p.add_argument('--noise-fraction',type=float,default=0.05); p.add_argument('--output-dir',type=Path,default=Path('data/processed/matrix-denoiser'))
    a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    if a.cache:
        adjs=np.load(a.cache/'averaged_positive_adjacencies.npy',mmap_mode='r')[:a.subjects]
        class CachedDataset:
            def __len__(self): return len(adjs)
            def __getitem__(self, i): return {'adj': torch.from_numpy(np.array(adjs[i], copy=True))}
        ds=CachedDataset()
    else:
        ds=HCPMatlabConnectomeDataset(a.archive,positive_only=True); ds.subjects=ds.subjects[:a.subjects]
    order=np.random.default_rng(42).permutation(len(ds)); n=int(.7*len(order)); nv=int(.15*len(order)); train,val,test=order[:n],order[n:n+nv],order[n+nv:]
    tri=np.triu_indices(264,1); d=len(tri[0]); model=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,d)).cuda(); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    def evaluate(indices,seed):
        model.eval(); rng=np.random.default_rng(seed); base=[]; pred=[]
        with torch.no_grad():
            for i in indices:
                clean=ds[int(i)]['adj'].numpy()[tri]; noisy=add_connectome_noise(ds[int(i)]['adj'].numpy(),noise_fraction=a.noise_fraction,mode='positive',rng=rng)[tri]
                out=model(torch.from_numpy(noisy).cuda()).cpu().numpy(); base.append(float(np.mean((noisy-clean)**2))); pred.append(float(np.mean((out-clean)**2)))
        return {'baseline_mse':float(np.mean(base)),'denoiser_mse':float(np.mean(pred))}
    history=[]
    for ep in range(a.epochs):
        model.train(); losses=[]
        for i in train:
            clean=ds[int(i)]['adj'].numpy()[tri]; noisy=add_connectome_noise(ds[int(i)]['adj'].numpy(),noise_fraction=a.noise_fraction,mode='positive',rng=np.random.default_rng(42+ep*10000+int(i)))[tri]
            loss=nn.functional.mse_loss(model(torch.from_numpy(noisy).cuda()),torch.from_numpy(clean).cuda()); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(loss.item())
        row={'epoch':ep+1,'train_loss':float(np.mean(losses)),'val':evaluate(val,1000+ep)}; history.append(row); print(row)
    report={'device':'cuda','noise_fraction':a.noise_fraction,'subjects':len(ds),'split':{'train':len(train),'validation':len(val),'test':len(test)},'history':history,'test':evaluate(test,2000),'peak_memory':int(torch.cuda.max_memory_allocated())}; a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
