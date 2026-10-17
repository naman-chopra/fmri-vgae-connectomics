"""Train the selected bounded BOLD-feature VGAE and materialize all outputs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.noise import add_connectome_noise
from src.models.losses import vgae_loss
from src.models.vgae import VariationalGraphAutoEncoder

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--epochs',type=int,default=10); p.add_argument('--noise-fraction',type=float,default=.30); p.add_argument('--output-dir',type=Path,default=Path('data/processed/final-bounded-vgae')); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.manual_seed(a.seed); np.random.seed(a.seed); adjs=np.load(a.cache/'averaged_positive_adjacencies.npy').astype(np.float32); features=np.load(a.cache/'standardized_bold_features.npy',mmap_mode='r'); n=len(adjs); order=np.random.default_rng(a.seed).permutation(n); nt=int(.7*n); nv=int(.15*n); train,val,test=order[:nt],order[nt:nt+nv],order[nt+nv:]
    model=VariationalGraphAutoEncoder(input_dim=4800,hidden_dim=128,latent_dim=64,decoder_activation='bounded').cuda(); opt=torch.optim.Adam(model.parameters(),lr=2e-5,weight_decay=1e-4); mask=torch.triu(torch.ones((264,264),dtype=torch.bool,device='cuda'),diagonal=1); history=[]
    def evaluate(indices,seed):
        model.eval(); rng=np.random.default_rng(seed); base=[]; pred=[]
        with torch.no_grad():
            for i in indices:
                clean=torch.from_numpy(adjs[int(i)]).cuda(); noisy=torch.from_numpy(add_connectome_noise(adjs[int(i)],noise_fraction=a.noise_fraction,mode='positive',rng=rng)).cuda(); out,_,_,_=model(torch.from_numpy(np.array(features[int(i)],copy=True)).cuda(),noisy); base.append(float(torch.mean((noisy-clean)[mask]**2).item())); pred.append(float(torch.mean((out-clean)[mask]**2).item()))
        return {'baseline_mse':float(np.mean(base)),'vgae_mse':float(np.mean(pred))}
    for ep in range(a.epochs):
        model.train(); losses=[]
        for i in train:
            clean=torch.from_numpy(adjs[int(i)]).cuda(); noisy=torch.from_numpy(add_connectome_noise(adjs[int(i)],noise_fraction=a.noise_fraction,mode='positive',rng=np.random.default_rng(a.seed+ep*10000+int(i)))).cuda(); x=torch.from_numpy(np.array(features[int(i)],copy=True)).cuda(); opt.zero_grad(set_to_none=True); out,_,mu,logvar=model(x,noisy); loss,recon,kl=vgae_loss(out,clean,mu,logvar,kl_weight=0,reconstruction_mask=mask); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),.5); opt.step(); losses.append(loss.item())
        row={'epoch':ep+1,'train_loss':float(np.mean(losses)),'val':evaluate(val,a.seed+1000+ep)}; history.append(row); print(row,flush=True)
    report={'device':'cuda','feature_source':'standardized_four_run_bold','decoder':'bounded','latent_dim':64,'noise_fraction':a.noise_fraction,'subjects':n,'split':{'train':len(train),'validation':len(val),'test':len(test)},'history':history,'test':evaluate(test,a.seed+2000),'cuda_peak_memory_allocated_bytes':int(torch.cuda.max_memory_allocated())}
    a.output_dir.mkdir(parents=True,exist_ok=True); model.eval(); outputs=np.empty_like(adjs)
    with torch.no_grad():
        for i in range(n):
            noisy=torch.from_numpy(add_connectome_noise(adjs[i],noise_fraction=a.noise_fraction,mode='positive',rng=np.random.default_rng(a.seed+50000+i))).cuda(); x=torch.from_numpy(np.array(features[i],copy=True)).cuda(); outputs[i]=model(x,noisy)[0].cpu().numpy()
    np.save(a.output_dir/'denoised_connectomes.npy',outputs); np.save(a.output_dir/'clean_connectomes.npy',adjs); (a.output_dir/'subject_ids.json').write_text((a.cache/'subjects.json').read_text()); torch.save(model.state_dict(),a.output_dir/'model.pt'); (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
