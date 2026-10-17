"""Select residual blend strength on validation subjects, then evaluate once on test."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.noise import add_connectome_noise

def main():
    p=argparse.ArgumentParser(); p.add_argument('--clean',type=Path,required=True); p.add_argument('--denoised',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--noise-fraction',type=float,default=.30); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    clean=np.load(a.clean); den=np.load(a.denoised); order=np.random.default_rng(a.seed).permutation(len(clean)); nt=int(.7*len(order)); nv=int(.15*len(order)); val=order[nt:nt+nv]; test=order[nt+nv:]; tri=np.triu_indices(clean.shape[1],1); noisy=np.stack([add_connectome_noise(clean[i],noise_fraction=a.noise_fraction,mode='positive',rng=np.random.default_rng(a.seed+50000+i)) for i in range(len(clean))])
    candidates=[0,.25,.5,.75,1.0]
    val_scores={str(alpha):float(np.mean((((1-alpha)*noisy[val]+alpha*den[val])-clean[val])[:,tri[0],tri[1]]**2)) for alpha in candidates}; selected=float(min(candidates,key=lambda x:val_scores[str(x)])); test_blend=(1-selected)*noisy[test]+selected*den[test]; report={'selected_alpha':selected,'validation_mse':val_scores,'test_mse':float(np.mean((test_blend[:,tri[0],tri[1]]-clean[test][:,tri[0],tri[1]])**2)),'test_baseline_mse':float(np.mean((noisy[test][:,tri[0],tri[1]]-clean[test][:,tri[0],tri[1]])**2)),'split':{'validation':len(val),'test':len(test)},'candidate_alphas':candidates}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
