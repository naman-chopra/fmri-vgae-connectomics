"""Generate consolidated tables and figures from completed experiments."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def read(path): return json.loads(Path(path).read_text())

def main():
    out=Path('data/processed/final-report'); out.mkdir(parents=True,exist_ok=True)
    final=read('data/processed/final-bounded-vgae/report.json'); blend=read('data/processed/final-bounded-vgae/blends/selection_report.json'); robust=read('data/processed/final-bounded-vgae/robustness/frozen_report.json'); icsc=read('data/processed/final-bounded-vgae/icsc/report.json'); clean_demo=read('data/processed/final-bounded-vgae/clean-demographics.json'); blend_demo=read('data/processed/final-bounded-vgae/blend-demographics.json')
    summary={'cohort':818,'noise_fraction':.30,'vgae_test_mse':final['test']['vgae_mse'],'test_baseline_mse':final['test']['baseline_mse'],'selected_blend_alpha':blend['selected_alpha'],'blended_test_mse':blend['test_mse'],'blended_test_baseline_mse':blend['test_baseline_mse'],'icsc':icsc,'demographics':{'clean':{'gender_p':clean_demo.get('gender_chi2_pvalue'),'age_p':clean_demo.get('age_bracket_chi2_pvalue')},'blend':{'gender_p':blend_demo.get('gender_chi2_pvalue'),'age_p':blend_demo.get('age_bracket_chi2_pvalue')}},'robustness':robust['results']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); rows=[]
    for r in robust['results']: rows.append({'noise_fraction':r['noise_fraction'],'seed':r['seed'],'baseline_mse':r['baseline_mse'],'vgae_mse':r['vgae_mse'],'blend_mse':r['blend_mse'],'blend_ari':r['blend_ari']})
    (out/'robustness.csv').write_text('noise_fraction,seed,baseline_mse,vgae_mse,blend_mse,blend_ari\n'+'\n'.join(','.join(str(x[k]) for k in ['noise_fraction','seed','baseline_mse','vgae_mse','blend_mse','blend_ari']) for x in rows)+'\n')
    levels=sorted(set(r['noise_fraction'] for r in rows)); means={k:[np.mean([r[k] for r in rows if r['noise_fraction']==n]) for n in levels] for k in ['baseline_mse','vgae_mse','blend_mse','blend_ari']}; plt.figure(figsize=(7,4));
    for k,label in [('baseline_mse','Noisy baseline'),('vgae_mse','VGAE'),('blend_mse','Frozen blend')]: plt.plot(levels,means[k],marker='o',label=label)
    plt.xlabel('Noise fraction'); plt.ylabel('Mean edge MSE'); plt.legend(); plt.tight_layout(); plt.savefig(out/'robustness_mse.png',dpi=180); plt.close()
    names=['Noisy baseline','VGAE','Blend']; vals=[final['test']['baseline_mse'],final['test']['vgae_mse'],blend['test_mse']]; plt.figure(figsize=(6,4)); plt.bar(names,vals); plt.ylabel('Test edge MSE'); plt.tight_layout(); plt.savefig(out/'test_mse_comparison.png',dpi=180); plt.close()
    clean_sizes=[int(v) for v in icsc['clean']['module_sizes'].values()]; noisy_sizes=[int(v) for v in icsc['noisy']['module_sizes'].values()]; blend_sizes=[int(v) for v in icsc['blend']['module_sizes'].values()]; x=np.arange(7); w=.25; plt.figure(figsize=(8,4)); plt.bar(x-w,clean_sizes,w,label='Clean'); plt.bar(x,noisy_sizes,w,label='Noisy'); plt.bar(x+w,blend_sizes,w,label='Blend'); plt.xlabel('ROI module'); plt.ylabel('ROIs'); plt.legend(); plt.tight_layout(); plt.savefig(out/'module_sizes.png',dpi=180); plt.close(); print(json.dumps({'output_dir':str(out),'files':['summary.json','robustness.csv','robustness_mse.png','test_mse_comparison.png','module_sizes.png']},indent=2))
if __name__=='__main__': main()
