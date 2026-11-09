"""Reproduce supplied checkpoints, then evaluate a fresh continuous-time holdout.

No training occurs here.  Checkpoints retain the supplied rounded-time,
100-epoch training provenance.  The new main evaluation uses unrounded times
and corrected event-aware reference integration.  All envelopes are sampled
quantities, not validated bounds over a continuum region.
"""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
from artifact_io import save_npz_atomic
import ttc as T
from model_io import load_model


def references(s,y,batch=100000,snaps=()):
    keys=('s','V','hit','lam_imp','y_imp','y_final','initial_gap')
    parts={k:[] for k in keys};sp={float(k):[] for k in snaps}
    for a in range(0,len(s),batch):
        r=T.ttc(s[a:a+batch],y[a:a+batch],snap_times=snaps)
        for k in keys:parts[k].append(r[k])
        for k in snaps:sp[float(k)].append(r['snaps'][float(k)])
    out={k:np.concatenate(v,axis=0) for k,v in parts.items()}
    out['snaps']={k:np.concatenate(v,axis=0) for k,v in sp.items()}
    return out


def error_metrics(e,m):
    return dict(max=float(np.max(e[m])) if np.any(m) else None,
                rmse=float(np.sqrt(np.mean(e[m]**2))) if np.any(m) else None)


def summarize(z,r,models):
    D=T.in_D(r);DD=T.in_DD(r);pred={};R={}
    for tag,model in models.items():
        w=model(z);pred[tag]=w;e=np.abs(w-r['V'])
        R[tag]=dict(params=model.n_parameters,relu_units=model.n_relu_units,
                    linf_all=float(e.max()),rmse_all=float(np.sqrt(np.mean(e**2))),
                    linf_D=float(e[D].max()),rmse_D=float(np.sqrt(np.mean(e[D]**2))),
                    linf_DD=float(e[DD].max()))
    return R,pred


def original_test_coordinates():
    rng=np.random.default_rng(2026)
    for n in [1000000,100000]:
        # sample_box makes four arrays of uniform draws, in this order.
        for _ in range(4):rng.random(n)
    s,y=T.sample_box(300000,rng)
    s=np.rint(s/T.DT)*T.DT
    return s,y


def baseline(models,out,baseline_results=Path('results.json')):
    s,y=original_test_coordinates();t0=time.time()
    print('Original rounded test: reference integration',flush=True)
    r=references(s,y);z=np.c_[s,y]
    R,_=summarize(z,r,models)
    old=json.loads(Path(baseline_results).read_text())
    fields=['linf_all','rmse_all','linf_D','rmse_D','linf_DD']
    differences={tag:{f:R[tag][f]-old[tag][f] for f in fields} for tag in models}
    report=dict(n_test=len(s),n_D=int(T.in_D(r).sum()),n_hit=int(r['hit'].sum()),
                archived_n_D=old['n_test_D'],model_metrics=R,
                difference_from_supplied_results=differences,
                max_absolute_metric_difference=max(abs(v) for d in differences.values() for v in d.values()),
                note='Supplied weights evaluated with NumPy float32 BLAS, corrected solver on original rounded test coordinates.',
                elapsed_seconds=time.time()-t0)
    (out/'baseline_reproduction.json').write_text(json.dumps(report,indent=2))
    print('Baseline',json.dumps(report),flush=True)
    return report


def chains(zD,VD,models,HS):
    ks=tuple(np.round(np.arange(0.,T.TH+1e-12,min(HS)),10))
    r=T.ttc(zD[:,0],zD[:,1:],snap_times=ks)
    imp=np.c_[zD[:,0]+r['V'],r['y_imp']]
    reports={tag:{} for tag in models}
    bvals={tag:np.abs(model(imp)) for tag,model in models.items()}
    # Fresh references and chain snapshots are internally consistent;
    # their agreement with independent restarts is checked separately.
    for h in HS:
        N=np.maximum(1,np.ceil(VD/h-1e-12).astype(int))
        points=[];owners=[];levels=[];ell=[];landing=[]
        for j in range(int(N.max())):
            sel=np.flatnonzero(N>j)
            k=round(j*h,10)
            zz=np.c_[zD[sel,0]+k,r['snaps'][k][sel]]
            ll=np.minimum(h,VD[sel]-k)
            after=np.empty_like(zz)
            last=N[sel]==j+1
            after[last]=imp[sel[last]]
            if np.any(~last):
                kk=round((j+1)*h,10)
                after[~last]=np.c_[zD[sel[~last],0]+kk,r['snaps'][kk][sel[~last]]]
            points.append(zz);landing.append(after);ell.append(ll);owners.append(sel);levels.append(np.full(len(sel),j))
        pts=np.concatenate(points);land=np.concatenate(landing);ells=np.concatenate(ell);own=np.concatenate(owners)
        for tag,model in models.items():
            wp=model(pts);wl=model(land);res=wp-ells-wl
            delta=float(np.abs(res).max());b=float(bvals[tag].max())
            bound=N*delta+b
            error=np.abs(model(zD)-VD)
            path=np.bincount(own,weights=np.abs(res),minlength=len(VD))+bvals[tag]
            signed_sum=np.bincount(own,weights=res,minlength=len(VD))
            defect=np.max(np.abs((model(zD)-VD)-(model(imp)+signed_sum)))
            reports[tag][str(h)]=dict(delta=delta,delta_D=float(np.abs(res[:len(zD)]).max()),
                  b_sample=b,bound_max=float(bound.max()),err_max=float(error.max()),
                  frac_ok=float(np.mean(error<=bound+1e-9)),ratio_max=float(np.max(error/bound)),
                  n_chain=len(pts),path_sum_bound_max=float(path.max()),
                  max_algebraic_telescoping_defect=float(defect),
                  certified_uniform=False)
    return reports,{tag:float(vals.max()) for tag,vals in bvals.items()},imp,r


def main():
    p=argparse.ArgumentParser();p.add_argument('--n-test',type=int,default=300000)
    p.add_argument('--seed',type=int,default=20261103);p.add_argument('--skip-baseline',action='store_true')
    p.add_argument('--checkpoint-dir',type=Path,default=Path('provenance/checkpoints'))
    p.add_argument('--baseline-results',type=Path,default=Path('provenance/results.json'))
    p.add_argument('--out-dir',type=Path,default=Path('out'))
    args=p.parse_args();out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    if args.n_test < 1:p.error('--n-test must be positive')
    models={tag:load_model(tag,args.checkpoint_dir) for tag in ['mse','res']}
    if not args.skip_baseline:baseline(models,out,args.baseline_results)
    print('Fresh continuous-time holdout: generating references',flush=True);t0=time.time()
    s,y=T.sample_box(args.n_test,np.random.default_rng(args.seed));r=references(s,y);z=np.c_[s,y]
    report,pred=summarize(z,r,models);inD=T.in_D(r)
    R=dict(n_test=args.n_test,n_test_D=int(inD.sum()),frac_hit_test=float(r['hit'].mean()),
           frac_D_test=float(inD.mean()),frac_DD_test=float(T.in_DD(r).mean()),
           seed=args.seed,initial_times='continuous/unrounded',dt=T.DT,nbis=40,
           D_margins=dict(vmin=T.VMIN,gamma=T.GAMMA,gamma_t=T.GAMMA_T),
           checkpoint_provenance='Supplied 100-epoch runs; 1e6 training and 1e5 validation states with initial times rounded to 1e-3; no retraining in revised evaluation.',
           sampling='independent uniforms in (s,Xi,vF,zeta_L); Lambda=zeta_L-vF',
           sampling_box={k:list(v) for k,v in T.BOX.items()},
           certified_uniform=False,**report)
    zD=z[inD];VD=r['V'][inD]
    print('Holdout n_D',len(zD),'references_seconds',time.time()-t0,flush=True)
    cert,bvals,imp,rd=chains(zD,VD,models,[.025,.05,.1,.2])
    for tag in models:R[tag]['cert']=cert[tag];R[tag]['b']=bvals[tag]
    R['env_lambda']={}
    for lam in [.5,.9,1.,1.1]:
        R['env_lambda'][str(lam)]=dict(T_res_max=max(lam-1,0)*.05,
            S_res_max=abs(lam-1)*.05,err_max=float(np.max(abs(lam-1)*VD)))
    R['V_D_range']=[float(VD.min()),float(VD.max())]
    R['elapsed_seconds']=time.time()-t0
    R['checkpoint_sha256']={tag:hashlib.sha256((args.checkpoint_dir/f'model_{tag}.pt').read_bytes()).hexdigest() for tag in models}
    save_npz_atomic(out/'test_holdout.npz',s=s,y=y,V=r['V'],hit=r['hit'],lam_imp=r['lam_imp'],y_imp=r['y_imp'],
                        inD=inD,inDD=T.in_DD(r),pred_mse=pred['mse'],pred_res=pred['res'])
    save_npz_atomic(out/'holdout_D_chains.npz',z=zD,V=VD,impact=imp,**{f'snap_{k:g}':v for k,v in rd['snaps'].items()})
    (out/'evaluation_results.json').write_text(json.dumps(R,indent=2))
    print('Fresh evaluation',json.dumps(R,indent=2),flush=True)

if __name__=='__main__':main()
