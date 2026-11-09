"""Interpolated stopped-step iteration on an invariant computational domain.

The grid uses u=(s,q,v_F,omega), q=Xi/(1+2s), omega=zeta_L-A(s).
Its box [0,1]x[0,1]x[0,1+tanh(2)]x[.15,1.89] is invariant until
contact or the horizon: omega is constant, v_F stays in its interval,
and q'= (Lambda-2q)/(1+2s) is negative at q=1.  No state projection or
reference-value boundary fill is used. Contact and terminal states absorb.
Only q and v_F need interpolation when h equals the time-grid spacing.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from artifact_io import save_npz_atomic
from scipy.interpolate import RegularGridInterpolator
import ttc as T


def to_z(u):
    u=np.asarray(u)
    s,q,vf,omega=u.T
    ze=omega+T.primitive(s)
    return np.stack([s,q*(1+2*s),ze-vf,ze],axis=1)


def to_u(z):
    z=np.asarray(z)
    s,xi,lam,ze=z.T
    return np.stack([s,xi/(1+2*s),ze-lam,ze-T.primitive(s)],axis=1)


def grid_run(nq=30,nv=16,no=16,h=.05,outdir='out',common_z=None,common_v=None):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    ns=int(round(T.TH/h))
    if abs(ns*h-T.TH)>1e-12: raise ValueError('h must divide the horizon')
    axes=[np.linspace(0,1,ns+1),np.linspace(0,1,nq+1),
          np.linspace(0,T.VOPT_SUP,nv+1),np.linspace(.15,1.89,no+1)]
    shape=tuple(len(a) for a in axes)
    u=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,4)
    z=to_z(u);n=len(z)
    val=np.empty(n);hit=np.empty(n,bool);limp=np.empty(n)
    ell=np.empty(n);land=np.empty((n,4));hit_step=np.empty(n,bool)
    begin=time.perf_counter()
    for a in range(0,n,100000):
        b=min(a+100000,n)
        ref=T.ttc(z[a:b,0],z[a:b,1:])
        val[a:b]=ref['V'];hit[a:b]=ref['hit'];limp[a:b]=ref['lam_imp']
        adv=T.advance(z[a:b,0],z[a:b,1:],h)
        ell[a:b]=adv['ell'];hit_step[a:b]=adv['hit']
        land[a:b]=to_u(np.column_stack([adv['time'],adv['y']]))
        print('grid',shape,'reference',b,'/',n,round(time.perf_counter()-begin,1),flush=True)
    lower=np.array([a[0] for a in axes]);upper=np.array([a[-1] for a in axes])
    violation=np.maximum(np.maximum(lower-land,land-upper),0).max()
    if violation>2e-10: raise AssertionError(f'Invariant grid escaped by {violation}')
    # Roundoff-only clipping, recorded above; no out-of-domain projection.
    land=np.minimum(np.maximum(land,lower),upper)
    initial_boundary=(z[:,1]<=0)|(z[:,0]>=T.TH-1e-13)
    landing_boundary=hit_step|(land[:,0]>=T.TH-1e-13)
    it=np.repeat(np.arange(ns+1),(nq+1)*(nv+1)*(no+1))
    io=np.tile(np.arange(no+1),(ns+1)*(nq+1)*(nv+1))
    next_it=np.minimum(it+1,ns)
    regular=~landing_boundary & ~initial_boundary
    time_error=float(np.max(np.abs(land[regular,0]-axes[0][next_it[regular]]),initial=0))
    omega_error=float(np.max(np.abs(land[:,3]-u[:,3]),initial=0))
    if time_error>2e-10 or omega_error>2e-10: raise AssertionError('Conserved coordinates failed')
    uq=land[:,1]*nq;uv=land[:,2]/T.VOPT_SUP*nv
    iq=np.minimum(np.floor(uq).astype(int),nq-1);iv=np.minimum(np.floor(uv).astype(int),nv-1)
    wq=uq-iq;wv=uv-iv
    idx=[];weights=[]
    for aq,av in [(0,0),(0,1),(1,0),(1,1)]:
        idx.append(((next_it*(nq+1)+iq+aq)*(nv+1)+iv+av)*(no+1)+io)
        weights.append((wq if aq else 1-wq)*(wv if av else 1-wv))
    initial_box=(z[:,1]>0)&(z[:,1]<=1+1e-12)&(u[:,2]<=T.VMAX+1e-12)&(z[:,3]>=.25-1e-12)&(z[:,3]<=1.75+1e-12)
    selected=initial_box&hit&(limp<=-T.GAMMA)&(val>=T.VMIN)&(val<=T.TH-z[:,0]-T.GAMMA_T)
    result={'shape':shape,'n_nodes':n,'n_selected':int(selected.sum()),'h':h,
            'coordinate_mesh':[float(a[1]-a[0]) for a in axes],
            'maximum_outside_distance_before_roundoff_clip':float(violation),
            'time_grid_error':time_error,'conserved_omega_error':omega_error,
            'operator':'terminal/contact stopped step, bilinear q-vF interpolation',
            'initializations':{}}
    for name,constant in [('zero',0.),('half',.5)]:
        w=np.full(n,constant);rows=[];profiles={}
        for k in range(1,ns+2):
            nextvalue=sum(ww*w[ii] for ww,ii in zip(weights,idx))
            nextvalue[landing_boundary]=constant if k==1 else 0.
            w=ell+nextvalue;w[initial_boundary]=0.
            err=np.abs(w-val)
            def stats(mask):
                return {'n':int(mask.sum()),'maximum':float(err[mask].max()),'rmse':float(np.sqrt(np.mean(err[mask]**2)))} if mask.any() else {'n':0,'maximum':None,'rmse':None}
            row={'k':k,'all_selected':stats(selected),
                 'reached':stats(selected&(val<=k*h+1e-12)),
                 'previously_reached':stats(selected&(val<=(k-1)*h+1e-12)),
                 'new_slab':stats(selected&(val>(k-1)*h+1e-12)&(val<=k*h+1e-12))}
            if common_z is not None and k in [5,10,20,21]:
                wi=RegularGridInterpolator(axes,w.reshape(shape),bounds_error=True)
                ec=np.abs(wi(to_u(common_z))-common_v)
                row['common_holdout']={'n':len(ec),'maximum':float(ec.max()),'rmse':float(np.sqrt(np.mean(ec**2)))}
            if k in [5,10,20,21]:
                bins=np.minimum(np.floor(val[selected]/h).astype(int),ns-1)
                profiles[str(k)]={'centers':((np.arange(ns)+.5)*h).tolist(),
                    'rmse':[float(np.sqrt(np.mean(err[selected][bins==j]**2))) if np.any(bins==j) else None for j in range(ns)]}
            rows.append(row)
        result['initializations'][name]={'rows':rows,'profiles':profiles}
    result['elapsed_seconds']=time.perf_counter()-begin
    return result


def run(outdir='out',fine=True):
    out=Path(outdir);out.mkdir(exist_ok=True,parents=True)
    # Independent common points for a fair comparison of the two grids.
    rng=np.random.default_rng(320261103)
    s,y=T.sample_box(30000,rng);r=T.ttc(s,y);m=T.in_D(r)
    z=np.column_stack([s[m],y[m]]);v=r['V'][m]
    save_npz_atomic(out/'iteration_common_test.npz',z=z,V=v)
    configs=[(30,16,16),(60,32,32)] if fine else [(30,16,16)]
    report={'common_seed':320261103,'common_pool_size':30000,'common_selected_size':len(v),'grids':[]}
    for config in configs:
        report['grids'].append(grid_run(*config,outdir=out,common_z=z,common_v=v))
        (out/'iteration_results.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--outdir',default='out');parser.add_argument('--coarse-only',action='store_true')
    args=parser.parse_args();run(args.outdir,not args.coarse_only)
