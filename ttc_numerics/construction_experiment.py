"""Independent complete-grid/ReLU experiment on a provably colliding 4D box.

Run: python3 construction_experiment.py
The fixed geometry and all grid nodes are defined before held-out queries.
No test coordinate is used for support selection. Existing grid files can be
reused with --reuse-grids; use --meshes 0.02 for a short smoke run.

The analytical error columns concern exact TTC nodal values and exact real
arithmetic. Reference-solver and floating-point checks are empirical; no
uniform validated ODE-error bound is claimed.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from scipy.integrate import solve_ivp
import construct as C
import ttc as T
from model_io import load_model

B_LO = np.array([.15, .05, -1.4, .27])
B_HI = np.array([.27, .12, -1.1, .52])
D_LO = np.array([.15, .08, -1.4, .30])
D_HI = np.array([.25, .12, -1.2, .50])
H = .02
SEED_B, SEED_D, SEED_TUBE = 202611031, 202611032, 202611033
SEED_PERTURB, SEED_VALIDATION = 202611034, 202611035


def constants():
    c, amax = .3, 1.2
    max_time = .5 * np.log(2.26 / 1.46)
    b = (1 - np.exp(-2 * max_time)) / 2
    lopt = 1 / np.cosh(1.88)**2
    denom = 1 - lopt * max_time**2
    L = np.sqrt((2 * amax * max_time)**2 + 1 + b*b + (max_time-b)**2) / (c*denom)
    return dict(impact_speed_lower_bound=c, maximum_contact_time=max_time,
        minimum_contact_time=.05/1.4, integrated_closing_lower_bound=.4-1.16*max_time,
        leader_speed_bounds=[.03,.76], closing_speed_upper_bound=1.4,
        optimal_velocity_lipschitz_on_precontact_gaps=lopt,
        leader_acceleration_absolute_bound=amax, filter_coefficient=b,
        volterra_denominator=denom, analytical_ttc_lipschitz=L,
        construction_box_lo=B_LO.tolist(), construction_box_hi=B_HI.tolist(),
        initial_box_lo=D_LO.tolist(), initial_box_hi=D_HI.tolist(), stopped_step=H)


def reference_one(z):
    """Independent piecewise DOP853 check; not an interval enclosure."""
    s, yy = float(z[0]), np.array(z[1:], float)
    start = s
    def event(t, y): return y[0]
    event.terminal, event.direction = True, -1
    for end in T.AL_BREAKS[T.AL_BREAKS > start + 1e-14]:
        a = float(T.aL((start+end)/2))
        sol = solve_ivp(lambda t,y:T.field_const(a,y), (start,float(end)), yy,
            method='DOP853', rtol=2e-12, atol=2e-14, max_step=.005, events=event)
        yy = sol.y[:,-1]
        if len(sol.t_events[0]):
            return float(sol.t_events[0][0] - s), float(yy[1])
        start = float(end)
    return 1-s, float('nan')


def metrics(w, v):
    e = np.abs(np.asarray(w)-np.asarray(v))
    return dict(maximum=float(e.max()), rmse=float(np.sqrt(np.mean(e*e))))


def sha256_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def save_json_atomic(path,value):
    path=Path(path)
    with tempfile.NamedTemporaryFile(mode='w',suffix='.json',dir=path.parent,delete=False) as handle:
        temporary=Path(handle.name)
        json.dump(value,handle,indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        json.loads(temporary.read_text())
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def build_grid(h0, out, reuse=False):
    lo, hi, ns = C.TensorGrid.geometry(B_LO,B_HI,h0)
    key=f'{h0:g}'.replace('.','p')
    path=out/f'construction_grid_{key}.npz'
    if reuse and path.exists():
        grid=C.TensorGrid.load(path)
        if not (np.array_equal(grid.lo,lo) and np.array_equal(grid.hi,hi) and np.array_equal(grid.nseg,ns)):
            raise ValueError('Existing grid geometry differs from the declared geometry')
        return grid, dict(source='reused saved complete grid',seconds=None)
    vals=np.empty(int(np.prod(ns+1)))
    min_normal=0.;max_normal=-np.inf
    t0=time.perf_counter()
    for a,z in C.TensorGrid.node_batches(lo,hi,ns,batch_size=100000):
        rr=T.ttc(z[:,0],z[:,1:],dt=.001,nbis=40)
        if not np.all(rr['hit'] & (rr['lam_imp'] < -.3)):
            raise AssertionError('A complete-grid node failed the proved collision margin')
        vals[a:a+len(z)]=rr['V']
        min_normal=min(min_normal,float(rr['lam_imp'].min()))
        max_normal=max(max_normal,float(rr['lam_imp'].max()))
    grid=C.TensorGrid(lo,hi,ns,vals)
    grid.save(path, reference_dt=np.array(.001), event_bisections=np.array(40))
    return grid, dict(source='new complete-grid numerical TTC coefficients',seconds=time.perf_counter()-t0,
        observed_impact_velocity_min=min_normal,observed_impact_velocity_max=max_normal)


def run(meshes,n_test,out,reuse=False,model_directory=Path('out')):
    out.mkdir(parents=True,exist_ok=True)
    A=constants()
    assert A['integrated_closing_lower_bound'] > B_HI[1]
    R=dict(geometry=A,query_protocol='independent uniform queries after fixing the complete grid; no support selection',
        reference_protocol='RK4 dt<=1e-3, all forcing jumps and stops split, 40 event bisections; independent sampled DOP853 checks',
        certificate_scope='analytical exact-node interpolation/realization bounds; floating-point reference errors are empirically checked, not uniformly certified',
        seeds=dict(box=SEED_B,initial=SEED_D,tube=SEED_TUBE,perturbation=SEED_PERTURB,validation=SEED_VALIDATION),
        n_test_per_box=int(n_test),meshes={})
    # Geometry above is fixed and independent of these draws.
    zB=np.random.default_rng(SEED_B).uniform(B_LO,B_HI,(n_test,4))
    zD=np.random.default_rng(SEED_D).uniform(D_LO,D_HI,(n_test,4))
    rb=T.ttc(zB[:,0],zB[:,1:]);rd=T.ttc(zD[:,0],zD[:,1:],snap_times=(H,))
    VB,VD=rb['V'],rd['V']
    assert np.all(rd['hit'] & (VD>H))
    landing=np.column_stack([zD[:,0]+H,rd['snaps'][H]])
    kappa=np.random.default_rng(SEED_TUBE).uniform(0,H,n_test)
    rt=T.advance(zD[:,0],zD[:,1:],kappa)
    ztube=np.column_stack([zD[:,0]+kappa,rt['y']])
    for points in [landing,ztube]:
        assert np.all(points>=B_LO-1e-12) and np.all(points<=B_HI+1e-12)
    vt_direct=T.ttc(ztube[:,0],ztube[:,1:])['V']
    R['semigroup_check_max']=float(np.max(np.abs(vt_direct-(VD-kappa))))
    # Nodal data and held-out inputs can be reproduced separately.
    query_arrays=dict(z_box=zB,V_box=VB,z_initial=zD,V_initial=VD,
        z_landing=landing,V_landing=VD-H,z_tube=ztube,V_tube=vt_direct,kappa=kappa)
    R['query_sha256']=dict(box=sha256_array(zB),initial=sha256_array(zD),tube=sha256_array(ztube))
    # A common slice, wholly in B_c, for plotting trained and constructed maps.
    xi=np.linspace(B_LO[1],B_HI[1],121);lam=np.linspace(B_LO[2],B_HI[2],121)
    xx,ll=np.meshgrid(xi,lam,indexing='xy')
    zslice=np.column_stack([np.full(xx.size,.2),xx.ravel(),ll.ravel(),np.full(xx.size,.4)])
    vslice=T.ttc(zslice[:,0],zslice[:,1:])['V']
    slice_arrays=dict(xi=xi,lam=lam,z=zslice,V=vslice,s=np.array(.2),zeta=np.array(.4))
    # The supplied models are evaluated unchanged on precisely the same local
    # queries, so the model comparison does not confound different domains.
    R['unchanged_learned_models']={}
    for tag in ('mse','res'):
        checkpoint=Path(model_directory)/f'model_{tag}.pt'
        if not checkpoint.exists():
            raise FileNotFoundError(f'Supplied model needed for comparison: {checkpoint}')
        model=load_model(tag,model_directory)
        mb,md,ml,mt=model(zB),model(zD),model(landing),model(ztube)
        R['unchanged_learned_models'][tag]=dict(box=metrics(mb,VB),initial=metrics(md,VD),
            tube=metrics(mt,vt_direct),local_stopped_residual=float(np.max(abs(md-H-ml))),
            relu_units=int(model.n_relu_units),parameters=int(model.n_parameters),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            training_status='unchanged supplied checkpoint; no retraining or local fitting')
        query_arrays.update({f'{tag}_box':mb,f'{tag}_initial':md,
            f'{tag}_landing':ml,f'{tag}_tube':mt})
        slice_arrays[f'model_{tag}']=model(zslice)
    C.save_npz_atomic(out/'construction_queries.npz',**query_arrays)
    # Reference checks use a different fixed stream and include domain corners.
    valrng=np.random.default_rng(SEED_VALIDATION)
    corners=np.array(list(itertools.product(*zip(B_LO,B_HI))))
    check_z=np.vstack([corners,zB[valrng.choice(n_test,min(100,n_test),replace=False)]])
    dop=np.array([reference_one(z)[0] for z in check_z])
    co=T.ttc(check_z[:,0],check_z[:,1:],dt=.001)['V']
    fi=T.ttc(check_z[:,0],check_z[:,1:],dt=.0005)['V']
    R['reference_checks']=dict(n=int(len(check_z)),rk4_vs_dop853_max=float(np.max(abs(co-dop))),
        rk4_refinement_max=float(np.max(abs(co-fi))),refined_vs_dop853_max=float(np.max(abs(fi-dop))),
        is_uniform_validated_bound=False)
    C.save_npz_atomic(out/'construction_reference_checks.npz',z=check_z,dop853=dop,rk4=co,rk4_refined=fi)
    for h0 in meshes:
        t0=time.perf_counter()
        grid,build_info=build_grid(h0,out,reuse)
        k=int(np.ceil(np.log2(1/h0)))
        meta=C.realization_metadata(4,grid.values.size,k,A['maximum_contact_time'])
        interp_bound=.5*A['analytical_ttc_lipschitz']*float(np.linalg.norm(grid.mesh))
        item=dict(h0=float(h0),actual_mesh=grid.mesh.tolist(),nseg=grid.nseg.tolist(),n_nodes=int(grid.values.size),
            n_cells=int(np.prod(grid.nseg)),coefficient_sha256=sha256_array(grid.values),
            nodal_coefficient_bytes=int(grid.values.nbytes),build=build_info,realization=meta,
            exact_node_interpolation_bound=interp_bound,
            exact_node_relu_bound=interp_bound+meta['realization_error_bound'],
            exact_node_local_residual_bound=2*(interp_bound+meta['realization_error_bound']))
        WI=grid.evaluate(zB);WN=grid.evaluate(zB,k=k)
        item['box_interpolant']=metrics(WI,VB);item['box_relu']=metrics(WN,VB)
        item['relu_minus_interpolant']=metrics(WN,WI)
        ID=grid.evaluate(zD);ND=grid.evaluate(zD,k=k)
        IL=grid.evaluate(landing);NL=grid.evaluate(landing,k=k)
        IT=grid.evaluate(ztube);NT=grid.evaluate(ztube,k=k)
        item['initial_interpolant']=metrics(ID,VD);item['initial_relu']=metrics(ND,VD)
        item['tube_interpolant']=metrics(IT,vt_direct);item['tube_relu']=metrics(NT,vt_direct)
        item['local_stopped_residual_interpolant']=float(np.max(abs(ID-H-IL)))
        item['local_stopped_residual_relu']=float(np.max(abs(ND-H-NL)))
        eta=.01
        pert=np.random.default_rng(SEED_PERTURB+int(round(h0*100000)))
        gp=C.TensorGrid(grid.lo,grid.hi,grid.nseg,grid.values+pert.uniform(-eta,eta,grid.shape))
        perturb_obs=float(np.max(abs(gp.evaluate(zB,k=k)-WN)))
        item['nodal_perturbation']=dict(eta=eta,observed_relu_change=perturb_obs,
            proven_change_bound=eta*meta['nodal_error_amplification'])
        # These finite checks do not stand in for the analytic global product bound.
        if h0==meshes[0] and grid.values.size<=20000:
            test=np.vstack([zB[:8],corners])
            item['full_sum_equivalence_n']=len(test)
            item['full_sum_equivalence_max']=float(np.max(abs(grid.evaluate(test,k=k)-grid.full_sum(test,k=k))))
            outside=np.tile((B_LO+B_HI)/2,(4,1))
            outside[0,0]=B_LO[0]-2*grid.mesh[0]
            outside[1,1]=B_HI[1]+2*grid.mesh[1]
            outside[2,2]=B_LO[2]-2*grid.mesh[2]
            outside[3,3]=B_HI[3]+2*grid.mesh[3]
            item['global_network_outside_support_max']=float(abs(grid.full_sum(outside,k=k)).max())
        zero_factors=np.random.default_rng(SEED_VALIDATION).uniform(0,1,(1000,4))
        zero_factors[np.arange(len(zero_factors)),np.arange(len(zero_factors))%4]=0.
        item['zero_factor_preservation_max']=float(abs(C.product_relu(zero_factors,k=k)).max())
        validation_ids=valrng.choice(grid.values.size,100,replace=False)
        validation_multi=np.column_stack(np.unravel_index(validation_ids,grid.shape))
        validation_nodes=grid.lo+validation_multi*grid.mesh
        exact_checks=np.array([reference_one(z)[0] for z in validation_nodes])
        item['sampled_nodal_solver_error_max']=float(np.max(abs(grid.values.ravel()[validation_ids]-exact_checks)))
        suffix=f'{h0:g}'.replace('.','p')
        slice_arrays[f'interpolant_{suffix}']=grid.evaluate(zslice)
        slice_arrays[f'relu_{suffix}']=grid.evaluate(zslice,k=k)
        item['total_seconds']=time.perf_counter()-t0
        R['meshes'][f'{h0:g}']=item
        print(json.dumps(dict(mesh=h0,n_nodes=item['n_nodes'],relu=item['box_relu'],
            interpolant=item['box_interpolant'],bound=item['exact_node_relu_bound'],seconds=item['total_seconds'])),flush=True)
        save_json_atomic(out/'construction_results.json',R)
        C.save_npz_atomic(out/'construction_slice.npz',**slice_arrays)
    return R


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--meshes',nargs='+',type=float,default=[.02,.01,.005])
    parser.add_argument('--n-test',type=int,default=10000)
    parser.add_argument('--out',type=Path,default=Path('out'))
    parser.add_argument('--reuse-grids',action='store_true')
    parser.add_argument('--model-directory',type=Path,default=Path('provenance/checkpoints'))
    args=parser.parse_args()
    run(args.meshes,args.n_test,args.out,args.reuse_grids,args.model_directory)
