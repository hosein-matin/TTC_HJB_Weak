"""Independent solver checks for the revised TTC experiments.

These are floating-point comparisons, not interval/validated error bounds.
Run: python3 reference_checks.py
"""
from pathlib import Path
import json
import numpy as np
from artifact_io import save_npz_atomic
from scipy.integrate import solve_ivp
import ttc as T


def independent_reference(s, y, rtol=2e-12, atol=2e-14, max_step=.002):
    y=np.asarray(y,dtype=float).copy()
    if y[0] <= 0: return 0.0, True, float(y[1]), y
    if s >= T.TH: return 0.0, False, None, y
    def event(t,x): return x[0]
    event.terminal=True; event.direction=-1
    t0=float(s)
    for t1 in T.AL_BREAKS[T.AL_BREAKS>s+1e-14]:
        a=float(T.aL((t0+t1)/2))
        sol=solve_ivp(lambda t,x:T.field_const(a,x), (t0,float(t1)), y,
                      method='DOP853',rtol=rtol,atol=atol,max_step=max_step,events=event)
        y=sol.y[:,-1]
        if len(sol.t_events[0]):
            tt=float(sol.t_events[0][0]);y[0]=0.
            return tt-s,True,float(y[1]),y
        t0=float(t1)
    return T.TH-s,False,None,y


def main():
    out=Path('out');out.mkdir(exist_ok=True)
    rng=np.random.default_rng(882711)
    s,y=T.sample_box(2500,rng)
    coarse=T.ttc(s,y,snap_times=(.05,))
    selD=np.flatnonzero(T.in_D(coarse))
    selected=np.r_[selD,np.arange(80)]
    selected=np.unique(selected)
    ss,yy=s[selected],y[selected]
    # Explicit arbitrary starts near every forcing jump and terminal/target cases.
    sb=np.array([0.,.1999999,.2,.2000001,.3999999,.4,.4000001,.5999999,.6,.6000001,.8,.99991,1.,.31])
    yb=np.tile([.12,-1.15,.5],(len(sb),1));yb[-1,0]=0.
    ss=np.r_[ss,sb];yy=np.vstack([yy,yb])
    a=T.ttc(ss,yy,snap_times=(0.,.0137,.05,.2,1.))
    b=T.ttc(ss,yy,dt=.0005,snap_times=(0.,.0137,.05,.2,1.))
    c=T.ttc(ss,yy,dt=.00025,snap_times=(0.,.0137,.05,.2,1.))
    independent=[independent_reference(x,v) for x,v in zip(ss,yy)]
    vv=np.array([x[0] for x in independent]);hh=np.array([x[1] for x in independent])
    same=a['hit']==hh
    e=np.abs(a['V']-vv)
    snaperr=max(float(np.max(np.abs(a['snaps'][k]-c['snaps'][k]))) for k in a['snaps'])
    sd,yd=s[selD],y[selD]
    pre=T.ttc(sd,yd,snap_times=(.05,))
    restarted=T.ttc(sd+.05,pre['snaps'][.05])
    semigroup=np.abs(pre['V']-(.05+restarted['V']))
    # A genuine within-step closing-to-opening contact: both step endpoint gaps
    # can be positive. Compare the protected minimum check to adaptive reference.
    near_s=np.array([.4]);near_y=np.array([[1e-8,-.0003,.5]])
    near=T.ttc(near_s,near_y,dt=.001)
    nr=independent_reference(near_s[0],near_y[0],max_step=1e-6)
    R=dict(method='piecewise constant RK4, dt<=1e-3, 40 bisections; independent DOP853 split at jumps',
           rigorous_enclosure=False,n_random_screen=2500,n_comparison=int(len(ss)),
           n_selected_D=int(len(selD)),classification_agreement=int(same.sum()),
           n_classification_mismatch=int((~same).sum()),
           max_abs_TTC_vs_DOP853=float(e[same].max()),
           max_abs_TTC_dt1e3_vs_dt5e4=float(np.max(np.abs(a['V']-b['V']))),
           max_abs_TTC_dt5e4_vs_dt25e5=float(np.max(np.abs(b['V']-c['V']))),
           max_snapshot_component_difference_dt1e3_vs_dt25e5=snaperr,
           max_semigroup_TTC_defect=float(semigroup.max()) if len(semigroup) else None,
           target_stops_immediately=bool(a['V'][-1]==0 and a['hit'][-1]),
           terminal_stops_immediately=bool(a['V'][-2]==0 and np.array_equal(a['y_final'][-2],yy[-2])),
           arbitrary_start_times_preserved=bool(np.array_equal(a['s'],ss)),
           within_step_minimum_case=dict(s=float(near_s[0]),y=near_y[0].tolist(),
               RK4_V=float(near['V'][0]),RK4_hit=bool(near['hit'][0]),
               DOP853_V=float(nr[0]),DOP853_hit=bool(nr[1]),DOP853_max_step=1e-6),
           caveat='Finite numerical checks do not certify classification at grazing or a uniform flow/event error.')
    assert R['arbitrary_start_times_preserved'] and R['target_stops_immediately'] and R['terminal_stops_immediately']
    assert R['n_classification_mismatch']==0
    assert R['within_step_minimum_case']['RK4_hit'] and R['within_step_minimum_case']['DOP853_hit']
    assert abs(R['within_step_minimum_case']['RK4_V']-R['within_step_minimum_case']['DOP853_V'])<1e-9
    assert R['max_abs_TTC_vs_DOP853'] < 1e-7
    save_npz_atomic(out/'reference_check_cases.npz',s=ss,y=yy,V_RK4=a['V'],V_DOP853=vv,hit_RK4=a['hit'],hit_DOP853=hh)
    (out/'reference_checks.json').write_text(json.dumps(R,indent=2))
    print(json.dumps(R,indent=2),flush=True)

if __name__=='__main__':main()
