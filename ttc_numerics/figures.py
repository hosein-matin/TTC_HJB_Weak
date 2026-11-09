"""Regenerate revised figures after the evaluation, construction and iteration runs."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import ttc as T
from model_io import load_model


def run(outdir='out', checkpoint_dir='provenance/checkpoints'):
    out=Path(outdir)
    plt.rcParams.update({'font.size':8,'axes.titlesize':8,'axes.labelsize':8,
        'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,
        'figure.dpi':220,'pdf.fonttype':42,'ps.fonttype':42})
    R=json.loads((out/'evaluation_results.json').read_text())
    I=json.loads((out/'iteration_results.json').read_text())
    S=np.load(out/'construction_slice.npz',allow_pickle=False)
    ss,zz=.2,.6
    xi=np.linspace(1e-6,.8,241)
    lam=np.linspace(zz-T.VMAX,zz,241)
    xx,ll=np.meshgrid(xi,lam,indexing='xy')
    z=np.column_stack([np.full(xx.size,ss),xx.ravel(),ll.ravel(),np.full(xx.size,zz)])
    r=T.ttc(z[:,0],z[:,1:])
    w=load_model('res',directory=checkpoint_dir)(z)
    v=r['V'].reshape(xx.shape)
    error=np.abs(w-r['V']).reshape(xx.shape)
    chosen=T.in_D(r).reshape(xx.shape)
    nohit=(~r['hit']).reshape(xx.shape)
    info=dict(s=ss,zeta=zz,n_slice_D=int(chosen.sum()),
        maximum_error_full=float(error.max()),maximum_error_D=float(error[chosen].max()),
        construction_slice=dict(s=float(S['s']),zeta=float(S['zeta']),mesh=.01,
        maximum_error=float(np.max(abs(S['relu_0p01']-S['V'])))))
    (out/'slice_info.json').write_text(json.dumps(info,indent=2))
    fig,axs=plt.subplots(1,3,figsize=(7.0,2.25),layout='constrained')
    im=axs[0].pcolormesh(xx,ll,v,cmap='viridis',vmin=0,vmax=1-ss,
        shading='auto',rasterized=True)
    fig.colorbar(im,ax=axs[0],fraction=.048,pad=.02)
    axs[0].set_title(r'(a) Reference TTC, $s=.2,\ \zeta_L=.6$')
    im=axs[1].pcolormesh(xx,ll,np.maximum(error,1e-6),cmap='magma',
        norm=LogNorm(vmin=1e-4,vmax=1),shading='auto',rasterized=True)
    fig.colorbar(im,ax=axs[1],fraction=.048,pad=.02)
    axs[1].set_title(r'(b) Residual-trained $|W-V|$')
    for ax in axs[:2]:
        ax.contour(xx,ll,nohit.astype(float),levels=[.5],colors='white',
            linewidths=.7,linestyles='--')
        ax.contour(xx,ll,chosen.astype(float),levels=[.5],colors='#e53935',linewidths=.8)
    cx,cl=np.meshgrid(S['xi'],S['lam'],indexing='xy')
    ce=np.abs(S['relu_0p01']-S['V']).reshape(cx.shape)
    im=axs[2].pcolormesh(cx,cl,np.maximum(ce,1e-10),cmap='magma',
        norm=LogNorm(vmin=1e-7,vmax=1e-4),shading='auto',rasterized=True)
    fig.colorbar(im,ax=axs[2],fraction=.048,pad=.02)
    axs[2].set_title(r'(c) Constructed ReLU, $s=.2,\ \zeta_L=.4$')
    axs[2].set_xticks([.05,.08,.12]);axs[2].set_yticks([-1.4,-1.3,-1.2,-1.1])
    for ax in axs:
        ax.set_xlabel(r'$\Xi$');ax.set_ylabel(r'$\Lambda$')
    fig.savefig(out/'fig_slice.pdf');fig.savefig(out/'fig_slice.png');plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(3.5,2.0),layout='constrained')
    ax=axs[0]
    profiles=I['grids'][-1]['initializations']['zero']['profiles']
    for k,color,ls in [(5,'#0072B2','-'),(10,'#D55E00','--'),(20,'#009E73',':')]:
        p=profiles[str(k)]
        y=np.array([np.nan if a is None else a for a in p['rmse']])
        ax.semilogy(p['centers'],y,color=color,ls=ls,lw=1.2,label=f'$k={k}$')
        if k<20:ax.axvline(k*.05,color=color,ls=':',lw=.6,alpha=.5)
    ax.set_xlim(.1,.6);ax.set_ylim(7e-4,.5)
    ax.set_xticks([.1,.3,.5]);ax.set_xlabel('Reference TTC')
    ax.set_ylabel('Binned RMSE');ax.set_title('(a) Interpolated sweeps')
    ax.legend(frameon=False,loc='upper left',handlelength=1.5)
    ax=axs[1]
    for tag,mk,color,label in [('mse','o','#0072B2','MSE'),('res','s','#D55E00','MSE + residual')]:
        hs=sorted(float(h) for h in R[tag]['cert'])
        b=[R[tag]['cert'][str(h)]['bound_max'] for h in hs]
        d=[R[tag]['cert'][str(h)]['delta'] for h in hs]
        ax.loglog(hs,b,color=color,marker=mk,ms=3,lw=1.1,label=label)
        ax.loglog(hs,d,color=color,marker=mk,mfc='none',ms=3,lw=.9,ls='--')
        ax.axhline(R[tag]['linf_D'],color=color,lw=.8,ls=':')
    ax.set_xticks([.025,.05,.1,.2],labels=['.025','.05','.1','.2'])
    ax.minorticks_off();ax.set_ylim(.007,.5);ax.set_xlabel('$h$')
    ax.set_title('(b) Sampled chains')
    ax.legend(frameon=False,loc='upper right',handlelength=1.4,fontsize=5.8)
    fig.savefig(out/'fig_iter_cert.pdf');fig.savefig(out/'fig_iter_cert.png');plt.close(fig)
    print(json.dumps(info,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--outdir',default='out')
    p.add_argument('--checkpoint-dir',default='provenance/checkpoints')
    a=p.parse_args();run(a.outdir,a.checkpoint_dir)
