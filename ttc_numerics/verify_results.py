"""Check archive integrity and recompute the reported error columns.

This is a reproducibility check, not a uniform numerical error certificate.
"""
from pathlib import Path
import argparse
import json
import zipfile
import numpy as np
import construct as C


def error_metrics(w,v):
    e=np.asarray(w)-np.asarray(v)
    return float(np.max(abs(e))),float(np.sqrt(np.mean(e*e)))


def close_pair(actual,expected):
    if not np.allclose(actual,expected,rtol=1e-11,atol=1e-12):
        raise AssertionError((actual,expected))


def verify(outdir='out'):
    out=Path(outdir);checked={}
    for path in sorted(out.glob('*.npz')):
        with zipfile.ZipFile(path) as archive:
            bad=archive.testzip()
            if bad is not None:raise IOError(f'{path}: corrupt member {bad}')
        with np.load(path,allow_pickle=False) as data:
            checked[path.name]={key:list(data[key].shape) for key in data.files}
    E=json.loads((out/'evaluation_results.json').read_text())
    with np.load(out/'test_holdout.npz',allow_pickle=False) as d:
        for tag in ['mse','res']:
            for selected,suffix in [(np.ones(len(d['V']),bool),'all'),(d['inD'],'D')]:
                actual=error_metrics(d['pred_'+tag][selected],d['V'][selected])
                close_pair(actual,(E[tag]['linf_'+suffix],E[tag]['rmse_'+suffix]))
    R=json.loads((out/'construction_results.json').read_text())
    with np.load(out/'construction_queries.npz',allow_pickle=False) as q:
        for tag in ['mse','res']:
            expected=R['unchanged_learned_models'][tag]['initial']
            close_pair(error_metrics(q[tag+'_initial'],q['V_initial']),
                       (expected['maximum'],expected['rmse']))
        for h,item in R['meshes'].items():
            suffix=h.replace('.','p')
            grid=C.TensorGrid.load(out/f'construction_grid_{suffix}.npz')
            w=grid.evaluate(q['z_initial'],k=item['realization']['square_layers'])
            close_pair(error_metrics(w,q['V_initial']),
                       (item['initial_relu']['maximum'],item['initial_relu']['rmse']))
    report={'archives_checked':checked,'reported_error_metrics_recomputed':True,
            'uniform_numerical_certificate':False}
    (out/'reproduction_checks.json').write_text(json.dumps(report,indent=2))
    print(f'Validated {len(checked)} complete NPZ archives; reported model and construction error columns agree.')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--outdir',default='out')
    a=p.parse_args();verify(a.outdir)
