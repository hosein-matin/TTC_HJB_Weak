"""Reproduce revised experiments from unchanged supplied checkpoints.

This entry point performs evaluation only. It never retrains the models.
Run with --reuse-grids to reuse already computed construction coefficients.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reuse-grids',action='store_true')
    args=p.parse_args()
    root=Path(__file__).resolve().parent
    env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='2';env['OMP_NUM_THREADS']='2'
    jobs=[['reference_checks.py'],['evaluate_models.py'],['construction_experiment.py'],
          ['iteration_experiment.py'],['figures.py']]
    if args.reuse_grids:jobs[2].append('--reuse-grids')
    for job in jobs:
        print('Running',*job,flush=True)
        subprocess.run([sys.executable,*job],cwd=root,env=env,check=True)
    subprocess.run([sys.executable,'verify_results.py'],cwd=root,env=env,check=True)


if __name__=='__main__':main()
