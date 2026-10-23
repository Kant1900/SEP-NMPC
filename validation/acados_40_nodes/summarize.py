from pathlib import Path
import sys,json
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from run_sim import get_scene,parameters_at
from sep_nmpc.model import make_model, Parameters
from sep_nmpc.solvers import FORCE_LOWER,FORCE_UPPER,INPUT_LOWER,INPUT_UPPER
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent / 'runs'
m=make_model(Parameters())
summary={}
fig,axs=plt.subplots(3,1,figsize=(10,10),constrained_layout=True)
for s in ('hover','static','dynamic'):
 out=root/f'onboarding_{s}_acados'
 rows=json.loads((out/'diagnostics.json').read_text())
 d=np.load(out/'data.npz'); scene=get_scene(s)
 assert len(rows)==100 and not d['failed'] and d['states'].shape==(101,10)
 for name in ('states','controls','passivity','cbf','h','clearance','solve_ms'):
  assert np.isfinite(d[name]).all()
 checks=[m.checks(x,u,parameters_at(scene,t)) for x,u,t in zip(d['states'][:-1],d['controls'],d['t'][:-1])]
 force=np.array([np.array(c[6]).ravel() for c in checks])
 res=np.array([r['residuals_stat_eq_ineq_comp'] for r in rows])
 tm=np.array([r['time_tot'] for r in rows])*1000
 good=np.all(res<=1e-5,axis=1)
 suffix=next((k for k in range(100) if good[k:].all()),None)
 summary[s]={
 'completed':len(rows),'statuses':sorted(set(r['status'] for r in rows)),
 'final_position_error_m':float(np.linalg.norm(d['states'][-1,:3]-d['target'])),
 'min_sampled_clearance_m':float(d['clearance'].min()),
 'min_sampled_passivity':float(d['passivity'].min()),
 'passivity_violations_below_minus_1e_minus_3':int(np.sum(d['passivity'] < -1e-3)),
 'min_sampled_cbf':float(d['cbf'].min()),
 'min_sampled_psi1':min(float(np.min(c[3])) for c in checks),
 'force_violation':float(max(0,np.max(FORCE_LOWER-force),np.max(force-FORCE_UPPER))),
 'input_violation':float(max(0,np.max(INPUT_LOWER-d['controls']),np.max(d['controls']-INPUT_UPPER))),
 'max_swing_deg':float(np.max(abs(np.rad2deg(d['states'][:,3:5])))),
 'horizon_min_passivity':min(r['horizon_min_passivity'] for r in rows),
 'horizon_min_cbf':min(r['horizon_min_cbf'] for r in rows),
 'horizon_max_force_violation':max(r['horizon_force_violation'] for r in rows),
 'horizon_max_angle_violation':max(r['horizon_angle_violation'] for r in rows),
 'residuals_max_stat_eq_ineq_comp':res.max(axis=0).tolist(),
 'residuals_final_stat_eq_ineq_comp':res[-1].tolist(),
 'all_residuals_under_1e_minus_5_from_step':suffix,
 'internal_time_mean_ms':float(tm.mean()),'internal_time_p95_ms':float(np.percentile(tm,95)),
 'internal_time_max_ms':float(tm.max())}
 axs[0].plot(d['t'][:-1],d['passivity'],label=s)
 axs[1].semilogy(d['t'][:-1],np.maximum(res[:,2],1e-15),label=s)
 axs[2].semilogy(d['t'][:-1],np.maximum(res[:,0],1e-15),label=s)
for ax,title in zip(axs,['Applied-control passivity margin (must be >= 0)','NLP inequality residual','NLP stationarity residual']):
 ax.set_title(title);ax.set_xlabel('Simulation time [s]');ax.grid(alpha=.3);ax.legend()
axs[0].axhline(0,color='k',linewidth=.7)
fig.savefig(Path(__file__).with_name('rti_diagnostics.png'),dpi=150)
Path(__file__).with_name('summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
