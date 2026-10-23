"""Observe the unmodified repository controller; do not tune its OCP."""
import sys, json, runpy, os
from pathlib import Path
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from sep_nmpc import solvers
if os.environ.get('SEP_SQP_GLOBALIZATION') == '1':
    import acados_template
    original_factory=acados_template.AcadosOcpSolver
    def configured(ocp, *args, **kwargs):
        ocp.solver_options.globalization='MERIT_BACKTRACKING'
        ocp.solver_options.nlp_solver_max_iter=300
        ocp.solver_options.levenberg_marquardt=1e-3
        return original_factory(ocp, *args, **kwargs)
    acados_template.AcadosOcpSolver=configured
rows=[]
base=solvers.AcadosController
class Observed(base):
    def solve(self, x0, p):
        row={'step':len(rows)}
        try:
            result=super().solve(x0,p)
            row['status']=int(result[1])
            return result
        except RuntimeError as exc:
            row['error']=str(exc)
            row['status']=int(self.solver.status)
            raise
        finally:
            row['residuals_stat_eq_ineq_comp']=self.solver.get_residuals(recompute=True).tolist()
            row['qp_status']=np.asarray(self.solver.get_stats('qp_stat')).tolist()
            row['time_tot']=float(self.solver.get_stats('time_tot'))
            xs=np.array([self.solver.get(k,'x') for k in range(self.settings.nodes+1)])
            us=np.array([self.solver.get(k,'u') for k in range(self.settings.nodes)])
            checks=[self.m.checks(xs[k],us[k],p[k]) for k in range(self.settings.nodes)]
            row['horizon_min_passivity']=min(float(c[0]) for c in checks)
            row['horizon_min_cbf']=min(float(np.min(c[1])) for c in checks)
            row['horizon_min_h']=min(float(np.min(c[2])) for c in checks)
            row['horizon_min_psi1']=min(float(np.min(c[3])) for c in checks)
            force=np.array([np.array(c[6]).ravel() for c in checks])
            row['horizon_force_violation']=float(max(0,np.max(solvers.FORCE_LOWER-force),np.max(force-solvers.FORCE_UPPER)))
            row['horizon_input_violation']=float(max(0,np.max(solvers.INPUT_LOWER-us),np.max(us-solvers.INPUT_UPPER)))
            row['horizon_angle_violation']=float(max(0,np.max(abs(xs[:-1,3:5]))-np.deg2rad(self.m.cfg.max_swing_deg)))
            terminal=self.m.checks(xs[-1],np.zeros(3),p[-1])
            row['terminal_min_h']=float(np.min(terminal[2]))
            row['terminal_min_psi1']=float(np.min(terminal[3]))
            rows.append(row)
solvers.AcadosController=Observed
out=Path(sys.argv[sys.argv.index('--output')+1]); out.mkdir(parents=True,exist_ok=True)
try:
    runpy.run_path(str(REPO_ROOT / 'run_sim.py'),run_name='__main__')
finally:
    (out/'diagnostics.json').write_text(json.dumps(rows,indent=2))
