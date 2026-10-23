"""Numerical consistency probes of the displayed PDF equations (not author code)."""
import sys,json
from pathlib import Path
import casadi as ca
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from sep_nmpc.model import Parameters,make_model
from run_sim import get_scene,parameters_at
m=make_model();cfg=m.cfg;q=m.x[:5];dq=m.x[5:];a,b=q[3],q[4];ad,bd=dq[3],dq[4];ml,l,g=cfg.m_l,cfg.length,cfg.g
Mc=ml*l*ca.vertcat(ca.horzcat(ca.cos(a)*ca.cos(b),-ca.sin(a)*ca.sin(b)),ca.horzcat(0,ca.cos(b)),ca.horzcat(ca.sin(a)*ca.cos(b),ca.cos(a)*ca.sin(b)))
Mp=ml*l*l*ca.diag(ca.vertcat(ca.cos(b)**2,1))
M=ca.vertcat(ca.horzcat((cfg.m_q+ml)*ca.SX.eye(3),Mc),ca.horzcat(Mc.T,Mp))
C=ca.SX.zeros(5,5)
items={(0,3):-ml*l*ad*ca.sin(a)*ca.cos(b)-ml*l*bd*ca.cos(a)*ca.sin(b),(0,4):-ml*l*ad*ca.cos(a)*ca.sin(b)-ml*l*bd*ca.sin(a)*ca.cos(b),(1,4):-ml*l*bd*ca.sin(b),(2,3):ml*l*ad*ca.cos(a)*ca.cos(b)-ml*l*bd*ca.sin(a)*ca.sin(b),(2,4):-ml*l*ad*ca.sin(a)*ca.sin(b)+ml*l*bd*ca.cos(a)*ca.cos(b),(3,4):-ml*l*l*ad*ca.cos(b)*ca.sin(b)}
for (i,j),val in items.items(): C[i,j]=val;C[j,i]=-val
# Retain the explicitly printed nonzero diagonal despite its conflict with c_ij=-c_ji.
C[3,3]=-ml*l*l*bd*ca.cos(b)*ca.sin(b)
Mdot=ca.reshape(ca.jacobian(ca.reshape(M,25,1),q)@dq,5,5)
power=ca.mtimes([dq.T,.5*Mdot-C,dq])
G=ca.vertcat(0,0,(cfg.m_q+ml)*g,ml*g*l*ca.sin(a)*ca.cos(b),ml*g*l*ca.cos(a)*ca.sin(b))
probe=ca.Function('paper_probe',[m.x,m.u,m.p],[M-m.mass,G-m.gravity,C@dq-m.coriolis_times_velocity,power,ca.jacobian(m.a_quad,m.u)-ca.jacobian(m.a_load,m.u)])
x=np.array([.4,-.6,1.2,.23,-.16,.8,-.3,.13,.5,-.2]);p=parameters_at(get_scene('static'),0);u=np.array([-1.2,.4,-.8])
v=probe(x,u,p)
r={'sample_state':x.tolist(),'mass_max_abs_difference':float(np.max(abs(v[0]))),'gravity_max_abs_difference':float(np.max(abs(v[1]))),'paper_printed_Cdq_minus_lagrangian_Cdq':np.array(v[2]).ravel().tolist(),'paper_eq9_power_residual':float(v[3]),'quad_minus_load_acceleration_input_jacobian':np.array(v[4]).tolist(),'jacobian_difference_frobenius_norm':float(np.linalg.norm(v[4])),'missing_hover_compensation_power_term_at_sample':float(-(cfg.m_q+ml)*g*x[7])}
print(json.dumps(r,indent=2));open(Path(__file__).with_name('paper_audit.json'),'w').write(json.dumps(r,indent=2))
