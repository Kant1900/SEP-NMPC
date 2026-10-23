# SEP-NMPC 第一阶段：Python + acados + RK4 纯数学闭环仿真

本项目以 [SEP-NMPC: Safety Enhanced Passivity-Based Nonlinear Model Predictive Control for a UAV Slung Payload System](https://arxiv.org/abs/2603.08860) 为依据，**独立实现**无人机–刚性绳索–点质量负载的 10 状态平移动力学、NMPC、严格无源性约束、机体/负载 HOCBF、RK4 被控对象和结果可视化。代码不是作者官方源码。

这里暂不引入 Gazebo、PX4、ROS 或实际的飞控姿态闭环。`acados` 用于实时 OCP 求解；`CasADi + IPOPT` 是安装 acados 前可选的数学验证后备方案，**不能拿它的速度与论文的实时性能比较**。

## 0. 快速运行

进入本项目目录后：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 不依赖 acados 的烟雾测试：10 个节点，3.5 s 静态避障
python run_sim.py --solver ipopt --scenario static --nodes 10 --horizon 1.0 --steps 70 --output results/static_ipopt

# 检查数学一致性
python -m unittest discover -s tests -v
```

在 Linux + acados 配置完成后，可跑论文设定的 2 s / 40 节点（注意下面关于实时性和初始可行性的说明）：

```bash
python run_sim.py --solver acados --scenario hover --nodes 40 --horizon 2.0 --steps 100 --output results/hover_acados
python run_sim.py --solver acados --scenario static --nodes 40 --horizon 2.0 --steps 100 --output results/static_acados
python run_sim.py --solver acados --scenario dynamic --nodes 40 --horizon 2.0 --steps 100 --output results/dynamic_acados
```

上述 acados 指令是**待在已安装 acados 的机器上验证的运行入口**；本项目开发环境内没有 `acados_template`，不能据此声称 acados 实测成功。如果 SQP-RTI 因初始猜测出现困难，可尝试 `--full-sqp`，用完整 SQP 辅助调试（不再等同论文 SQP-RTI 的速度）。

输出目录内会生成：

- `data.npz`：时间、状态、控制、HOCBF、能量耗散裕量、净安全距离、求解耗时及是否中断。
- `trajectories.png`：机体与负载的俯视轨迹、位置、摆角和约束裕量。
- `clearances.png`：机体/负载分别对两个障碍的净安全距离曲线。

压缩包还附有 `example_results/static_ipopt/` 与 `example_results/dynamic_ipopt/`：这是在测试环境以 IPOPT、\(T=1\,\mathrm s\)、\(N=10\)、控制周期 0.05 s 运行 70 次所得的可视化与完整数值文件。可作为自检参考，**不代表论文的 acados/40 节点配置**。

## 1. 坐标、状态与动力学

采用惯性系 **z 轴向上**（ENU 类型平移轴）、负载在机体下方。状态定义为：

\[
q=[x,y,z,\alpha,\beta]^\top,\qquad
X=[q^\top,\dot q^\top]^\top\in\mathbb R^{10}.
\]

负载位置：

\[
p_L=p_Q+l
\begin{bmatrix}
\sin\alpha\cos\beta\\
\sin\beta\\
-\cos\alpha\cos\beta
\end{bmatrix},\qquad p_Q=[x,y,z]^\top.
\]

设 \(J_L=\partial p_L/\partial q\)，质量矩阵以及重力势能为：

\[
M(q)=m_Q\operatorname{diag}(1,1,1,0,0)+m_LJ_L^\top J_L,
\]

\[
U_g=(m_Q+m_L)gz-m_Lgl\cos\alpha\cos\beta.
\]

代码由拉格朗日结构**自动推导**惯性、科氏与重力项：

\[
T=\tfrac12\dot q^\top M(q)\dot q,\quad
c(q,\dot q)=\frac{\partial[M(q)\dot q]}{\partial q}\dot q-\frac{\partial T}{\partial q},
\]

\[
M(q)\ddot q+c(q,\dot q)+\nabla_q U_g=
\begin{bmatrix}F_x&F_y&F_z&0&0\end{bmatrix}^\top.
\]

这里的质量、绳长默认取论文实验值：\(m_Q=1.5\,\mathrm{kg}\)、\(m_L=0.20\,\mathrm{kg}\)、\(l=0.5\,\mathrm m\)。机体姿态动力学没有并入这 10 个状态；假设已有足够快的内环能将三维期望合力跟踪为实际推力方向。

### 关于文中公式与本实现的两处重要区别

1. **悬停重力补偿**：文章 (1)、(2) 的力 \(F\) 包含对抗重力的悬停力，而 (8)–(10) 写出 \(\dot V=v^\top u_a\) 时必须把 \((m_Q+m_L)ge_z\) 从输入平衡点移除。因此我们明确规定

   \[
   F=u_a-K(p_Q-p_d)+(m_Q+m_L)ge_z.
   \]

   否则原样使用 \(u_a=F+Ke\) 不能在物理悬停点使 \(u_a=0\)。**此处是为能量一致性作出的解释性修正**，不是未经改动的论文代码。

2. **负载 HOCBF 的加速度输入映射**：论文第 III-D 节称机体和负载的 \(G_v\) 相同。但完整刚性绳索约束下，通常

   \[
   \frac{\partial\ddot p_Q}{\partial u_a}\ne
   \frac{\partial\ddot p_L}{\partial u_a}.
   \]

   本代码对两者分别通过完整的 \(\ddot q\) 计算 \(\ddot p_j\)，再得到各自的 HOCBF；这与直接照抄共享输入映射不同。单元测试会验证这一区别。

## 2. NMPC 优化

\[
\begin{aligned}
\min_{X_{0:N},u_{0:N-1}}\quad
&\frac12\sum_{k=0}^{N-1}\Delta t\big(\|X_k-X_d\|_Q^2+\|u_{a,k}\|_R^2\big)
+\frac12\|X_N-X_d\|_{Q_f}^2\\
\text{s.t.}\quad &X_{k+1}=\operatorname{RK4}(X_k,u_{a,k}),\\
&X_0=X_{\rm measured},\\
&u_{a,k}\in\mathcal U,\quad |\alpha_k|,|\beta_k|\le65^\circ,\\
&F_k\in[F_{\min},F_{\max}],\\
&\mathrm{passivity}_k\ge0,\quad \psi_{2,i,j,k}\ge0,\\
&h_{i,j,k}\ge0,\quad\psi_{1,i,j,k}\ge0.
\end{aligned}
\]

增加离散节点处的 \(h\)、\(\psi_1\) 约束，是对连续时间安全集条件的数值保护，不保证节点间一定安全。代价权重、外力/摆角上界见 `sep_nmpc/solvers.py`。可通过 `--nodes`、`--horizon` 设置预测离散；例如论文为 \(N=40,T=2\,\mathrm s\)，因此预测节点间距是 0.05 s。

## 3. 无源性约束

在目标位置**固定**时，带势能整形的存储函数为

\[
V(X)=\frac12\dot q^\top M(q)\dot q
+m_Lgl(1-\cos\alpha\cos\beta)
+\frac12(p_Q-p_d)^\top K(p_Q-p_d).
\]

在以上补偿定义下，代码可数值核验精确功率等式：

\[
\dot V=\dot p_Q^\top u_a.
\]

通过如下严格无源性约束施加耗散：

\[
\boxed{-\dot p_Q^\top u_a-\rho\|\dot p_Q\|^2-\epsilon\|u_a\|^2\ge0.}
\]

注意：当 \(\dot p_Q=0\) 时，\(\epsilon>0\) 迫使 \(u_a=0\)，**但实际力并不一定为零**，因为 \(F\) 还包含 \(-Ke\) 与重力补偿。离障碍太近、初始期望位置不合适时，**passivity 与 HOCBF 的硬约束可能相互冲突，使优化在初始时刻即不可行**；这不是靠调求解器精度就能消除的问题。

## 4. HOCBF（机体与负载各自一份）

障碍 \(i\) 有中心 \(p_{o,i}(t)\)、速度 \(v_{o,i}\)、加速度 \(a_{o,i}\)。默认使用无限高的水平圆柱障碍（只计 xy 平面距离），适合直立障碍；`--geometry sphere` 切到三维球形距离：

\[
r_{i,j}=p_j-p_{o,i},\quad
h_{i,j}=\|r_{i,j}\|^2-(R_i+r_j+\Delta)^2,
\]

\[
\dot h_{i,j}=2r_{i,j}^\top(v_j-v_{o,i}),
\]

\[
\ddot h_{i,j}=2\|v_j-v_{o,i}\|^2
+2r_{i,j}^\top(a_j-a_{o,i}),
\]

\[
\psi_1=\dot h+\kappa_1h,\qquad
\boxed{\psi_2=\ddot h+(\kappa_1+\kappa_2)\dot h+\kappa_1\kappa_2h\ge0.}
\]

\(a_j\) 是从全耦合状态方程得到的机体/负载加速度。由于控制仅作用到加速度，\(\psi_2\) 对 \(u_a\) 保持**仿射**（测试覆盖），可以在 SQP 线性化后形成 QP 约束。用 `h>=0` 和 `psi1>=0` 附加节点保护并不能消除初始可行性要求，也不是实机的碰撞证明。

## 5. 闭环与结果

每个控制周期：

1. 更新各预测节点的目标位置和静态/动态障碍中心、速度与加速度；
2. 将当前状态设置为 OCP 的初始条件，并用上一次求解轨迹热启动；
3. 求解控制序列，仅施加首个 \(u_a\)；
4. 用独立的 RK4 被控对象仿真器推进 \(\Delta t_{\rm ctrl}\)，重新反馈状态；
5. 存储位置、摆角、机体/负载净间距及约束裕量；求解失败则**停止并记录**，不伪造可行轨迹。

`--dt` 是控制周期与外部物理对象积分周期，和 MPC 的 \(T/N\) **不是同一个概念**。默认 `--dt 0.05` 是 20 Hz，论文真实机报告的是 100 Hz；若 acados 在本机足够快，可用 `--dt 0.01` 模拟 100 Hz。

可运行的场景：

- `hover`：无近距离障碍的定点迁移，先验证动力学/约束；
- `static`：两个静态障碍，观察机体与吊挂负载共同绕行；
- `dynamic`：第二个障碍以恒定速度移动，计算其对 \(\dot h,\ddot h\) 的影响。

所有障碍及目标的默认坐标在 `run_sim.py` 的 `get_scene()`；所有质量、半径及 HOCBF/无源性参数在 `sep_nmpc/model.py` 的 `Parameters`。测试较困难的新场景时，应先验证初值 \(h\ge0,\psi_1\ge0\)，再确认初值存在同时满足 \(\psi_2\ge0\) 与无源性的不饱和输入；否则严格硬约束没有可行解。

## 6. acados 安装（Ubuntu / Python）

参照 [acados 官方安装文档](https://docs.acados.org/installation/) 与 [官方 Python 入门例程](https://github.com/acados/acados/blob/main/examples/acados_python/getting_started/minimal_example_ocp.py)。概略过程：

```bash
# 在自己有写权限的目录中操作（此处只是示例路径）
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=OFF ..
make -j4
cd ..

# 在你执行仿真的同一 Python 虚拟环境中安装接口
python -m pip install -e interfaces/acados_template

# 依据实际 clone 位置更新环境变量
export ACADOS_SOURCE_DIR="$(pwd)"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
python -c 'import acados_template; print("acados template OK")'
```

如遇到你之前遇到过的 `Rendering file main.in.c failed`，先检查 acados 的 `bin/t_renderer` 是否存在且具有执行权限，并且与安装的接口/模板版本兼容；参照官方安装文档的 Tera Renderer 步骤，不要把 Python 文件直接当 shell 脚本执行。

## 7. 实现边界及验证状态

- **已经本机验证**：CasADi 动力学构造；悬停平衡、质量矩阵、能量功率恒等式、负载/机体输入映射、HOCBF 仿射性、RK4 等单元测试；IPOPT 短时定点迁移和静态障碍闭环。
- **测试记录（IPOPT，70 步，3.5 s，10 节点，1 s 预测）**：静态障碍最终位置误差 0.0041 m、最大摆角 30.06°、机体/负载最小净安全距离 0.218 m、平均求解约 333 ms；移动障碍最终位置误差 0.0037 m、最大摆角 18.46°、最小净安全距离 0.169 m、平均求解约 302 ms。这是单次演示的离散采样指标，不是安全性统计保证。
- **尚未本机验证**：acados 模板生成 / RTI 实测、论文 100 Hz 与 40 节点性能；实物姿态内环、估计误差、柔性绳索、松绳和接地。
- **非作者逐字节复现**：本文实现了基于论文方法的可运行第一阶段基线，修正了重力平衡点并采用更严格的负载加速度表达，同时自行选取场景、权重、饱和限幅与初值。结果不能宣称与论文表格数字一致。
- **安全限制**：HOCBF 连续时间前向不变性要求正确建模、初始条件与求解始终可行；单点 RK4 采样、SQP-RTI 近似、硬约束冲突、模型失配和传感延迟都会影响实际安全性。

## 文件结构

```text
sep_nmpc_stage1/
  run_sim.py             # 主程序、场景、RK4 闭环、绘图及 .npz 记录
  sep_nmpc/model.py      # 10 状态耦合动力学、无源性能量与 HOCBF
  sep_nmpc/solvers.py    # acados SQP-RTI 与 IPOPT 多重打靶 OCP
  sep_nmpc/__init__.py
  tests/test_model.py    # 不依赖 acados 的数学单元测试
  requirements.txt
  README.md
  example_results/      # 经运行验证的静态/移动障碍演示图及 .npz 数据
```
