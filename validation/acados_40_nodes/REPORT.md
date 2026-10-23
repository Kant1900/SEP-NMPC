# SEP-NMPC：acados 40 节点验证

已完成 acados 安装、模板渲染、C 代码生成、HPIPM 求解与 40 节点闭环运行。当前实现不能认定为全程约束可行的论文复现。

## 配置

- 仓库 HEAD：b303692f55e0fb233d7c55dd9ddf66588335d2df；源码、测试、requirements 均未修改。
- acados：5b74695a51c09637664318f7c9b2f1ee2efa05b7；递归子模块使用记录的提交；Tera Renderer v0.2.1，从官方 HTTPS 发布地址安装。
- Python 3.12.14，CasADi 3.8.1；HPIPM、BLASFEO GENERIC、Release、无 OpenMP。
- N=40，T=2 秒，预测步长 0.05 秒，控制周期 0.05 秒，100 步（5 秒）。Gauss–Newton、ERK4、SQP_RTI。
- 所有 7 项数学测试通过，pip check 通过；安装脚本重复执行通过。

## 原始 SQP-RTI 结果

| 场景 | 完成步数 | 最终位置误差 m | 最小采样净间距 m | 最小施加输入无源裕量 | 无源裕量 <−0.001 的步数 | 最小采样 HOCBF |
|---|---:|---:|---:|---:|---:|---:|
| hover | 100/100 | 0.0000157 | 140.107（远离障碍） | −0.8260 | 3 | 177516 |
| static | 100/100 | 0.000301 | 0.2121 | −2.2562 | 7 | −1.31e−7 |
| dynamic | 100/100 | 0.000671 | 0.1744 | −1.1598 | 5 | −3.01e−6 |

所有周期 NLP 返回 status=0，但这不表示非线性约束全部满足。上述无源性违反远大于数值容差；不能用完成运行或到达目标代替约束验证。采样 HOCBF 的微小负值应与显著的无源性违反区别看待。采样 psi1、实际力与输入边界、摆角边界检查通过。

预测轨迹还出现更大的违反：static 的最小无源裕量 −3.465、最小 HOCBF −0.7555；dynamic 分别 −2.619、−0.1573。检查包括路径约束、终端 h/psi1、输入/力/摆角边界和 acados 动力学残差。

最终 [stationarity, equality, inequality, complementarity] 残差：

- hover：[4.18e−6, 1.33e−13, 1.25e−12, 1.0e−7]
- static：[1.16e−6, 2.70e−12, 6.66e−14, 1.0e−7]
- dynamic：[1.59e−6, 7.79e−12, 2.21e−12, 1.0e−7]

所有四项残差从第 96、79、82 步起分别持续低于 1e−5；这是随闭环接近目标的残差下降，不是每个 MPC 问题都求解收敛。最大不等式残差分别为 1.363、3.465、2.619。

求解器内部平均耗时约 12.22、14.04、12.83 ms，最大值约 15.13、23.60、17.04 ms。三个场景并发执行，使用 GENERIC 构建，Python诊断另有开销：这些数据仅作本次运行记录，不能用来认证实时截止时间，也没有证明论文 100 Hz。

## 完整 SQP 对照与诊断

三个场景用原仓库 `--full-sqp` 运行时均在 k=0 返回 status=2（达到 100 次 NLP 迭代上限），没有完成控制步。首步不等式残差分别为 0.7117、4.6987、0.6401。不能将这些运行记为成功。

静态场景额外使用受支持的外部选项覆盖：MERIT_BACKTRACKING、Levenberg–Marquardt=1e−3、最多 300 次迭代。仍在 k=0 返回 status=2，不等式残差 0.06269；该实验没有放松约束，也没有修改仓库。

结构性问题：无源裕量为 g=−vᵀu−ρ‖v‖²−ε‖u‖²。初始 v=0 且 ε>0 时，精确可行性要求 u=0；但在线性化初值 v=u=0，g 及其梯度均为零，线性化约束退化为 0≥0。首轮 QP 因而无法通过该行限制非零输入。这解释了冷启动 RTI 的风险；完整 SQP 也未在本次配置中解决，不能据此证明原非线性问题不可行。

建议后续算法工作：设计满足动力学与无源约束的初始轨迹，分析退化约束，验证求解失败/约束违反时的处理策略。若修改约束或引入松弛，需明确其与论文严格无源性结论的差别。本次保留原始实现以便复核。

## 复跑

```bash
cd /workspace/SEP-NMPC
source /workspace/sep-nmpc-setup/acados-env.sh
python -m unittest discover -s tests -v
python validation/acados_40_nodes/diagnose.py --solver acados --scenario static --nodes 40 --horizon 2.0 --steps 100 --output results/recheck_static_acados
```

将 static 替换为 hover 或 dynamic；增加 `--full-sqp` 运行默认完整 SQP。诊断文件与常规 data.npz、trajectories.png、clearances.png 一起输出。诊断通过外部包装器调用未修改的仓库控制器；`get_residuals(recompute=True)` 用于标准 RTI，不能照搬到 AS-RTI。

本次云环境安装脚本保留在云环境配置中；通用依赖安装见仓库 README 的 acados 安装章节。不需要后台服务。

归档运行数据在本目录的 `runs/onboarding_{hover,static,dynamic}_acados/`；完整 SQP 在对应 `_sqp` 目录。原始运行日志在 `logs/`。本地生成的 C 编译产物未归档。`summary.json` 保存汇总数据，`rti_diagnostics.png` 展示约束裕量及残差。

## 复现边界

当前仓库是独立数学实现，README 已说明重力补偿、负载输入映射及场景/权重与逐字复现的差异。随后已取得用户上传的论文全文并完成公式与参数核对，详见 PAPER_AUDIT.md 和 paper_audit.json。此前 arxiv.org 网络403不再妨碍原文核对；原文核对使用用户上传的 PDF。仍未验证实物系统。此次完成的是现有实现的40节点acados验证及论文一致性审计。
