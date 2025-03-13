import time
import numpy as np
import sympy as sp
from pockit.optimizer import ipopt
from pockit.radau import System, linear_guess

from color import *
import matplotlib.pyplot as plt

### Uncomment the following lines to use LaTeX in the plot
# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
#     }
# )


S = System(0)
P = S.new_phase(6, 3)
y_1, y_2, y_3, y_4, y_5, y_6 = P.x
u_1, u_2, u_3 = P.u
L = 5
I_theta = ((L - y_1) ** 3 + y_1**3) / 3 * sp.sin(y_5) ** 2
I_phi = ((L - y_1) ** 3 + y_1**3) / 3
P.set_dynamics([y_2, u_1 / L, y_4, u_2 / I_theta, y_6, u_3 / I_phi])
P.set_integral([1])
P.set_boundary_condition(
    [9 / 2, 0, 0, 0, np.pi / 4, 0], [9 / 2, 0, 2 * np.pi / 3, 0, np.pi / 4, 0], 0, None
)
P.set_phase_constraint([u_1, u_2, u_3], [-1, -1, -1], [1, 1, 1], True)
P.set_discretization(5, 3)
S.set_phase([P])
S.set_objective(P.I[0])


v = linear_guess(P)
S.objective(v.data)
S.gradient(v.data)
S.jacobian(v.data)
S.hessian_o(v.data)  # Pre-compile JIT functions
mesh_s = [P._mesh]
time_0 = time.time()
v, info = ipopt.solve(S, v, optimizer_options={"print_level": 0})
assert info["status"] == 0


max_iter = 10
for it in range(max_iter):
    if S.check_discontinuous(v):
        break
    v = S.refine_discontinuous(v, num_point_min=2, num_point_max=3, mesh_length_max=0.1)
    mesh_s.append(P._mesh)
    v, info = ipopt.solve(S, v, optimizer_options={"print_level": 0})
    assert info["status"] == 0
    print(it, info["obj_val"])

time_f = time.time()
print(f"Elapsed time: {time_f - time_0:.2f} s")

t_out = []
for i in range(len(mesh_s[-1]) - 1):
    t_out.append(
        np.linspace(v.t_f * mesh_s[-1][i], v.t_f * mesh_s[-1][i + 1], 10, endpoint=True)
    )
t_out = np.concatenate(t_out)
V_x = v.V_x(t_out)
V_u = v.V_u(t_out)


fig = plt.figure()
gs = fig.add_gridspec(3, 2)
ax = []
name = [rf"$\boldsymbol{{x}}_{i}$" for i in range(1, 7)]
for i in range(3):
    for j in range(2):
        ax.append(fig.add_subplot(gs[i, j]))
for i, ax_ in enumerate(ax):
    ax_.plot(t_out, V_x @ v.x[i], color=color_analytical, zorder=1)
    for j in range(len(v.t_x[::3])):
        ax_.axvline(
            v.t_x[3 * j], color=color_mesh, linestyle=":", zorder=0, linewidth=1
        )
    ax_.minorticks_on()
    ax_.grid(linestyle="--", linewidth="0.2")
    ax_.set_ylabel("{}".format(name[i]))
ax[4].set_xlabel("$t$")
ax[5].set_xlabel("$t$")
for i in range(3):
    ax[2 * i + 1].yaxis.tick_right()
    ax[2 * i + 1].yaxis.set_label_position("right")
for i in range(4):
    ax[i].set_xticklabels([])
fig.tight_layout()
fig.savefig("robot_arm_problem_state.pdf", bbox_inches="tight")


fig, ax_2 = plt.subplots()
ax = ax_2.twinx()
ax.plot(t_out, V_u @ v.u[0], color=color_foreground[4], label=r"$\boldsymbol{u}_1$")
ax.plot(t_out, V_u @ v.u[1], color=color_foreground[2], label=r"$\boldsymbol{u}_2$")
ax.plot(t_out, V_u @ v.u[2], color=color_foreground[0], label=r"$\boldsymbol{u}_3$")
ax.set_ylabel(r"$\boldsymbol{u}$")
ax.yaxis.tick_left()
ax.yaxis.set_label_position("left")
ax.minorticks_on()
ax.grid(linestyle="--", linewidth="0.2")
ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.1))
ax.set_ylim(-1.1, 1.1)
for i in range(len(mesh_s)):
    ax_2.scatter(
        v.t_f * mesh_s[i],
        np.zeros_like(mesh_s[i]) + i,
        marker="+",
        color="dimgrey",
        s=30,
    )
ax_2.set_yticks(range(0, len(mesh_s)))
ax_2.yaxis.tick_right()
ax_2.yaxis.set_label_position("right")
ax_2.set_xlabel("$t$")
ax_2.set_ylabel("iteration")
ax_2.grid(linestyle="--", linewidth="0.2", axis="x")
ax_2.legend(["mesh points"], loc="upper right", bbox_to_anchor=(1, 1.1))
ax_2.set_ylim(-0.2, 4.2)
fig.savefig("robot_arm_problem_control.pdf", bbox_inches="tight")
