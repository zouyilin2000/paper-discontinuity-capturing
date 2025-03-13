import numpy as np
from pockit.optimizer import ipopt
from pockit.radau import System, linear_guess
from pockit.radau.discretization import xw_lgr, P_lgr
from pockit.base.phasebase import _find_root_discontinuous

from color import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

### Uncomment the following lines to use LaTeX in the plot
# plt.rcParams.update(
#     {
#         "text.usetex": True,
#     }
# )


def to_scientific_notation(x):
    base_str = r"{:.2g}".format(x)
    if "e" in base_str:
        base, exponent = base_str.split("e")
        return rf"${base} \times 10^{{{int(exponent)}}}$"
    else:
        return f"${base_str}$"


find_root = lambda x: _find_root_discontinuous(x, P_lgr)

S = System(0)
P = S.new_phase(["x", "v"], ["u"])
x, v = P.x
(u,) = P.u
P.set_dynamics([v, u])
P.set_integral([1])
P.set_phase_constraint([u], [-1], [1], [True])
P.set_boundary_condition([1, 0], [0, 0], 0, None)
P.set_discretization(1, 1)
S.set_phase([P])
S.set_objective(P.I[0])

n_node = np.arange(2, 16)
eps_m = np.geomspace(1e-5, 0.1, 15, endpoint=True)

est_integral = [[] for _ in eps_m]
est_polynomial = [[] for _ in eps_m]

for n_node_ in n_node:
    _, w = xw_lgr(n_node_)
    for i, eps_m_ in enumerate(eps_m):
        P.set_discretization([0, 0.5 + eps_m_, 1], int(n_node_))
        S.update()
        v = linear_guess(P)
        v.t_f = 2
        v, info = ipopt.solve(
            S,
            v,
            optimizer_options={
                "tol": 1e-14,
                "acceptable_tol": 1e-18,
                "linear_solver": "mumps",
                "print_level": 0,
            },
        )
        assert info["status"] == 0
        f_b = (v.u[0][:n_node_] + 1) / 2
        roots = find_root(f_b - 0.5)
        if len(roots) == 0:
            mean_u = w @ f_b / 2
            switch_est = (0.5 + eps_m_) * (1 - mean_u)
            est_integral[i].append(switch_est)
            est_polynomial[i].append(np.nan)
        else:
            est_integral[i].append(np.nan)
            est_polynomial[i].append((roots[0] + 1) / 2 * (0.5 + eps_m_))

aerr_integral = [[] for _ in eps_m]
for i, eps_m_ in enumerate(eps_m):
    for j, est_integral_ in enumerate(est_integral[i]):
        aerr_integral[i].append(abs(est_integral_ - 0.5))

aerr_polynomial = [[] for _ in eps_m]
for i, eps_m_ in enumerate(eps_m):
    for j, est_polynomial_ in enumerate(est_polynomial[i]):
        aerr_polynomial[i].append(abs(est_polynomial_ - 0.5))

aerr_integral = np.array(list(reversed(aerr_integral)))
aerr_polynomial = np.array(list(reversed(aerr_polynomial)))


cmap_foreground = LinearSegmentedColormap.from_list("custom", color_foreground, N=256)
cmap_foreground.set_bad(color="lightgrey")

fig, (ax_0, ax_1) = plt.subplots(1, 2)

cmap = cmap_foreground
norm_integral = mpl.colors.Normalize(
    vmin=np.nanmin(aerr_integral), vmax=np.nanmax(aerr_integral)
)
norm_polynomial = mpl.colors.Normalize(
    vmin=np.nanmin(aerr_polynomial), vmax=np.nanmax(aerr_polynomial)
)
sm_integral = mpl.cm.ScalarMappable(norm=norm_integral, cmap=cmap)
sm_polynomial = mpl.cm.ScalarMappable(norm=norm_polynomial, cmap=cmap)

ax_0.imshow(aerr_polynomial, cmap=cmap_foreground, norm=norm_polynomial)
ax_0.set_xticks(np.arange(len(n_node)), [rf"${n_node_}$" for n_node_ in n_node])
ax_0.set_yticks(
    np.arange(len(eps_m)),
    [to_scientific_notation(eps_m_) for eps_m_ in reversed(eps_m)],
)
ax_0.set_ylabel(r"$\epsilon_m$")
ax_0.set_xlabel(r"$n_d$")
ax_0.set_xticks(np.arange(-0.5, len(n_node), 1), minor=True)
ax_0.set_yticks(np.arange(-0.5, len(eps_m), 1), minor=True)
ax_0.grid(which="minor", color="w", linestyle="-", linewidth=2)
ax_0.tick_params(which="minor", bottom=False, left=False)

ax_1.imshow(aerr_integral, cmap=cmap_foreground, norm=norm_integral)
ax_1.set_xticks(np.arange(len(n_node)), [rf"${n_node_}$" for n_node_ in n_node])
ax_1.set_yticks(
    np.arange(len(eps_m)),
    [to_scientific_notation(eps_m_) for eps_m_ in reversed(eps_m)],
)
ax_1.set_ylabel(r"$\epsilon_m$")
ax_1.set_xlabel(r"$n_d$")
ax_1.set_xticks(np.arange(-0.5, len(n_node), 1), minor=True)
ax_1.set_yticks(np.arange(-0.5, len(eps_m), 1), minor=True)
ax_1.yaxis.tick_right()
ax_1.yaxis.set_label_position("right")
ax_1.grid(which="minor", color="w", linestyle="-", linewidth=2)
ax_1.tick_params(which="minor", bottom=False, right=False)

fig.tight_layout(pad=2)
fig.colorbar(
    sm_polynomial,
    label="absolute error of the \npolynomial method",
    ax=[ax_0],
    location="top",
)
fig.colorbar(
    sm_integral,
    label="absolute error of the \nintegral method",
    ax=[ax_1],
    location="top",
)

fig.savefig("double_integrator_absolute_error.pdf", bbox_inches="tight")

print("Maximum absolute error of the polynomial method:", np.nanmax(aerr_polynomial))
print("Maximum absolute error of the integral method:", np.nanmax(aerr_integral))
