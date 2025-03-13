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


kilogram = 1
second = 1 / 86400 / 365.25
day = 1 / 365.25
meter = 1 / 149597870700
newton = kilogram * meter / second**2

i_sp = 3800 * second
t_max = 0.33 * newton
m_0 = 1500 * kilogram
g_0 = 9.80665 * meter / second**2
mu = 39.476926

r_0 = [9.708322e-1, 2.375844e-1, -1.671055e-6]
v_0 = [-1.598191, 6.081958, 9.443368e-5]
r_f = [-3.277178e-1, 6.389172e-1, 2.765929e-2]
v_f = [-6.598211, -3.412933, 3.340902e-1]


def Des2MEOE(rv):
    r = np.array(rv[:3])
    v = np.array(rv[3:])
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    xi = v_norm**2 / 2 - mu / r_norm
    a = -mu / 2 / xi
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    p = h_norm**2 / mu
    i = np.arccos(h[2] / h_norm)
    Omega = np.arctan2(h[0], -h[1])
    e_norm = np.sqrt(1 - p / a)
    e = np.cross(v, h) / mu - r / r_norm
    ON = np.array([np.cos(Omega), np.sin(Omega), 0])
    omega = np.arccos(np.dot(ON, e) / e_norm)
    if e[2] < 0:
        omega = 2 * np.pi - omega
    theta = np.arccos(np.dot(e, r) / e_norm / r_norm)
    if np.dot(r, v) < 0:
        theta = 2 * np.pi - theta
    p_meoe = a * (1 - e_norm**2)
    f_meoe = e_norm * np.cos(omega + Omega)
    g_meoe = e_norm * np.sin(omega + Omega)
    h_meoe = np.tan(i / 2) * np.cos(Omega)
    k_meoe = np.tan(i / 2) * np.sin(Omega)
    L_meoe = (theta + omega + Omega) % (2 * np.pi)
    return np.array([p_meoe, f_meoe, g_meoe, h_meoe, k_meoe, L_meoe])


def MEOE2Des(meoe):
    p, f, g, h, k, L = meoe
    rv = np.empty(6, dtype=np.float64)
    q = 1 + f * np.cos(L) + g * np.sin(L)
    r = p / q
    s_2 = 1 + h**2 + k**2
    alpha_2 = h**2 - k**2
    rv[0] = r / s_2 * (np.cos(L) + alpha_2 * np.cos(L) + 2 * h * k * np.sin(L))
    rv[1] = r / s_2 * (np.sin(L) - alpha_2 * np.sin(L) + 2 * h * k * np.cos(L))
    rv[2] = 2 * r / s_2 * (h * np.sin(L) - k * np.cos(L))
    rv[3] = (
        -1
        / s_2
        * np.sqrt(mu / p)
        * (
            np.sin(L)
            + alpha_2 * np.sin(L)
            - 2 * h * k * np.cos(L)
            + g
            - 2 * f * h * k
            + alpha_2 * g
        )
    )
    rv[4] = (
        -1
        / s_2
        * np.sqrt(mu / p)
        * (
            -np.cos(L)
            + alpha_2 * np.cos(L)
            + 2 * h * k * np.sin(L)
            - f
            + 2 * g * h * k
            + alpha_2 * f
        )
    )
    rv[5] = 2 / s_2 * np.sqrt(mu / p) * (h * np.cos(L) + k * np.sin(L) + f * h + g * k)
    return rv


S = System(0, fastmath=True)
P = S.new_phase(["p", "f", "g", "h", "k", "L", "m"], ["u_r", "u_t", "u_n"])
p, f, g, h, k, L, m = P.x
u_r, u_t, u_n = P.u
a_r = u_r * t_max / m
a_t = u_t * t_max / m
a_n = u_n * t_max / m
w = 1 + f * sp.cos(L) + g * sp.sin(L)
s_2 = 1 + h**2 + k**2
u_norm = sp.sqrt(u_r**2 + u_t**2 + u_n**2)
dot_p = 2 * p / w * sp.sqrt(p / mu) * a_t
dot_f = sp.sqrt(p / mu) * (
    sp.sin(L) * a_r
    + ((w + 1) * sp.cos(L) + f) * a_t / w
    - (h * sp.sin(L) - k * sp.cos(L)) * g * a_n / w
)
dot_g = sp.sqrt(p / mu) * (
    -sp.cos(L) * a_r
    + ((w + 1) * sp.sin(L) + g) * a_t / w
    + (h * sp.sin(L) - k * sp.cos(L)) * f * a_n / w
)
dot_h = sp.sqrt(p / mu) * s_2 * a_n / 2 / w * sp.cos(L)
dot_k = sp.sqrt(p / mu) * s_2 * a_n / 2 / w * sp.sin(L)
dot_L = (
    sp.sqrt(mu * p) * (w / p) ** 2
    + 1 / w * sp.sqrt(p / mu) * (h * sp.sin(L) - k * sp.cos(L)) * a_n
)
dot_m = -t_max * u_norm / i_sp / g_0
P.set_dynamics([dot_p, dot_f, dot_g, dot_h, dot_k, dot_L, dot_m], cache="./cache")
meoe_0 = Des2MEOE([*r_0, *v_0])
meoe_f = Des2MEOE([*r_f, *v_f])
P.set_boundary_condition(
    [*meoe_0, m_0], [*meoe_f[:5], meoe_f[5] + 2 * np.pi * 3, None], 0, 1000 * day
)
P.set_integral([u_norm], cache="./cache")
P.set_phase_constraint([u_norm], [0], [1], [True], cache="./cache")
P.set_discretization(30, 3)
S.set_phase([P])
S.set_objective(P.I[0], cache="./cache")


v = linear_guess(P, 1e-6)
S.objective(v.data)
S.gradient(v.data)
S.jacobian(v.data)
S.hessian_o(v.data)  # Pre-compile JIT functions
time_0 = time.time()
v, info = ipopt.solve(S, v, optimizer_options={"print_level": 0, "tol": 1e-9})
assert info["status"] == 0


max_iter = 10
for it in range(max_iter):
    if S.check_discontinuous(v):
        break
    v = S.refine_discontinuous(
        v, num_point_min=2, num_point_max=3, mesh_length_max=1 / 30
    )
    v, info = ipopt.solve(
        S,
        v,
        optimizer_options={
            "print_level": 0,
            "tol": 1e-9,
        },
    )
    assert info["status"] == 0
    print(it, info["obj_val"])

print(f"Fuel consumption: {(m_0 - v.x[6][-1]) / kilogram} kg")

time_f = time.time()
print(f"Elapsed time: {time_f - time_0:.2f} s")


mesh = P._mesh
t_out = []
for i in range(len(mesh) - 1):
    t_out.append(np.linspace(v.t_f * mesh[i], v.t_f * mesh[i + 1], 10, endpoint=True))
t_out = np.concatenate(t_out)
V_x = v.V_x(t_out)
V_u = v.V_u(t_out)


fig = plt.figure()
gs = fig.add_gridspec(4, 2)
axs = [fig.add_subplot(gs[0, :])]
for i in range(1, 4):
    axs.append(fig.add_subplot(gs[i, 0]))
    axs.append(fig.add_subplot(gs[i, 1]))
axs[0].plot(t_out / day, V_x @ v.x[6], color=color_foreground[-1], zorder=1)
axs[0].minorticks_on()
axs[0].grid(linestyle="--", linewidth="0.2")
for j in range(len(mesh)):
    axs[0].axvline(
        mesh[j] * v.t_f / day, color=color_mesh, linestyle=":", zorder=0, linewidth=1
    )
for i in range(6):
    axs[i + 1].plot(t_out / day, V_x @ v.x[i], color=color_analytical, zorder=1)
    axs[i + 1].minorticks_on()
    axs[i + 1].grid(linestyle="--", linewidth="0.2")
    for j in range(len(mesh)):
        axs[i + 1].axvline(
            mesh[j] * v.t_f / day,
            color=color_mesh,
            linestyle=":",
            zorder=0,
            linewidth=1,
        )
axs[0].set_ylabel(r"$m$")
axs[1].set_ylabel(r"$p$")
axs[2].set_ylabel(r"$f$")
axs[3].set_ylabel(r"$g$")
axs[4].set_ylabel(r"$h$")
axs[5].set_ylabel(r"$k$")
axs[6].set_ylabel(r"$L$")
axs[0].set_xlabel(r"$t$ (day)")
axs[5].set_xlabel(r"$t$ (day)")
axs[6].set_xlabel(r"$t$ (day)")
axs[2].yaxis.tick_right()
axs[2].yaxis.set_label_position("right")
axs[4].yaxis.tick_right()
axs[4].yaxis.set_label_position("right")
axs[6].yaxis.tick_right()
axs[6].yaxis.set_label_position("right")
axs[0].xaxis.tick_top()
axs[0].xaxis.set_label_position("top")
for i in range(1, 5):
    axs[i].set_xticklabels([])
fig.tight_layout()
fig.savefig("minimum_fuel_transfer_state.pdf", bbox_inches="tight")


fig, axs = plt.subplots(4, 1, sharex=True)
u_out = [V_u @ v.u[i] for i in range(3)]
axs[3].plot(
    t_out / day,
    np.sqrt(u_out[0] ** 2 + u_out[1] ** 2 + u_out[2] ** 2),
    color=color_foreground[-1],
    zorder=1,
)
axs[3].set_ylabel(r"$u$")
for i in range(3):
    axs[i].plot(
        t_out / day,
        u_out[i],
        color=color_analytical,
        zorder=1,
    )
    axs[i].set_ylabel(r"$u_{}$".format("rtn"[i]))
for i in range(4):
    axs[i].minorticks_on()
    axs[i].grid(linestyle="--", linewidth="0.2")
    for j in range(len(mesh)):
        axs[i].axvline(
            mesh[j] * v.t_f / day,
            color=color_mesh,
            linestyle=":",
            zorder=0,
            linewidth=1,
        )
axs[3].set_xlabel(r"$t$ (day)")
fig.tight_layout()
fig.savefig("minimum_fuel_transfer_control.pdf", bbox_inches="tight")


fig, ax = plt.subplots()
rv = []
for i in range(len(v.t_x)):
    rv.append(
        MEOE2Des([v.x[0][i], v.x[1][i], v.x[2][i], v.x[3][i], v.x[4][i], v.x[5][i]])
    )
rv = np.array(rv).T
r_x = V_x @ rv[0]
r_y = V_x @ rv[1]
label = [False, False]
for i in range(len(mesh) - 1):
    u_ = np.sqrt(
        v.u[0][3 * i : 3 * (i + 1)] ** 2
        + v.u[1][3 * i : 3 * (i + 1)] ** 2
        + v.u[2][3 * i : 3 * (i + 1)] ** 2
    )
    color = color_foreground[-1] if np.mean(u_) > 0.5 else color_foreground[0]
    linestyle = "solid" if np.mean(u_) > 0.5 else "dotted"
    if np.mean(u_) > 0.5 and not label[0]:
        label[0] = True
        ax.plot(
            r_x[10 * i : 10 * (i + 1) + 1],
            r_y[10 * i : 10 * (i + 1) + 1],
            color=color,
            label="thrusting",
        )
    elif np.mean(u_) < 0.5 and not label[1]:
        label[1] = True
        ax.plot(
            r_x[10 * i : 10 * (i + 1) + 1],
            r_y[10 * i : 10 * (i + 1) + 1],
            color=color,
            label="coasting",
            linestyle=linestyle,
        )
    else:
        ax.plot(
            r_x[10 * i : 10 * (i + 1) + 1],
            r_y[10 * i : 10 * (i + 1) + 1],
            color=color,
            linestyle=linestyle,
        )
ax.legend()
ax.axis("equal")
ax.set_xlabel(r"$x$ (AU)")
ax.set_ylabel(r"$y$ (AU)")
ax.minorticks_on()
ax.grid(linestyle="--", linewidth="0.2")
fig.savefig("minimum_fuel_transfer_orbit.pdf", bbox_inches="tight")
