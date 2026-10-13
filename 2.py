import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

from Euler import euler
from EulerKromer import euler_kromer
from EMP import emp
from KD import kd
from RK4 import rk4


# ============================================================
# Настройки
# ============================================================

# Параметры системы фиксированы, меняется только шаг h
a1, a2, a3 = 1.0, 1.0, 1.0

u0 = np.array([0.1, 0.1, 0.1], dtype=float)

T = 100.0          # время интегрирования (одинаковое для всех h)
transient = 0.5    # доля траектории, отбрасываемая как переходный процесс
MAX_PEAKS = 2000

min_h = 1e-4
max_h = 0.1
res = 1000         # число значений шага

# Нелинейная (геометрическая) сетка шагов:
#   h_i = min_h * (max_h / min_h) ** (i / (res - 1)),  i = 0..res-1
# На графике с логарифмической осью точки лежат равномерно.
h_values = min_h * (max_h / min_h) ** (np.arange(res) / (res - 1))
# то же самое: np.geomspace(min_h, max_h, res)

save_folder = "bifurcation_plots_h"

variables = [
    (0, "x", "X", (-2, 4)),
    (1, "y", "Y", (-2, 4)),
    (2, "z", "Z", (-2, 4)),
]

for _, _, folder, _ in variables:
    os.makedirs(os.path.join(save_folder, folder), exist_ok=True)


# ============================================================
# Система: Sprott Case C
# ============================================================

@njit(cache=True)
def f(u, a1, a2, a3):
    x = u[0]
    y = u[1]
    z = u[2]

    return np.array([
        a1 * y * z,
        x - a2 * y,
        a3 - x * x
    ])


methods = [
    ("Euler", njit(euler)),
    ("Euler-Cromer", njit(euler_kromer)),
    ("EMP", njit(emp)),
    ("VSKD", njit(kd)),
]
rk4_jit = njit(rk4)


@njit(cache=True)
def find_extrema(x, transient, out):
    cut = int(round(len(x) * transient))
    x_cut = x[cut:]

    cnt = 0
    for i in range(1, len(x_cut) - 1):
        if x_cut[i] > x_cut[i - 1] and x_cut[i] > x_cut[i + 1]:
            if cnt < out.shape[0]:
                out[cnt] = x_cut[i]
                cnt += 1
    return cnt


# ============================================================
# Прогон метода по всем шагам h (параллельно по ядрам)
# ============================================================

@njit(parallel=True)
def scan_h(method, f, h_values, u0, T, a1, a2, a3, transient, max_peaks):

    n = h_values.shape[0]
    buf = np.empty((3, n, max_peaks))
    counts = np.zeros((3, n), dtype=np.int64)

    for k in prange(n):
        hk = h_values[k]

        t, sol = method(f, u0.copy(), 0.0, T, hk, a1, a2, a3)

        ok = True
        for i in range(sol.shape[0]):
            for j in range(3):
                v = sol[i, j]
                if not np.isfinite(v) or abs(v) > 1e6:
                    ok = False
                    break
            if not ok:
                break

        if ok:
            for c in range(3):
                counts[c, k] = find_extrema(sol[:, c], transient, buf[c, k])

    return buf, counts


def run_scan(method):
    buf, counts = scan_h(method, f, h_values, u0, T, a1, a2, a3,
                         transient, MAX_PEAKS)
    result = {}
    for c in range(3):
        p = np.concatenate(
            [np.full(cnt, v) for v, cnt in zip(h_values, counts[c])]
        )
        x = np.concatenate(
            [buf[c, k, :cnt] for k, cnt in enumerate(counts[c])]
        )
        result[c] = (p, x)
    return result


# ============================================================
# График
# ============================================================

def plot_h(method_name, var_name, folder, ylim, mh, mx, rh, rx):
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3, which="both")

    dot = dict(marker=".", linestyle="none", markersize=1.5,
               markeredgewidth=0)

    if len(mx) > 0:
        plt.plot(mh, mx, color="blue", label=method_name, **dot)
    if len(rx) > 0:
        plt.plot(rh, rx, color="red", alpha=0.5, label="RK4", **dot)

    plt.xscale("log")
    plt.xlabel("h")
    plt.ylabel(f"Экстремумы {var_name}")
    plt.title(
        f"Sprott Case C: зависимость от шага h, {method_name} и RK4 "
        f"(переменная {var_name})"
    )
    plt.legend(loc="best", markerscale=6)
    plt.xlim(min_h, max_h)
    plt.ylim(*ylim)

    filepath = os.path.join(save_folder, folder,
                            f"Case_C_h_{method_name}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {filepath}")


if __name__ == "__main__":

    print("RK4...")
    rk4_res = run_scan(rk4_jit)

    for method_name, method in methods:
        print(f"{method_name}...")
        res_m = run_scan(method)

        for idx, var_name, folder, ylim in variables:
            mh, mx = res_m[idx]
            rh, rx = rk4_res[idx]
            plot_h(method_name, var_name, folder, ylim, mh, mx, rh, rx)

    print("\nГотово.")