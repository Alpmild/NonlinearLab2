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

h = 0.001

u0 = np.array([0.1, 0.1, 0.1], dtype=float)

start_params = (1.0, 1.0, 1.0)

N = 2_000_000
transient = 0.6
MAX_PEAKS = 2000  # максимум сохраняемых максимумов на одно значение параметра

a1_values = np.linspace(0, 1.5, 1000)
a2_values = np.linspace(0, 3.0, 1000)
a3_values = np.linspace(0, 1.5, 1000)

save_folder = "bifurcation_plots"

variables = [
    (0, "x", "X", (-2, 4)),
    (1, "y", "Y", (-2, 4)),
    (2, "z", "Z", (-2, 4)),
]

for _, _, folder, _ in variables:
    os.makedirs(os.path.join(save_folder, folder), exist_ok=True)


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

parameters = [
    ("a1", a1_values),
    ("a2", a2_values),
    ("a3", a3_values),
]



@njit(cache=True)
def find_extrema(x, transient, out):
    """
    Поиск локальных максимумов
    """
    cut = int(round(len(x) * transient))
    x_cut = x[cut:]

    cnt = 0
    for i in range(1, len(x_cut) - 1):
        if x_cut[i] > x_cut[i - 1] and x_cut[i] > x_cut[i + 1]:
            if cnt < out.shape[0]:
                out[cnt] = x_cut[i]
                cnt += 1
    return cnt


@njit(parallel=True)
def scan(method, f, values, which, u0, t1, h,
         a1_0, a2_0, a3_0, transient, max_peaks):
    """
    Прогон метода по всем значениям параметра
    """
    n = values.shape[0]
    buf = np.empty((3, n, max_peaks))
    counts = np.zeros((3, n), dtype=np.int64)

    for k in prange(n):
        a1, a2, a3 = a1_0, a2_0, a3_0

        if which == 0:
            a1 = values[k]
        elif which == 1:
            a2 = values[k]
        else:
            a3 = values[k]

        t, sol = method(f, u0.copy(), 0.0, t1, h, a1, a2, a3)

        # решение разошлось -> точек нет
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


def run_scan(method, which, values):
    """Возвращает {индекс переменной: (значения параметра, максимумы)}."""
    buf, counts = scan(
        method, f, values, which, u0, N * h, h,
        start_params[0], start_params[1], start_params[2],
        transient, MAX_PEAKS
    )

    result = {}
    for c in range(3):
        p = np.concatenate(
            [np.full(cnt, v) for v, cnt in zip(values, counts[c])]
        )
        x = np.concatenate(
            [buf[c, k, :cnt] for k, cnt in enumerate(counts[c])]
        )
        result[c] = (p, x)
    return result


def plot_bifurcation(method_name, parameter_name, params,
                     var_name, folder, ylim, mp, mx, rp, rx):
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)

    dot = dict(marker=".", linestyle="none", markersize=1.5,
               markeredgewidth=0)

    if len(mx) > 0:
        plt.plot(mp, mx, color="blue", label=method_name, **dot)

    if len(rx) > 0:
        plt.plot(rp, rx, color="red", alpha=0.5, label="RK4", **dot)

    plt.xlabel(parameter_name)
    plt.ylabel(f"Экстремумы {var_name}")
    plt.title(
        f"Sprott Case B: {parameter_name}, {method_name} и RK4 "
        f"(переменная {var_name})"
    )
    plt.legend(loc="best", markerscale=6)
    plt.xlim(min(params), max(params))
    plt.ylim(*ylim)

    filepath = os.path.join(
        save_folder, folder, f"Case_B_{parameter_name}_{method_name}.png"
    )
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"График сохранен: {filepath}")


if __name__ == "__main__":

    for which, (parameter_name, params) in enumerate(parameters):
        print(f"RK4, параметр {parameter_name}...")
        rk4_res = run_scan(rk4_jit, which, params)

        for method_name, method in methods:
            print(f"{method_name}, параметр {parameter_name}...")
            res = run_scan(method, which, params)

            for idx, var_name, folder, ylim in variables:
                mp, mx = res[idx]
                rp, rx = rk4_res[idx]
                plot_bifurcation(
                    method_name, parameter_name, params,
                    var_name, folder, ylim, mp, mx, rp, rx
                )

    print("\nПостроение всех бифуркационных диаграмм завершено.")

