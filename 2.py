import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

from Euler import euler
from EulerKromer import euler_kromer
from EMP import emp
from KD import kd
from RK4 import rk4

u0 = np.array([0.1, 0.1, 0.1], dtype=float)

params = (1.0, 1.0, 1.0)

min_h = 0.001
max_h = 0.6
res = 1000

T = 1000.0

transient = 0.5
MAX_PEAKS = 2000

CASE_NAME = "Case_B"

save_folder = "step_plots"

variables = [
    (0, "x", "X", (-2, 4)),
    (1, "y", "Y", (-2, 4)),
    (2, "z", "Z", (-2, 4)),
]

for _, _, folder, _ in variables:
    os.makedirs(os.path.join(save_folder, folder), exist_ok=True)

i_arr = np.arange(res)
h_values = ((max_h / min_h) ** (1.0 / (res - 1))) ** i_arr * min_h


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
    ("RK4", njit(rk4)),
]


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


@njit(parallel=True)
def scan(method, f, h_values, u0, t1, a1, a2, a3, transient, max_peaks):
    n = h_values.shape[0]
    buf = np.empty((3, n, max_peaks))
    counts = np.zeros((3, n), dtype=np.int64)

    for k in prange(n):
        h = h_values[k]

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


def run_scan(method):
    """Возвращает {индекс переменной: (значения шага, максимумы)}."""
    buf, counts = scan(
        method, f, h_values, u0, T,
        params[0], params[1], params[2],
        transient, MAX_PEAKS
    )

    result = {}
    for c in range(3):
        hh = np.concatenate(
            [np.full(cnt, v) for v, cnt in zip(h_values, counts[c])]
        )
        x = np.concatenate(
            [buf[c, k, :cnt] for k, cnt in enumerate(counts[c])]
        )
        result[c] = (hh, x)
    return result


def plot_vs_step(method_name, var_name, folder, ylim, mh, mx):
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3, which="both")

    if len(mx) > 0:
        plt.plot(mh, mx, color="blue", marker=".", linestyle="none",
                 markersize=1.5, markeredgewidth=0)

    plt.xscale("log")  # шаги расположены в геометрической прогрессии
    plt.xlabel("Шаг h")
    plt.ylabel(f"Максимумы {var_name}")
    plt.title(f"Sprott {CASE_NAME}: {method_name}, зависимость от шага h "
              f"(переменная {var_name})")
    plt.xlim(min_h, max_h)
    plt.ylim(*ylim)

    filepath = os.path.join(
        save_folder, folder, f"{CASE_NAME}_step_{method_name}.png"
    )
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"График сохранен: {filepath}")



if __name__ == "__main__":

    for method_name, method in methods:
        print(f"{method_name}...")
        res_m = run_scan(method)

        for idx, var_name, folder, ylim in variables:
            mh, mx = res_m[idx]
            plot_vs_step(method_name, var_name, folder, ylim, mh, mx)

    print("\nПостроение всех графиков завершено.")
