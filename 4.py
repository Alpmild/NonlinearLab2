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

RES = 100                 # разрешение

h = 0.01
T_TOTAL = 3000.0          # полное время счёта
T_DISCARD = 2850.0        # сколько отбрасываем как переходный процесс

a3 = 1.0
u0 = np.array([0.1, 0.1, 0.1], dtype=float)

A1_MIN, A1_MAX = 0.0, 3.0
A2_MIN, A2_MAX = 0.0, 3.0

CLUSTER_TOL = 1e-3        # максимумы ближе этого значения = один кластер
MAX_PEAKS = 20000
DIVERGE = 1e6             # порог расходимости

VMAX = 60                 # верхний предел цветовой шкалы (None = автоматически)
DIVERGED_VALUE = 0        # чем закрашивать расходящиеся решения:
                          # 0 -> как "нет максимумов" (тёмно-синий, без белых пятен)
                          # -1 -> белым (чтобы видеть расходимость)

CASE_NAME = "Case_B"
save_folder = "cluster_maps"

variables = [
    (0, "x", "X"),
    (1, "y", "Y"),
    (2, "z", "Z"),
]

for _, _, folder in variables:
    os.makedirs(os.path.join(save_folder, folder), exist_ok=True)

a1_values = np.linspace(A1_MIN, A1_MAX, RES)
a2_values = np.linspace(A2_MIN, A2_MAX, RES)


@njit(cache=True)
def f(u, a1, a2, a3):
    x = u[0]
    y = u[1]
    z = u[2]

    return np.array([
        a1 * y * z,
        x - a2 * y,
        a3 - x * y
    ])


methods = [
    ("Euler", njit(euler)),
    ("Euler-Cromer", njit(euler_kromer)),
    ("EMP", njit(emp)),
    ("VSKD", njit(kd)),
    ("RK4", njit(rk4)),
]


@njit(cache=True)
def count_clusters(x, cut, tol, max_peaks):
    peaks = np.empty(max_peaks)
    cnt = 0
    for i in range(cut + 1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            if cnt < max_peaks:
                peaks[cnt] = x[i]
                cnt += 1

    if cnt == 0:
        return 0

    p = np.sort(peaks[:cnt])
    clusters = 1
    for i in range(1, cnt):
        if p[i] - p[i - 1] > tol:
            clusters += 1
    return clusters


@njit(parallel=True)
def scan(method, f, a1_values, a2_values, u0, t1, h, a3,
         cut, tol, max_peaks, diverge, diverged_value):
    n1 = a1_values.shape[0]
    n2 = a2_values.shape[0]
    out = np.zeros((3, n2, n1), dtype=np.int64)

    for k in prange(n1 * n2):
        i2 = k // n1
        i1 = k % n1

        t, sol = method(f, u0.copy(), 0.0, t1, h,
                        a1_values[i1], a2_values[i2], a3)

        ok = True
        for i in range(sol.shape[0]):
            for j in range(3):
                v = sol[i, j]
                if not np.isfinite(v) or abs(v) > diverge:
                    ok = False
                    break
            if not ok:
                break

        for c in range(3):
            if ok:
                out[c, i2, i1] = count_clusters(sol[:, c], cut, tol, max_peaks)
            else:
                out[c, i2, i1] = diverged_value

    return out


def run_scan(method):
    cut = int(round(T_DISCARD / h))
    return scan(method, f, a1_values, a2_values, u0, T_TOTAL, h, a3,
                cut, CLUSTER_TOL, MAX_PEAKS, DIVERGE, DIVERGED_VALUE)


def plot_map(method_name, var_name, folder, data):
    plt.figure(figsize=(9, 6))

    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("white")
    masked = np.ma.masked_where(data < 0, data)   # -1 останется белым

    plt.imshow(masked, origin="lower", aspect="auto", cmap=cmap,
               extent=(A1_MIN, A1_MAX, A2_MIN, A2_MAX),
               interpolation="nearest", vmin=0, vmax=VMAX)
    plt.colorbar(label="Число кластеров",
                 extend="max" if VMAX is not None else "neither")

    plt.xlabel("a1")
    plt.ylabel("a2")
    plt.title(f"Sprott {CASE_NAME}: {method_name}, число кластеров "
              f"(переменная {var_name}, h = {h}, "
              f"T = {T_TOTAL:g}, отброшено {T_DISCARD:g}, {RES}x{RES})")

    filepath = os.path.join(
        save_folder, folder,
        f"{CASE_NAME}_clusters_{method_name}_{RES}x{RES}.png"
    )
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {filepath}")


if __name__ == "__main__":

    for method_name, method in methods:
        print(f"{method_name}...")
        res = run_scan(method)

        n_div = int(np.count_nonzero(res[0] == DIVERGED_VALUE)) \
            if DIVERGED_VALUE < 0 else None
        if n_div is not None:
            print(f"  расходящихся точек: {n_div} из {RES * RES}")

        for idx, var_name, folder in variables:
            plot_map(method_name, var_name, folder, res[idx])

    print("\nПостроение всех карт завершено.")
