import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

from KD import kd

# ============================================================
# Настройки
# ============================================================

u0 = np.array([0.1, 0.1, 0.1], dtype=float)

# фиксированные параметры системы (a1, a2, a3)
params = (1.0, 1.0, 1.0)

h = 0.001
N = 1_000_000
T = N * h

transient = 0.5
MAX_PEAKS = 2000

CASE_NAME = "Case_B"
save_folder = "kd_s_plots"

# Диапазоны s в порядке построения:
s_ranges = [
    ("0_1",       0.0,    1.0,   1000),   # теоретический диапазон
    ("m100_100", -100.0, 100.0,  2000),   # рабочий диапазон
]

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
        a3 - x * y
    ])


kd_jit = njit(kd)


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
def scan(method, f, s_values, u0, t1, h, a1, a2, a3, transient, max_peaks):
    n = s_values.shape[0]
    buf = np.empty((3, n, max_peaks))
    counts = np.zeros((3, n), dtype=np.int64)

    for k in prange(n):
        s = s_values[k]

        t, sol = method(f, u0.copy(), 0.0, t1, h, a1, a2, a3, s)

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


def run_scan(s_values):
    """Возвращает {индекс переменной: (значения s, максимумы)} и counts."""
    buf, counts = scan(
        kd_jit, f, s_values, u0, T, h,
        params[0], params[1], params[2],
        transient, MAX_PEAKS
    )

    result = {}
    for c in range(3):
        ss = np.concatenate(
            [np.full(cnt, v) for v, cnt in zip(s_values, counts[c])]
        )
        x = np.concatenate(
            [buf[c, k, :cnt] for k, cnt in enumerate(counts[c])]
        )
        result[c] = (ss, x)
    return result, counts


def plot_bifurcation(var_name, folder, ylim, s_values, tag, ps, px):
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)

    if len(px) > 0:
        plt.plot(ps, px, color="blue", marker=".", linestyle="none",
                 markersize=1.5, markeredgewidth=0)

    # на широком диапазоне отмечаем теоретический интервал [0, 1]
    if s_values.min() < 0.0 or s_values.max() > 1.0:
        plt.axvspan(0, 1, color="green", alpha=0.15)

    plt.xlabel("s")
    plt.ylabel(f"Максимумы {var_name}")
    plt.title(f"Sprott {CASE_NAME}: KD, бифуркационная диаграмма по s "
              f"(переменная {var_name}, h = {h})")
    plt.xlim(s_values.min(), s_values.max())
    plt.ylim(*ylim)

    filepath = os.path.join(save_folder, folder,
                            f"{CASE_NAME}_KD_s_{tag}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {filepath}")


if __name__ == "__main__":

    for tag, s_min, s_max, npts in s_ranges:
        s_values = np.linspace(s_min, s_max, npts)
        print(f"KD, s ∈ [{s_min}, {s_max}], {npts} точек...")

        res, counts = run_scan(s_values)

        alive = np.count_nonzero(counts[0] > 0)
        print(f"  значений s с найденными максимумами x: "
              f"{alive} из {npts}")

        for idx, var_name, folder, ylim in variables:
            ps, px = res[idx]
            plot_bifurcation(var_name, folder, ylim, s_values, tag, ps, px)

    print("\nПостроение всех диаграмм завершено.")
