import numpy as np
import matplotlib.pyplot as plt
from Euler import euler as method   # можно подставить и rk4 (сигнатура та же)

# ---------------------------------------------------------------
# Параметры системы: a[0]=0.5, a[1]=1, a[2]=1, a[3]=1
#   x' = a[1] * y * z
#   y' = x - a[2] * y
#   z' = a[3] - x * y
# (a[0] в уравнениях не участвует, оставлен, чтобы совпадала нумерация)
# ---------------------------------------------------------------
A = [0.5, 1.0, 1.0, 1.0]
U0 = (1.0, 0.5, 0.0)          # начальное условие (одно и то же для всех точек диаграммы)

# ---------------------------- настройки ------------------------
MODE = 'param'                # 'param' - перебираем параметр a[PARAM_INDEX], 'step' - перебираем шаг h
PARAM_INDEX = 3               # номер перебираемого параметра (1, 2 или 3)
P_MIN, P_MAX = 0.2, 2.0       # границы перебора параметра

H_FIXED = 0.01                # шаг метода в режиме 'param'
H_MIN, H_MAX = 0.005, 0.5     # границы перебора шага в режиме 'step'

RESOLUTION = 500              # разрешение диаграммы (число интервалов сетки)

T_TRANS = 50.0                # время "успокоения" (аналог N итераций из вики)
T_REC = 20.0                  # время записи точек (аналог M итераций из вики)

VAR = 0                       # какую переменную рисуем: 0 - x, 1 - y, 2 - z
RECORD = 'maxima'             # 'maxima' - локальные максимумы, 'points' - все точки участка
BLOWUP = 1e6                  # если |u| больше - считаем, что метод "взорвался"


def make_f(a):
    """Правая часть системы с параметрами a. Возвращает f(u), как ожидает euler."""
    def f(u):
        x, y, z = u
        return np.array([a[1] * y * z,
                         x - a[2] * y,
                         a[3] - x * y])
    return f


def grid_value(i):
    """Значение перебираемой величины на i-м узле сетки."""
    if MODE == 'param':
        # линейная сетка: (max - min) / resolution * i + min
        return (P_MAX - P_MIN) / RESOLUTION * i + P_MIN
    # шаг перебираем нелинейно: геометрическая (логарифмическая) сетка
    return H_MIN * (H_MAX / H_MIN) ** (i / RESOLUTION)


def diagram_points(a, h):
    """Точки диаграммы для одного значения параметра/шага."""
    f = make_f(a)
    N = int(round(T_TRANS / h))
    M = int(round(T_REC / h))

    # шаг 3 из вики: N итераций, чтобы система стабилизировалась
    _, u = method(f, U0, 0.0, N * h, h)
    u_N = u[-1]
    if not np.all(np.isfinite(u_N)) or np.max(np.abs(u_N)) > BLOWUP:
        return np.empty(0)

    # шаг 4 из вики: ещё M итераций, значения записываем
    _, u = method(f, u_N, 0.0, M * h, h)
    v = u[:, VAR]
    if not np.all(np.isfinite(v)):
        return np.empty(0)

    if RECORD == 'maxima':
        mask = (v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:])
        peaks = v[1:-1][mask]
        return peaks if peaks.size else v[-1:]   # нет максимумов -> положение равновесия
    return v[::max(1, M // 200)]


if __name__ == '__main__':
    xs, ys = [], []

    with np.errstate(all='ignore'):
        for i in range(RESOLUTION + 1):
            s = grid_value(i)          # текущее значение параметра или шага
            if MODE == 'param':
                a = list(A)
                a[PARAM_INDEX] = s
                h = H_FIXED
            else:
                a = A
                h = s

            pts = diagram_points(a, h)
            xs.append(np.full(pts.size, s))
            ys.append(pts)

            if i % 20 == 0:
                print(f'{i}/{RESOLUTION}')

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys, ',k', alpha=0.6) if xs.size > 200000 else ax.scatter(xs, ys, s=0.5, color='k')

    if MODE == 'param':
        ax.set_xlabel(f'a[{PARAM_INDEX}]')
        ax.set_title(f'Бифуркационная диаграмма по a[{PARAM_INDEX}] (h = {H_FIXED})')
    else:
        ax.set_xscale('log')
        ax.set_xlabel('Шаг h')
        ax.set_title('Бифуркационная диаграмма по шагу h')
    ax.set_ylabel('xyz'[VAR])
    ax.grid(True, which='both', ls=':')
    plt.show()