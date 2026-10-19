from RK4 import rk4 as method
from RK4 import NAME_ERROR
from RK4 import rk4

import numpy as np
import matplotlib.pyplot as plt

T = 15
T_ERR = 2.0


def f(u):
    x, y, z = u
    return np.array([y * z, x - y, 1 - x * y])


def reference(u0, t_end, h):
    if method != rk4:
        return rk4(f, u0, 0.0, t_end, h)[1]
    return rk4(f, u0, 0.0, t_end, h / 2)[1][::2]


def error_curve(u0, t_end, h):
    t_h, u_h = method(f, u0, 0.0, t_end, h)
    return t_h, np.linalg.norm(u_h - reference(u0, t_end, h), np.inf, axis=1)


if __name__ == '__main__':

    # Фазовый портрет
    u0_list = [(1, 0.5, 0), (-2, 1, 1), (0, -1.5, 2), (2, 2, -1), (-1, -1, -1)]
    colors = ['darkred', 'navy', 'darkgreen', 'darkorange', 'purple']

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    for u0, c in zip(u0_list, colors):
        _, u = method(f, u0, 0.0, 40.0, 1e-3)

        ax.plot(u[:, 0], u[:, 1], u[:, 2], lw=0.6, color=c, label=f'u0={u0}')
        ax.scatter(*u0, color=c, s=30)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.legend(fontsize=8)
    plt.show()

    # Зависимости переменных от времени
    u0 = u0_list[0]
    t, u = method(f, u0, 0.0, T, 1e-3)

    fig_t, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    labels = ['x(t)', 'y(t)', 'z(t)']
    colors_t = ['darkred', 'navy', 'darkgreen']

    for i, ax_i in enumerate(axes):
        ax_i.plot(t, u[:, i], color=colors_t[i], lw=0.8)
        ax_i.set_ylabel(labels[i])
        ax_i.grid(True, which='both', ls=':')

    axes[-1].set_xlabel('t')
    fig_t.suptitle(f'Зависимости от времени, u0={tuple(u0)}')
    plt.tight_layout()
    plt.show()

    hs = 0.1 / 2.0 ** np.arange(6)

    # График ошибки от времени
    fig_e, ax_e = plt.subplots(figsize=(8, 6))
    for h in hs[:4]:
        t_h, e_h = error_curve(u0, T, h)
        ax_e.semilogy(t_h[1:], e_h[1:], lw=1.2, label=f'h = {h:.5f}')
    ax_e.set_xlabel('t')
    ax_e.set_ylabel('Ошибка')
    ax_e.set_title(f'{NAME_ERROR}: зависимость от времени')
    ax_e.grid(True, which='both', ls=':')
    ax_e.legend()
    plt.show()

    # Порядок метода на отрезке [0, T_ERR]
    errs = np.array([error_curve(u0, T_ERR, h)[1][-1] for h in hs])

    # Порядок метода
    p_pairs = np.log2(errs[:-1] / errs[1:])
    p_fit = p_pairs.mean()

    print(f"Оценка порядка на отрезке")
    print("h          E(h)        p = log2(E(h)/E(h/2))")
    for i, h in enumerate(hs):
        p_str = f"{p_pairs[i]:.4f}" if i < len(p_pairs) else "  ---"
        print(f"{h:<10.5f} {errs[i]:>11.3e}   {p_str}")
    print(f"\nПорядок метода (среднее по парам): {p_fit:.4f}")

    # График ошибки от шага
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(hs, errs, 'o-', color='darkred', label=NAME_ERROR)
    ax.loglog(hs, errs[0] * (hs / hs[0]) ** p_fit, '--', color='gray',
              label=f'Наклон p = {p_fit:.3f}')

    ax.set_xlabel('Шаг h')
    ax.set_ylabel('Глобальная ошибка')
    ax.set_title('Порядок метода')
    ax.grid(True, which='both', ls=':')
    ax.legend()
    plt.show()
