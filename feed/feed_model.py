import numpy as np


def f0(x):
    out = np.power(np.sqrt(2 * x) - b / lam, 2) * (2 * x - 1) \
          - np.power(gain / 2 / np.pi * np.sqrt(3 / 2 / np.pi) / np.sqrt(x) - a / lam, 2) \
          * (np.power(gain, 2) / 6 / np.power(np.pi, 3) / x - 1)
    return out


def f1(x):
    out = 2 * np.power(np.sqrt(2 * x) - b / lam, 2) \
          + (np.sqrt(2 * x) - b / lam) * np.sqrt(2 / x) * (2 * x - 1) \
          + (gain / 2 / np.pi * np.sqrt(3 / 2 / np.pi) / np.sqrt(x) - a / lam) * gain / 2 / np.pi \
          * np.sqrt(3 / 2 / np.pi) / np.power(x, 3 / 2) \
          * (np.power(gain, 2) / 6 / np.power(np.pi, 3) / x - 1) \
          + np.power(gain / 2 / np.pi * np.sqrt(3 / 2 / np.pi) / np.sqrt(x) - a / lam, 2) \
          * np.power(gain, 2) / 6 / np.power(np.pi, 3) / np.power(x, 2)
    return out


def get_x(x0, max_iter=1000, loss_min=1e-5):
    loss = 0
    x_reg = x0
    for i in range(max_iter):
        x_next = x_reg - f0(x_reg) / f1(x_reg)
        loss = np.abs(x_next - x_reg)
        x_reg = x_next
        if loss < loss_min:
            loss_min = loss
            print("x={:.9f}, loss={:.9f}".format(x_reg, loss))
            break
    return x_reg, loss


if __name__ == "__main__":
    fc = 11
    lam = 0.3 / 11
    gain = np.power(10, 22.6 / 10)
    a = 0.9 * 25.4 / 1e3
    b = 0.4 * 25.4 / 1e3
    x0 = gain / (2 * np.pi * np.sqrt(2 * np.pi))
    x, loss = get_x(x0)
    pe = x * lam
    ph = gain ** 2 / 8 / np.pi ** 3 / x * lam
    a1 = np.sqrt(3 * lam * ph)
    b1 = np.sqrt(2 * lam * pe)
    lh = (b1 - b) * np.sqrt(np.power(pe / b1, 2) - 1 / 4)
    le = (a1 - a) * np.sqrt(np.power(ph / a1, 2) - 1 / 4)

    taper_h = np.arctan(a1 / 2 / ph) / np.pi * 180
    taper_e = np.arctan(b1 / 2 / pe) / np.pi * 180
