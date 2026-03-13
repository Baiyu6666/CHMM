import numpy as np
import matplotlib.pyplot as plt

def vmf_grad_wrt_g_full(X: np.ndarray, g: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    """
    Sum over all segments t:
      ∂/∂g [kappa * <v_hat_t, u_hat_t>] where u_hat_t = (g-x_t)/||g-x_t||.
    Implements: kappa * (I/r - uu^T/r^3) @ v_hat
    """
    g = np.asarray(g, float)
    D = X.shape[1]
    grad = np.zeros(D, dtype=float)
    for t in range(len(X) - 1):
        v = X[t + 1] - X[t]
        v_n = np.linalg.norm(v)
        if v_n < 1e-12:
            continue
        v_hat = v / v_n

        u = g - X[t]
        r = np.linalg.norm(u)
        if r < 1e-12:
            continue

        Jv = v_hat / r - (u * (np.dot(u, v_hat))) / (r**3 + 1e-12)
        grad += kappa * Jv
    return grad

# --- A simple polyline trajectory: right then down ---
X = np.array([
    [-3.0,  0.0],
    [-2.5, 0.0],
    [-2.0,  0.0],
    [-1.5, 0.0],
    [-1.0,  0.0],
    [-0.5, 0.0],
    [ 0.0,  0.0],
    [0.5, 0.0],
    [ 1.0,  0.0],
    [1.5, 0.0],
    [ 2.0,  0.0],
    [ 2.2, -1.0],
    [ 2.4, -2.0],
    [ 2.6, -3.0],
], dtype=float)

# --- Grid for gradient field ---
xmin, xmax = -4.0, 4.0
ymin, ymax = -4.0, 2.0

nx, ny = 29, 23
xs = np.linspace(xmin, xmax, nx)
ys = np.linspace(ymin, ymax, ny)
GX, GY = np.meshgrid(xs, ys)

U = np.zeros_like(GX)
V = np.zeros_like(GY)

for i in range(ny):
    for j in range(nx):
        g = np.array([GX[i, j], GY[i, j]])
        grad = vmf_grad_wrt_g_full(X, g, kappa=1.0)
        U[i, j] = grad[0]
        V[i, j] = grad[1]

# compute magnitudes
mag = np.sqrt(U**2 + V**2)

plt.figure(figsize=(10, 7))
# plot arrows using raw U,V (no normalization). Use `mag` as the color.
# Adjust `scale` to control arrow length (smaller -> longer arrows). Try 5..30 to taste.
Q = plt.quiver(GX, GY, U, V, mag, angles="xy", scale_units="xy", scale=8, cmap="viridis")

plt.colorbar(Q, label="gradient magnitude")
plt.plot(X[:, 0], X[:, 1], marker="o", linewidth=2)
plt.text(X[0, 0], X[0, 1] + 0.15, "start")
plt.text(X[-1, 0], X[-1, 1] - 0.35, "end")

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.gca().set_aspect("equal", adjustable="box")
plt.title("vMF progress term: gradient field (arrow lengths ~ magnitude)")
plt.xlabel("g_x")
plt.ylabel("g_y")
plt.grid(True)
plt.show()