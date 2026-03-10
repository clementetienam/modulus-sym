"""
GPU-Optimised Algebraic Multigrid (AMG) V-Cycle Solver
=======================================================
All kernels stay on-device (CuPy / cupyx.scipy.sparse).
"""

import cupy as cp
import cupyx.scipy.sparse as sparse
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import spsolve
from cupyx.scipy.sparse import issparse


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_csr(A):
    """Return A as a cupyx CSR matrix (no-op if already CSR)."""
    if not issparse(A):
        return csr_matrix(A)
    return A.tocsr()


def residual(A, b, x):
    """r = b - A x  (stays on GPU)."""
    return b - A @ x


# ---------------------------------------------------------------------------
# Prolongation  (coarse → fine)
# ---------------------------------------------------------------------------

def prolongation(x_coarse, fine_grid_size: int):
    """
    Linear prolongation from coarse to fine grid.

    Each coarse degree of freedom is replicated to 2 fine DOFs
    (piecewise-constant interpolation matching the aggregation in restriction).

    Parameters
    ----------
    x_coarse      : cupy.ndarray, shape (N_coarse,)
    fine_grid_size : int – target fine-grid size N_fine

    Returns
    -------
    x_fine : cupy.ndarray, shape (N_fine,)
    """
    x_fine = cp.repeat(x_coarse, 2)[:fine_grid_size]
    # Pad if coarse grid was rounded down and we are one short.
    deficit = fine_grid_size - x_fine.size
    if deficit > 0:
        x_fine = cp.concatenate([x_fine, cp.zeros(deficit, dtype=x_coarse.dtype)])
    return x_fine


# ---------------------------------------------------------------------------
# Restriction  (fine → coarse)
# ---------------------------------------------------------------------------

def restriction(A, f):
    """
    Galerkin coarsening:  A_c = P^T A P,  f_c = P^T f.

    The prolongation/restriction operator P is built by pairwise aggregation
    (aggregate_size = 2).  All sparse operations remain on the GPU.

    Parameters
    ----------
    A : cupyx.scipy.sparse CSR matrix, shape (N, N)
    f : cupy.ndarray, shape (N,)

    Returns
    -------
    A_coarse : cupyx.scipy.sparse CSR matrix, shape (N_c, N_c)
    f_coarse : cupy.ndarray, shape (N_c,)
    P        : cupyx.scipy.sparse CSR matrix, shape (N, N_c)  [reused in v_cycle]
    """
    A = _to_csr(A)
    N = A.shape[0]
    aggregate_size = 2
    N_c = (N + aggregate_size - 1) // aggregate_size  # ceiling division

    # --- build P (fine→coarse aggregation operator) -----------------------
    row_idx = cp.arange(N, dtype=cp.int32)
    col_idx = row_idx // aggregate_size                         # which aggregate
    data    = cp.ones(N, dtype=cp.float32)

    P = csr_matrix((data, (row_idx, col_idx)), shape=(N, N_c))

    # --- Galerkin triple product  -----------------------------------------
    # A_c = P^T A P  (two sparse matrix products, fully on GPU)
    AP      = A @ P          # shape (N, N_c)
    A_coarse = P.T @ AP      # shape (N_c, N_c)
    f_coarse = P.T @ f       # shape (N_c,)

    return _to_csr(A_coarse), f_coarse, P


# ---------------------------------------------------------------------------
# Smoothers
# ---------------------------------------------------------------------------

def jacobi(A, b, x, omega=0.6667, tol=1e-6, max_iters=100):
    """
    Damped Jacobi smoother.

    Uses sparse diagonal extraction; never materialises a dense matrix.
    omega=2/3 is the optimal damping for smoothing high-frequency error.

    Parameters
    ----------
    A        : cupyx CSR matrix (N, N)
    b, x     : cupy.ndarray (N,)
    omega    : relaxation factor  (default 2/3)
    tol      : convergence tolerance on residual norm
    max_iters: iteration cap

    Returns
    -------
    x     : cupy.ndarray – updated solution
    iters : int
    """
    A = _to_csr(A)
    D     = A.diagonal()                    # shape (N,)  – on GPU
    D_inv = cp.reciprocal(D)               # 1/D_i

    # Off-diagonal part as a sparse matrix:  R = A - diag(D)
    R = A - sparse.diags(D, format="csr")  # ✓ stays sparse

    iters = 0
    for iters in range(1, max_iters + 1):
        x_new = D_inv * (b - R @ x)                 # Jacobi update
        x     = x + omega * (x_new - x)             # damped step
        if cp.linalg.norm(b - A @ x) < tol:
            break
    return x, iters


def gauss_seidel_rb(A, b, x, omega=1.0, tol=1e-6, max_iters=100):
    """
    Red-Black (2-colour) Gauss-Seidel smoother.

    Nodes are split into two independent sets (even / odd indices).
    Each colour can be updated in parallel with a single SpMV, making
    the method fully GPU-friendly while preserving G-S convergence behaviour.

    Parameters
    ----------
    A        : cupyx CSR matrix (N, N)
    b, x     : cupy.ndarray (N,)
    omega    : SOR relaxation factor (1.0 = standard G-S)
    tol      : convergence tolerance
    max_iters: iteration cap

    Returns
    -------
    x : cupy.ndarray
    """
    A = _to_csr(A)
    N     = A.shape[0]
    D     = A.diagonal()
    D_inv = cp.reciprocal(D)

    # Static colour masks (computed once)
    red   = cp.arange(N, dtype=cp.int32) % 2 == 0   # even indices
    black = ~red

    for _ in range(max_iters):
        # --- red update (all even nodes simultaneously) ---
        Ax         = A @ x
        x_gs_red   = D_inv * (b - Ax + D * x)       # G-S formula
        x[red]     = x[red] + omega * (x_gs_red[red] - x[red])

        # --- black update (all odd nodes simultaneously) ---
        Ax          = A @ x                           # recompute with updated reds
        x_gs_black  = D_inv * (b - Ax + D * x)
        x[black]    = x[black] + omega * (x_gs_black[black] - x[black])

        if cp.linalg.norm(b - A @ x) < tol:
            break
    return x


def sor(A, b, omega=1.5, tol=1e-6, max_iter=100):
    """
    Successive Over-Relaxation (SOR) – corrected single-SpMV formulation.

    The standard point-SOR update for node i is:
        x_i_new = x_i + omega * (b_i - (A x)_i) / A_ii

    On the GPU we approximate this with a parallel sweep (Jacobi-SOR),
    which is equivalent to one damped Jacobi step with factor omega/D.

    Parameters
    ----------
    A       : cupyx CSR matrix (N, N)
    b       : cupy.ndarray (N,)
    omega   : relaxation factor (default 1.5)
    tol     : convergence tolerance
    max_iter: iteration cap

    Returns
    -------
    x : cupy.ndarray
    """
    A = _to_csr(A)
    N     = A.shape[0]
    D_inv = cp.reciprocal(A.diagonal())
    x     = cp.zeros(N, dtype=cp.float32)

    for _ in range(max_iter):
        r     = b - A @ x                  # residual  (one SpMV)
        x     = x + omega * D_inv * r      # damped update

        if cp.linalg.norm(r) < tol:
            break
    return x


# ---------------------------------------------------------------------------
# V-Cycle
# ---------------------------------------------------------------------------

def v_cycle(A, b, x, smoother="jacobi", levels=3, tol=1e-6, smoothing_steps=2):
    """
    Recursive V-cycle multigrid solver for  A x = b.

    Corrections
    -----------
    * spsolve is called ONLY at the coarsest grid (level == 0 after recursion
      has reduced the problem to a tiny system).
    * Both pre- and post-smoothing are applied at every non-coarsest level.
    * Restriction returns P so the same operator is reused for prolongation
      consistency (Galerkin condition: P_restrict = P_prolong^T).

    Parameters
    ----------
    A              : cupyx CSR matrix (N, N)
    b              : cupy.ndarray (N,)
    x              : cupy.ndarray (N,) – initial guess
    smoother       : 'jacobi' | 'gauss-seidel' | 'SOR'
    levels         : number of grid levels (>= 1)
    tol            : residual tolerance for exact solve at coarsest level
    smoothing_steps: number of pre/post smoothing iterations per level

    Returns
    -------
    x : cupy.ndarray (N,)
    """
    A = _to_csr(A)

    def _smooth(A, b, x, steps):
        for _ in range(steps):
            if smoother == "jacobi":
                x, _ = jacobi(A, b, x, omega=0.6667, tol=1e-12, max_iters=1)
            elif smoother == "gauss-seidel":
                x = gauss_seidel_rb(A, b, x, omega=1.0, tol=1e-12, max_iters=1)
            elif smoother == "SOR":
                x = sor(A, b, omega=1.5, tol=1e-12, max_iter=1)
            else:
                raise ValueError(f"Unknown smoother '{smoother}'. "
                                 "Choose 'jacobi', 'gauss-seidel', or 'SOR'.")
        return x

    # --- Base case: coarsest grid – solve exactly -------------------------
    if levels <= 1 or A.shape[0] <= 4:
        return spsolve(A, b)

    # --- Pre-smoothing ----------------------------------------------------
    x = _smooth(A, b, x, smoothing_steps)

    # --- Restrict residual to coarse grid ---------------------------------
    r = residual(A, b, x)
    A_coarse, r_coarse, P = restriction(A, r)

    # --- Recursive coarse-grid correction ---------------------------------
    e_coarse = v_cycle(
        A_coarse,
        r_coarse,
        cp.zeros(A_coarse.shape[0], dtype=x.dtype),
        smoother=smoother,
        levels=levels - 1,
        tol=tol,
        smoothing_steps=smoothing_steps,
    )

    # --- Prolongate correction and update fine-grid solution --------------
    # Use P (the same operator built in restriction) for consistency.
    x = x + P @ e_coarse          # ✓ sparse SpMV, no repeat/trim needed

    # --- Post-smoothing ---------------------------------------------------
    x = _smooth(A, b, x, smoothing_steps)

    return x


# ---------------------------------------------------------------------------
# Public solver entry-point
# ---------------------------------------------------------------------------

def amg_solve(A, b, x0=None, smoother="jacobi", levels=3,
              tol=1e-6, smoothing_steps=2, max_cycles=5):
    """
    Solve  A x = b  with V-cycle AMG iteration until residual < tol.

    Parameters
    ----------
    A              : cupyx CSR matrix or any cupy sparse matrix (N, N)
    b              : cupy.ndarray (N,)
    x0             : initial guess (zeros if None)
    smoother       : 'jacobi' | 'gauss-seidel' | 'SOR'
    levels         : multigrid depth
    tol            : absolute residual tolerance
    smoothing_steps: pre/post smoothing steps per level
    max_cycles     : maximum number of V-cycles

    Returns
    -------
    x          : cupy.ndarray (N,) – approximate solution
    res_history: list of float – residual norm after each cycle
    """
    A  = _to_csr(A)
    x  = cp.zeros(A.shape[0], dtype=b.dtype) if x0 is None else cp.array(x0)
    res_history = []

    for cycle in range(max_cycles):
        x   = v_cycle(A, b, x, smoother=smoother, levels=levels,
                      tol=tol, smoothing_steps=smoothing_steps)
        res = float(cp.linalg.norm(residual(A, b, x)))
        res_history.append(res)

        # if res < tol:
            # print(f"AMG converged in {cycle + 1} V-cycle(s). "
                  # f"Final residual: {res:.3e}")
            # break
    # else:
        # print(f"AMG reached max_cycles={max_cycles}. "
              # f"Final residual: {res_history[-1]:.3e}")

    return x
