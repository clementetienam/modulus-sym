"""
CuPy-Optimised BiCGSTAB Solver
================================
Van der Vorst (1992). SIAM J. Sci. Stat. Comput. 13(2), 631-644.

Key design decisions for reservoir simulation performance
---------------------------------------------------------
1. ILU is built ONCE externally and passed in as M — never inside the
   Newton loop.  Use build_ilu(A) before your Newton iterations and
   pass the result to bicgstab() on every call.
2. max_iter default is 200, not 10*N — reservoir Jacobians converge in
   20-100 iterations with a good preconditioner.
3. float() is used only for scalars that control Python branching.
   All heavy arithmetic stays as CuPy array ops.
4. best_x tracking removed from the hot loop — it costs an extra
   cp.linalg.norm and x.copy() every iteration.  Only saved on exit.
"""

import cupy as cp
import cupyx.scipy.sparse as sparse
from cupyx.scipy.sparse.linalg import spilu, LinearOperator
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  Build preconditioner — call ONCE per timestep, outside Newton loop
# ─────────────────────────────────────────────────────────────────────────────

def build_ilu(A) -> Optional[LinearOperator]:
    """
    Build an ILU(0) preconditioner from A.

    Call this ONCE before your Newton loop and pass the returned
    LinearOperator to bicgstab() on every Newton iteration.

    Parameters
    ----------
    A : cupyx CSR matrix

    Returns
    -------
    M : LinearOperator — preconditioner, or None if factorisation fails

    Example
    -------
    M = build_ilu(A_pressure)           # once per timestep
    for k in range(max_newton):
        x, info = bicgstab(A, b, M=M)  # reuse M every iteration
    """
    try:
        ilu = spilu(A)
        N   = A.shape[0]
        return LinearOperator((N, N), matvec=lambda v: ilu.solve(v))
    except Exception as e:
        print(f"[BiCGSTAB] ILU failed ({e}) — running unpreconditioned")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  BiCGSTAB solver
# ─────────────────────────────────────────────────────────────────────────────

def bicgstab(
    A,
    b: cp.ndarray,
    x0: Optional[cp.ndarray] = None,
    tol: float = 1e-6,
    atol: float = 0.0,
    max_iter: int = 200,              # sane default for reservoir systems
    M: Optional[LinearOperator] = None,
    verbose: bool = False,
) -> Tuple[cp.ndarray, int]:
    """
    Solve  A x = b  using BiCGSTAB, fully on GPU.

    Parameters
    ----------
    A        : cupyx CSR matrix or LinearOperator  (N, N)
    b        : cupy.ndarray  (N,)
    x0       : cupy.ndarray  (N,)  initial guess — pass previous timestep
               solution for warm start (cuts iteration count significantly)
    tol      : float  relative tolerance  ||r|| / ||b|| < tol
    atol     : float  absolute tolerance
    max_iter : int    maximum iterations (200 is enough for ILU-preconditioned
               reservoir systems; increase to 500 only if diverging)
    M        : LinearOperator — preconditioner from build_ilu()
               Build ONCE per timestep, reuse across Newton iterations.
    verbose  : bool

    Returns
    -------
    x    : cupy.ndarray — solution
    info : int  0=converged  1=max_iter reached  -1=breakdown
    """
    b = cp.asarray(b, dtype=cp.float64).ravel()
    N = b.shape[0]

    x = (cp.zeros(N, dtype=cp.float64) if x0 is None
         else cp.asarray(x0, dtype=cp.float64).ravel().copy())

    # Convergence threshold
    b_norm  = float(cp.linalg.norm(b))
    if b_norm == 0.0:
        return cp.zeros_like(b), 0
    tol_abs = max(tol * b_norm, atol)

    # Initial residual
    r  = b - A @ x
    r0 = r.copy()   # fixed shadow residual

    rho_prev = 1.0
    alpha    = 1.0
    omega    = 1.0
    v        = cp.zeros_like(b)
    p        = cp.zeros_like(b)

    for i in range(max_iter):

        rho = float(cp.dot(r0, r))

        # Breakdown: rho → 0
        if abs(rho) < 1e-300:
            if verbose:
                print(f"[BiCGSTAB] rho breakdown at iter {i}, restarting")
            r0       = r.copy()
            rho      = float(cp.dot(r0, r))
            rho_prev = 1.0; alpha = 1.0; omega = 1.0
            v        = cp.zeros_like(b)
            p        = cp.zeros_like(b)
            if abs(rho) < 1e-300:
                return x, -1

        beta = (rho / rho_prev) * (alpha / omega)
        p    = r + beta * (p - omega * v)

        # Precondition + SpMV 1
        p_hat = p if M is None else M.matvec(p)
        v     = A @ p_hat

        r0v = float(cp.dot(r0, v))
        if abs(r0v) < 1e-300:
            return x, -1
        alpha = rho / r0v

        s = r - alpha * v

        # Early exit on s
        if float(cp.linalg.norm(s)) < tol_abs:
            x = x + alpha * p_hat
            if verbose:
                print(f"[BiCGSTAB] converged (s) at iter {i+1}")
            return x, 0

        # Precondition + SpMV 2
        s_hat  = s if M is None else M.matvec(s)
        t      = A @ s_hat

        t_dot_t = float(cp.dot(t, t))
        omega   = (float(cp.dot(t, s)) / t_dot_t) if abs(t_dot_t) > 1e-300 else 0.0

        x = x + alpha * p_hat + omega * s_hat
        r = s - omega * t

        res_norm = float(cp.linalg.norm(r))
        if verbose and i % 10 == 0:
            print(f"  iter {i:4d}  ||r||/||b|| = {res_norm/b_norm:.3e}")

        if res_norm < tol_abs:
            if verbose:
                print(f"[BiCGSTAB] converged at iter {i+1}, "
                      f"||r||/||b|| = {res_norm/b_norm:.3e}")
            return x, 0

        if abs(omega) < 1e-300:
            return x, -1

        rho_prev = rho

    if verbose:
        print(f"[BiCGSTAB] max_iter={max_iter} reached, "
              f"||r||/||b|| = {res_norm/b_norm:.3e}")
    return x, 1


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience wrapper — for one-shot solves only
#  DO NOT call this inside a Newton loop
# ─────────────────────────────────────────────────────────────────────────────

def bicgstab_ilu(
    A,
    b: cp.ndarray,
    x0: Optional[cp.ndarray] = None,
    tol: float = 1e-6,
    atol: float = 0.0,
    max_iter: int = 200,
    verbose: bool = False,
) -> Tuple[cp.ndarray, int]:
    """
    One-shot BiCGSTAB + ILU solve.  Builds and discards the preconditioner.

    WARNING: Do NOT call this inside a Newton or timestep loop — it rebuilds
    the ILU factorisation every call.  Instead use:

        M = build_ilu(A)                        # once per timestep
        for k in range(newton_steps):
            x, info = bicgstab(A, b, M=M)      # reuse M
    """
    M = build_ilu(A)
    return bicgstab(A, b, x0=x0, tol=tol, atol=atol,
                    max_iter=max_iter, M=M, verbose=verbose)
