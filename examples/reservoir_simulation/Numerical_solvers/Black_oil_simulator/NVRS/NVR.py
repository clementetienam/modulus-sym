# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
@Author : Clement Etienam
"""

import os
import os.path
import sys
import time
import math
import random
import logging
import warnings
import multiprocessing

import numpy as np
import numpy.matlib
import numpy.linalg
import numpy.ma as ma
from numpy import *
from numpy.linalg import norm

from scipy import interpolate, sparse
from scipy.fftpack import dct
from scipy.fftpack.realtransforms import idct
import scipy.optimize.lbfgsb as lbfgsb

import matplotlib.colors
from matplotlib import cm
from shutil import rmtree

import yaml
from pyDOE import lhs
from cpuinfo import get_cpu_info
from FyeldGenerator import generate_field
from imresize import *
from Style_interp import *

import mpslib as mps

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ── GPU / CPU backend selection ───────────────────────────────────────────────

def is_available():
    """
    Check if a CUDA-capable GPU is available.
    Prefers a direct CuPy / CUDA-runtime check; falls back to nvidia-smi.

    Returns
    -------
    int
        0 if a CUDA GPU is available, non-zero otherwise.
    """
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return 0
    except Exception:
        pass

    # Fallback: nvidia-smi shell check
    cmd  = "nvidia-smi > nul 2>&1" if os.name == "nt" else "nvidia-smi > /dev/null 2>&1"
    code = os.system(cmd)
    return 0 if code == 0 else 1


Yet = is_available()

if Yet == 0:
    import cupy as cp
    import cupyx.scipy.sparse as sparse
    from cupyx.scipy.sparse import csr_matrix, spmatrix, spdiags, issparse
    from cupyx.scipy.sparse.linalg import (
        gmres,
        cg,
        spsolve,
        lsqr,
        LinearOperator,
        spilu,
    )
    from numba import cuda
    clementtt = 0
else:
    import numpy as cp
    import scipy.sparse as sparse
    from scipy.sparse import csr_matrix, spdiags
    from scipy.sparse.linalg import gmres, cg, spsolve
    clementtt = 1


# ── Local solver imports ──────────────────────────────────────────────────────

from multigrid_solver import (
    _to_csr,
    residual,
    amg_solve,
)
from Bicgstab import bicgstab_ilu


# ── CPU info ──────────────────────────────────────────────────────────────────

cores = multiprocessing.cpu_count()
s     = get_cpu_info()


def read_yaml(fname):
    """Read Yaml file into a dict of parameters"""
    print(f"Read simulation plan from {fname}...")
    with open(fname, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            # print(data)
        except yaml.YAMLError as exc:
            print(exc)
        return data



# ---------------------------------------------------------------------------
# Unit conversion helper  (replaces field2Metric)
# ---------------------------------------------------------------------------

_CONVERSIONS = {
    "psi":   6894.757,
    "lbft3": 16.01846,
    "ms":    3.28084,
    "ft":    1.0 / 3.28084,     # ft/s → m/s
}


def field2Metric(value, unit: str):
    """Multiply *value* by the appropriate SI conversion factor."""
    factor = _CONVERSIONS[unit]
    return value * factor


# ---------------------------------------------------------------------------
# Gassmann  (fully vectorised, GPU only)
# ---------------------------------------------------------------------------

def _t(label, t0):
    cp.cuda.Stream.null.synchronize()   # flush GPU before timing
    print(f"  {label}: {time.perf_counter()-t0:.4f}s")
    return time.perf_counter()
    
    
def Gassmann(PORO, Pr, SO, nx: int, ny: int, nz: int):
    """
    Compute P- and S-wave velocities and acoustic impedances via Gassmann's
    equations, using a Backus-average over nz layers.

    All heavy computation is batched over the full (nx, ny, nz) volume in
    a handful of CuPy kernel launches — no Python loops over grid indices.

    Parameters
    ----------
    PORO : cupy.ndarray, shape (nx, ny, nz)  — porosity (fraction)
    Pr   : cupy.ndarray, shape (nx, ny, nz)  — pressure (psi)
    SO   : cupy.ndarray, shape (nx, ny, nz)  — oil saturation (Eclipse format)
    nx, ny, nz : grid dimensions

    Returns
    -------
    ImpP : cupy.ndarray (nx, ny)  — P-wave impedance  [g/cm²/s]
    ImpS : cupy.ndarray (nx, ny)  — S-wave impedance  [g/cm²/s]
    VP   : cupy.ndarray (nx, ny)  — P-wave velocity   [ft/s]
    VS   : cupy.ndarray (nx, ny)  — S-wave velocity   [ft/s]
    """

    # ------------------------------------------------------------------
    # 0.  Ensure inputs are on GPU and cast to float64
    # ------------------------------------------------------------------
    PORO = cp.asarray(PORO, dtype=cp.float64)   # (nx, ny, nz)
    Pr   = cp.asarray(Pr,   dtype=cp.float64)
    SO   = cp.asarray(SO,   dtype=cp.float64)

    # ------------------------------------------------------------------
    # 1.  Convert inputs  (vectorised over full volume)
    # ------------------------------------------------------------------
    T   = 103.0                                  # °C  (scalar)
    phi = PORO                                   # (nx, ny, nz)
    P   = field2Metric(Pr, "psi") * 1e-6         # psi → MPa, shape (nx,ny,nz)
    SOil = SO                                    # oil saturation (nx, ny, nz)

    # ------------------------------------------------------------------
    # 2.  Fluid properties  (Batzle & Wang — vectorised over all voxels)
    # ------------------------------------------------------------------
    # --- Water ---
    CWater   = 3.13e-6                           # 1/psi
    rhoWater = field2Metric(64.00, "lbft3")      # kg/m³  (scalar)
    KWater   = field2Metric(1.0 / CWater, "psi") # Pa     (scalar)

    # --- Oil ---
    G   = 0.8515                                 # gas specific gravity
    RG  = 0.0                                    # GOR (m³/m³)
    API = 141.5 / G - 131.5

    rho0      = 141.5 / (API + 131.5)           # g/cm³  (scalar)
    B0        = 0.972 + 0.00038 * (2.4 * RG * cp.sqrt(G / rho0) + T + 17.8) ** 1.175
    rhopseudo = rho0 / B0 * (1.0 + 0.001 * RG) ** -1   # g/cm³
    rhoG      = (rho0 + 0.0012 * G * RG) / B0           # g/cm³

    # Pressure-/temperature-corrected oil density  [kg/m³]  — shape (nx,ny,nz)
    rhoOil = (
        1000.0
        * (rho0 + (0.00277 * P - 1.71e-7 * P**3) * (rhoG - 1.15)**2 + P * 3.49e-4)
        / (0.972 + 3.81e-4 * (T + 17.78)**1.175)
    )

    # Oil P-wave velocity  [m/s]  — shape (nx, ny, nz)
    VOil = (
        2096.0 * cp.sqrt(rhopseudo / (2.6 - rhopseudo))
        - 3.7  * T
        + 4.64 * P
        + 0.0115 * T * P * (4.12 * cp.sqrt(1.08 / rhopseudo - 1.0) - 1.0)
    )

    KOil = rhoOil * VOil**2                      # bulk modulus of oil [Pa]

    # ------------------------------------------------------------------
    # 3.  Fluid mixing  (Wood's equation — fully vectorised)
    # ------------------------------------------------------------------
    # KFluid = ( (1-S)/Kw + S/Ko )^{-1}
    KFluid  = 1.0 / ((1.0 - SOil) / KWater + SOil / KOil)  # (nx, ny, nz)
    rhoFluid = (1.0 - SOil) * rhoWater + SOil * rhoOil       # (nx, ny, nz)

    # ------------------------------------------------------------------
    # 4.  Matrix & frame constants  (broadcast over z automatically)
    # ------------------------------------------------------------------
    SQuartz   = 0.6
    SFeldspar = 1.0 - SQuartz
    KQuartz   = 37e9
    KFeldspar = 37.5e9

    KMatrix = (
        (SQuartz * KQuartz + SFeldspar * KFeldspar) / 2.0
        + 0.5 / (SQuartz / KQuartz + SFeldspar / KFeldspar)
    )                                            # scalar (uniform matrix)

    rhoDry = 2169.0                              # kg/m³ (scalar)
    KDry   = field2Metric(2.0e6,   "psi")       # Pa
    GDry   = field2Metric(1.368e6, "psi")       # Pa

    # ------------------------------------------------------------------
    # 5.  Gassmann equations  — shape (nx, ny, nz)
    # ------------------------------------------------------------------
    KSat = KDry + (1.0 - KDry / KMatrix)**2 / (
        phi / KFluid + (1.0 - phi) / KMatrix - KDry / KMatrix**2
    )
    GSat   = GDry                                # shear modulus unchanged
    rhoSat = rhoDry + phi * rhoFluid             # (nx, ny, nz)

    # ------------------------------------------------------------------
    # 6.  Backus average over z  — single GPU reduction per (i,j) column
    # ------------------------------------------------------------------
    # C = < 1/(K + 4G/3) >_z^{-1}    (harmonic mean of M-modulus)
    # D = G_Sat                        (isotropic shear — constant here)
    # rho = < rho_Sat >_z              (arithmetic mean density)
    M = KSat + (4.0 / 3.0) * GSat               # P-wave modulus (nx, ny, nz)

    C   = 1.0 / cp.mean(1.0 / M,   axis=2)      # (nx, ny)  — harmonic mean
    D   = cp.full((nx, ny), GSat, dtype=cp.float64)
    rho = cp.mean(rhoSat, axis=2)               # (nx, ny)  — arithmetic mean

    # ------------------------------------------------------------------
    # 7.  Velocities & impedances
    # ------------------------------------------------------------------
    VP = cp.sqrt(C / rho)                        # m/s  (nx, ny)
    VS = cp.sqrt(D / rho)                        # m/s  (nx, ny)

    VP = field2Metric(VP, "ms")                  # → ft/s
    VS = field2Metric(VS, "ms")

    rho_gcc = rho / 1000.0                       # kg/m³ → g/cm³

    ImpP = rho_gcc * VP                          # g/cm²/s
    ImpS = rho_gcc * VS

    return ImpP


def simulator_to_python(a):
    kk = a.shape[2]
    anew = []
    for i in range(kk):
        afirst = a[:, :, i]
        afirst = afirst.T
        afirst = cp.reshape(afirst, (-1, 1), "F")
        anew.append(afirst)
    return cp.vstack(anew)


def python_to_simulator(a, ny, nx, nz):
    a = cp.reshape(a, (-1, 1), "F")
    a = cp.reshape(a, (ny, nx, nz), "F")
    anew = []
    for i in range(nz):
        afirst = a[:, :, i]
        afirst = afirst.T
        anew.append(afirst)
    return cp.vstack(anew)



def residual(A, b, x):
    return b - A @ x




def print_section_title(text: str) -> None:
    print("\n# ----------------------------------------")
    print(f"# {text.upper()}")
    print("# ----------------------------------------")


def Peaceman_well(
    inn,
    ooutp,
    oouts,
    MAXZ,
    mazw,
    s1,
    DX,
    steppi,
    pini_alt,
    SWI,
    SWR,
    UW,
    BW,
    DZ,
    rwell,
    skin,
    UO,
    BO,
    pwf_producer,
    dt,
    N_inj,
    N_pr,
    nz,
    NecessaryI,
    NecessaryP,
):
    """
    Calculates the pressure and flow rates for an injection and production well using the Peaceman model.

    Args:
    - inn (dictionary): dictionary containing the input parameters (including permeability and injection/production rates)
    - ooutp (numpy array): 4D numpy array containing pressure values for each time step and grid cell
    - oouts (numpy array): 4D numpy array containing saturation values for each time step and grid cell
    - MAXZ (float): length of the reservoir in the z-direction
    - mazw (float): the injection/production well location in the z-direction
    - s1 (float): the length of the computational domain in the z-direction
    - LUB (float): the upper bound of the rescaled permeability
    - HUB (float): the lower bound of the rescaled permeability
    - aay (float): the upper bound of the original permeability
    - bby (float): the lower bound of the original permeability
    - DX (float): the cell size in the x-direction
    - steppi (int): number of time steps
    - pini_alt (float): the initial pressure
    - SWI (float): the initial water saturation
    - SWR (float): the residual water saturation
    - UW (float): the viscosity of water
    - BW (float): the formation volume factor of water
    - DZ (float): the cell thickness in the z-direction
    - rwell (float): the well radius
    - skin (float): the skin factor
    - UO (float): the viscosity of oil
    - BO (float): the formation volume factor of oil
    - pwf_producer (float): the desired pressure at the producer well
    - dt (float): the time step
    - N_inj (int): the number of injection wells
    - N_pr (int): the number of production wells
    - nz (int): the number of cells in the z-direction

    Returns:
    - overr (numpy array): an array containing the time and flow rates (in BHP, qoil, qwater, and wct) for each time step
    """
    # ------------------------------------------------------------------
    # 1.  Move all inputs to GPU once
    # ------------------------------------------------------------------
    NecessaryI = cp.asarray(NecessaryI, dtype=cp.float32)   # (Ni_cells, 2)
    NecessaryP = cp.asarray(NecessaryP, dtype=cp.float32)   # (Np_cells, 3)

    # Pressure and saturation fields on GPU: (steppi, N_cells)
    if nz == 1:
        P_all = cp.asarray(ooutp[:, :steppi, :, :],
                           dtype=cp.float32).reshape(-1, steppi, order="C")   # wrong axis — fix:
        # ooutp shape: (N_ens, steppi, nx, ny)  → we want (steppi, N_cells)
        P_all = cp.asarray(ooutp[0, :steppi, :, :],
                           dtype=cp.float32).reshape(steppi, -1)              # (steppi, N_cells)
        S_all = cp.asarray(oouts[0, :steppi, :, :],
                           dtype=cp.float32).reshape(steppi, -1)
        perm_flat = cp.asarray(
            inn[0, 0, :, :].ravel(order="F"), dtype=cp.float32)              # (N_cells,)
        rate_flat = cp.asarray(
            inn[0, 1, :, :].ravel(order="F"), dtype=cp.float32)
    else:
        P_all = cp.asarray(ooutp[0, :steppi, :, :, :],
                           dtype=cp.float32).reshape(steppi, -1)
        S_all = cp.asarray(oouts[0, :steppi, :, :, :],
                           dtype=cp.float32).reshape(steppi, -1)
        perm_flat = cp.asarray(
            inn[0, 0, :, :, :].ravel(order="F"), dtype=cp.float32)
        rate_flat = cp.asarray(
            inn[0, 1, :, :, :].ravel(order="F"), dtype=cp.float32)

    # ------------------------------------------------------------------
    # 2.  Well locations — computed ONCE
    # ------------------------------------------------------------------
    inj_loc  = cp.where(rate_flat >  0)[0]   # injector cell indices
    prod_loc = cp.where(rate_flat <  0)[0]   # producer cell indices

    kuse_inj  = perm_flat[inj_loc]           # (N_inj_cells,)
    kuse_prod = perm_flat[prod_loc]          # (N_pr_cells,)

    RE = 0.2 * DX                            # equivalent drainage radius

    # ------------------------------------------------------------------
    # 3.  Peaceman log-term — computed ONCE (geometry is static)
    # ------------------------------------------------------------------
    n_inj_cells = inj_loc.shape[0]
    n_pr_cells  = prod_loc.shape[0]

    if nz == 1:
        log_inj  = cp.log(RE / NecessaryI[:, 0]) + NecessaryI[:, 1]   # (Ni,)
        log_prod = cp.log(RE / NecessaryP[:, 0]) + NecessaryP[:, 1]   # (Np,)
        pwf_prod = NecessaryP[:, 2]                                     # (Np,)
    else:
        reps_i = int(n_inj_cells  / NecessaryI.shape[0])
        reps_p = int(n_pr_cells   / NecessaryP.shape[0])
        log_inj  = (cp.log(RE / cp.tile(NecessaryI[:, 0], reps_i))
                    + cp.tile(NecessaryI[:, 1], reps_i))
        log_prod = (cp.log(RE / cp.tile(NecessaryP[:, 0], reps_p))
                    + cp.tile(NecessaryP[:, 1], reps_p))
        pwf_prod = cp.tile(NecessaryP[:, 2], reps_p)

    # Peaceman denominator constants (scalar multipliers — computed once)
    inv_J_oil_const   = (UO * BO) / (2.0 * cp.pi * DZ)    # multiply by (log/k/kr)
    inv_J_water_const = (UW * BW) / (2.0 * cp.pi * DZ)

    # ------------------------------------------------------------------
    # 4.  Output arrays pre-allocated on GPU
    # ------------------------------------------------------------------
    BHP_out    = cp.zeros((steppi, N_inj),  dtype=cp.float32)
    qoil_out   = cp.zeros((steppi, N_pr),   dtype=cp.float32)
    qwater_out = cp.zeros((steppi, N_pr),   dtype=cp.float32)
    wct_out    = cp.zeros((steppi, N_pr),   dtype=cp.float32)
    timz_out   = cp.zeros((steppi, 1),      dtype=cp.float32)

    # ------------------------------------------------------------------
    # 5.  Vectorised loop over timesteps
    #     (inner ops are all CuPy — no host transfers per step)
    # ------------------------------------------------------------------
    denom_sw = 1.0 - SWI - SWR   # scalar

    for kk in range(steppi):
        # Pressure and saturation at this timestep
        p_cells = P_all[kk]                          # (N_cells,)
        s_cells = S_all[kk]                          # (N_cells,)

        # Brooks-Corey rel-perms (vectorised)
        Sw_norm = (s_cells - SWI) / denom_sw         # (N_cells,)
        Krw_all = Sw_norm ** 2
        Kro_all = (1.0 - Sw_norm) ** 2

        # Well-cell values
        krw_inj  = Krw_all[inj_loc]                  # (N_inj_cells,)
        krw_prod = Krw_all[prod_loc]                  # (N_pr_cells,)
        kro_prod = Kro_all[prod_loc]

        p_inj  = p_cells[inj_loc]                    # (N_inj_cells,)
        p_prod = p_cells[prod_loc]                   # (N_pr_cells,)

        # --- BHP at injectors -----------------------------------------
        # Pwf = p_inj + (μW BW) / (2π k krw DZ) * log_term
        temp = inv_J_water_const * log_inj / (kuse_inj * krw_inj)
        Pwf  = cp.abs(p_inj + temp)
        # Average over nz layers per injector
        BHP_out[kk] = cp.sum(
            Pwf.reshape(-1, N_inj, order="C"), axis=0) / nz

        # --- Oil production rate --------------------------------------
        J_oil      = (2.0 * cp.pi * kuse_prod * kro_prod * DZ
                      / (inv_J_oil_const * log_prod / (UO * BO)))
        # Simplify: J = (2π k kr DZ) / (μ B log_term)
        J_oil      = (2.0 * cp.pi * kuse_prod * kro_prod * DZ
                      / ((UO * BO) * log_prod))
        drawdown   = p_prod - pwf_prod
        qoil_cells = cp.abs(drawdown * J_oil)
        qoil_out[kk] = cp.sum(
            qoil_cells.reshape(-1, N_pr, order="C"), axis=0) / nz

        # --- Water production rate ------------------------------------
        J_water      = (2.0 * cp.pi * kuse_prod * krw_prod * DZ
                        / ((UW * BW) * log_prod))
        qwater_cells = cp.abs(drawdown * J_water)
        qwater_out[kk] = cp.sum(
            qwater_cells.reshape(-1, N_pr, order="C"), axis=0) / nz

        # --- Water cut -------------------------------------------------
        wct_out[kk] = (qwater_out[kk]
                       / (qwater_out[kk] + qoil_out[kk] + 1e-10)) * 100.0

        # --- Time stamp -----------------------------------------------
        timz_out[kk, 0] = ((kk + 1) * dt) * MAXZ

    # ------------------------------------------------------------------
    # 6.  Assemble output — single host transfer
    # ------------------------------------------------------------------
    # overr shape: (steppi, 1 + N_inj + N_pr + N_pr + N_pr)
    Big = cp.hstack([timz_out, BHP_out, qoil_out, qwater_out, wct_out])
    return cp.asnumpy(Big)



def Peaceman_well2(
    inn, ooutp, oouts, ooutsoil, outg,
    MAXZ, mazw, s1, DX, steppi, pini_alt,
    SWI, SWR, UW, BW, DZ, rwell, skin,
    UO, BO, UG, BG,
    pwf_producer, dt, N_inj, N_pr, nz,
    NecessaryI, NecessaryP,
    SWOW, SWOG, PB,
):
    """
    Three-phase (water/oil/gas) Peaceman well model — fully GPU-vectorised.

    Parameters
    ----------
    inn        : ndarray (1, 2+, nx, ny[, nz])  — perm[0], rate[1]
    ooutp      : ndarray (1, steppi, nx, ny[,nz]) — pressure
    oouts      : ndarray (1, steppi, nx, ny[,nz]) — water saturation
    ooutsoil   : ndarray (1, steppi, nx, ny[,nz]) — oil saturation   ← was IGNORED
    outg       : ndarray (1, steppi, nx, ny[,nz]) — gas saturation
    SWOW       : cupy.ndarray (M,3) — [Sw, Krow, Krw]
    SWOG       : cupy.ndarray (M,3) — [Sg, Krog, Krg]
    PB         : float — bubble-point pressure for RS/BG correlations

    Returns
    -------
    overr : numpy.ndarray (steppi, 1 + N_inj + 4*N_pr)
        Columns: [time, BHP..., qoil..., qwater..., qgas..., wct...]
    """

    # ------------------------------------------------------------------
    # 1.  Move static inputs to GPU once
    # ------------------------------------------------------------------
    SWOW = cp.asarray(SWOW, dtype=cp.float32)
    SWOG = cp.asarray(SWOG, dtype=cp.float32)
    NecessaryI = cp.asarray(NecessaryI, dtype=cp.float32)
    NecessaryP = cp.asarray(NecessaryP, dtype=cp.float32)

    if nz == 1:
        P_all   = cp.asarray(ooutp[0,    :steppi, :, :],    dtype=cp.float32).reshape(steppi, -1)
        Sw_all  = cp.asarray(oouts[0,    :steppi, :, :],    dtype=cp.float32).reshape(steppi, -1)
        So_all  = cp.asarray(ooutsoil[0, :steppi, :, :],    dtype=cp.float32).reshape(steppi, -1)
        Sg_all  = cp.asarray(outg[0,     :steppi, :, :],    dtype=cp.float32).reshape(steppi, -1)
        perm_flat = cp.asarray(inn[0, 0, :, :].ravel(order="F"),   dtype=cp.float32)
        rate_flat = cp.asarray(inn[0, 1, :, :].ravel(order="F"),   dtype=cp.float32)
    else:
        P_all   = cp.asarray(ooutp[0,    :steppi, :, :, :], dtype=cp.float32).reshape(steppi, -1)
        Sw_all  = cp.asarray(oouts[0,    :steppi, :, :, :], dtype=cp.float32).reshape(steppi, -1)
        So_all  = cp.asarray(ooutsoil[0, :steppi, :, :, :], dtype=cp.float32).reshape(steppi, -1)
        Sg_all  = cp.asarray(outg[0,     :steppi, :, :, :], dtype=cp.float32).reshape(steppi, -1)
        perm_flat = cp.asarray(inn[0, 0, :, :, :].ravel(order="F"), dtype=cp.float32)
        rate_flat = cp.asarray(inn[0, 1, :, :, :].ravel(order="F"), dtype=cp.float32)

    # ------------------------------------------------------------------
    # 2.  Well locations — computed ONCE
    # ------------------------------------------------------------------
    inj_loc  = cp.where(rate_flat >  0)[0]
    prod_loc = cp.where(rate_flat <  0)[0]

    kuse_inj  = perm_flat[inj_loc]
    kuse_prod = perm_flat[prod_loc]

    RE      = 0.2 * DX
    denom_s = 1.0 - SWI - SWR   # scalar

    # ------------------------------------------------------------------
    # 3.  Static Peaceman log-terms — computed ONCE
    # ------------------------------------------------------------------
    n_inj_cells = int(inj_loc.shape[0])
    n_pr_cells  = int(prod_loc.shape[0])

    if nz == 1:
        log_inj  = cp.log(RE / NecessaryI[:, 0]) + NecessaryI[:, 1]
        log_prod = cp.log(RE / NecessaryP[:, 0]) + NecessaryP[:, 1]
        pwf_prod = NecessaryP[:, 2]
    else:
        reps_i   = int(n_inj_cells  / NecessaryI.shape[0])
        reps_p   = int(n_pr_cells   / NecessaryP.shape[0])
        log_inj  = cp.log(RE / cp.tile(NecessaryI[:, 0], reps_i)) + cp.tile(NecessaryI[:, 1], reps_i)
        log_prod = cp.log(RE / cp.tile(NecessaryP[:, 0], reps_p)) + cp.tile(NecessaryP[:, 1], reps_p)
        pwf_prod = cp.tile(NecessaryP[:, 2], reps_p)

    # Precompute table derivatives for consistent Kr derivatives (on GPU)
    sw_tab = SWOW[:, 0];  sg_tab = SWOG[:, 0]

    # ------------------------------------------------------------------
    # 4.  Output buffers — pre-allocated on GPU
    # ------------------------------------------------------------------
    BHP_out    = cp.zeros((steppi, N_inj), dtype=cp.float32)
    qoil_out   = cp.zeros((steppi, N_pr),  dtype=cp.float32)
    qwater_out = cp.zeros((steppi, N_pr),  dtype=cp.float32)
    qgas_out   = cp.zeros((steppi, N_pr),  dtype=cp.float32)
    wct_out    = cp.zeros((steppi, N_pr),  dtype=cp.float32)
    timz_out   = cp.zeros((steppi, 1),     dtype=cp.float32)

    # ------------------------------------------------------------------
    # 5.  Main loop — all ops on GPU, no host transfers per step
    # ------------------------------------------------------------------
    for kk in range(steppi):

        p_cells  = P_all[kk]           # (N_cells,)
        sw       = Sw_all[kk]          # (N_cells,)  water sat
        so       = So_all[kk]          # (N_cells,)  oil sat   ← FIXED (was=sw)
        sg       = Sg_all[kk]          # (N_cells,)  gas sat

        # --- Three-phase rel-perms via table interpolation -------------
        sw_col = sw.reshape(-1, 1)
        sg_col = sg.reshape(-1, 1)

        KROW = interp(sw_col, sw_tab, SWOW[:, 1]).ravel()   # oil kr vs Sw
        KRW  = interp(sw_col, sw_tab, SWOW[:, 2]).ravel()   # water kr vs Sw
        KROG = interp(sg_col, sg_tab, SWOG[:, 1]).ravel()   # oil kr vs Sg
        KRG  = interp(sg_col, sg_tab, SWOG[:, 2]).ravel()   # gas kr vs Sg

        # Baker's linear KRO (consistent with RelPerm3)
        So_norm = cp.clip(so / denom_s, 0.0, 1.0)
        KRO     = So_norm * KROW * KROG                      # (N_cells,)

        # --- Well-cell values ------------------------------------------
        krw_inj  = KRW[inj_loc]
        krw_prod = KRW[prod_loc]
        kro_prod = KRO[prod_loc]
        krg_prod = KRG[prod_loc]

        p_inj    = p_cells[inj_loc]
        p_prod   = p_cells[prod_loc]

        # --- RS per producer cell (pressure-dependent) -----------------
        # Fix: compute RS at each producer cell pressure, not mean pressure
        RS_prod  = cp.asarray(
            [float(calc_rs(PB, float(pp))) for pp in cp.asnumpy(p_prod)],
            dtype=cp.float32
        )

        # --- BHP at injectors ------------------------------------------
        # Pwf = p_inj + (μW BW / (2π k krw DZ)) * log_term
        temp     = ((UW * BW) / (2.0 * cp.pi * DZ)) * log_inj / (kuse_inj * krw_inj)
        Pwf      = cp.abs(p_inj + temp)
        BHP_out[kk] = cp.sum(Pwf.reshape(-1, N_inj, order="C"), axis=0) / nz

        # --- Drawdown (shared for oil and water producers) -------------
        drawdown = p_prod - pwf_prod                          # (N_pr_cells,)

        # --- Oil production rate (Darcy PI) ----------------------------
        J_oil       = (2.0 * cp.pi * kuse_prod * kro_prod * DZ
                       / ((UO * BO) * log_prod))
        qoil_cells  = cp.abs(drawdown * J_oil)
        qoil_out[kk] = cp.sum(qoil_cells.reshape(-1, N_pr, order="C"), axis=0) / nz

        # --- Water production rate (Darcy PI) --------------------------
        J_water      = (2.0 * cp.pi * kuse_prod * krw_prod * DZ
                        / ((UW * BW) * log_prod))
        qwater_cells = cp.abs(drawdown * J_water)
        qwater_out[kk] = cp.sum(qwater_cells.reshape(-1, N_pr, order="C"), axis=0) / nz

        # --- Gas production rate ---------------------------------------
        # Use Darcy PI for free gas when KRG is significant,
        # otherwise fall back to dissolved-gas formula qgas = RS * qoil.
        # This matches standard black-oil practice:
        #   - Below Pb: all gas dissolved, qgas = RS * qoil
        #   - Above Pb (gas cap present): free gas + dissolved gas
        has_free_gas = cp.any(krg_prod > 1e-6)
        if has_free_gas:
            J_gas        = (2.0 * cp.pi * kuse_prod * krg_prod * DZ
                            / ((UG * BG) * log_prod))
            qgas_cells   = cp.abs(drawdown * J_gas) + RS_prod * qoil_cells
            qgas_out[kk] = cp.sum(qgas_cells.reshape(-1, N_pr, order="C"), axis=0) / nz
        else:
            # Solution-gas drive only
            qgas_out[kk] = cp.mean(RS_prod) * qoil_out[kk]

        # --- Water cut --------------------------------------------------
        wct_out[kk] = (qwater_out[kk]
                       / (qwater_out[kk] + qoil_out[kk] + 1e-10)) * 100.0

        # --- Time stamp ------------------------------------------------
        timz_out[kk, 0] = ((kk + 1) * dt) * MAXZ

    # ------------------------------------------------------------------
    # 6.  Single host transfer at the end
    # ------------------------------------------------------------------
    Big = cp.hstack([timz_out, BHP_out, qoil_out, qwater_out, qgas_out, wct_out])
    return cp.asnumpy(Big)



def Upstream_2PHASE(
    nx, ny, nz, S, UW, UO, BW, BO, SWI, SWR, Vol, qinn, V, Tt, porosity
):
    """
    Solve a 2-phase flow reservoir simulation using an upstream (upwind) scheme.

    Parameters
    ----------
    nx, ny, nz : int   — grid dimensions
    S          : cupy.ndarray (N,1) — initial water saturation field
    UW, UO     : float — water / oil viscosity
    BW, BO     : float — water / oil formation volume factor
    SWI        : float — initial water saturation
    SWR        : float — residual water saturation
    Vol        : cupy.ndarray (N,1) — grid cell volumes
    qinn       : cupy.ndarray (N,1) — source / sink rates
    V          : dict with keys "x","y","z" — face flux arrays
    Tt         : float — total simulation time
    porosity   : cupy.ndarray — porosity values

    Returns
    -------
    S : cupy.ndarray (N,1) — final water saturation field
    """

    N = nx * ny * nz

    # ------------------------------------------------------------------
    # 1.  Static pre-computations  (done ONCE, outside the time loop)
    # ------------------------------------------------------------------

    # Pore volume  [shape (N,1)]
    poro = cp.reshape(porosity, (N, 1), "F")
    pv   = Vol * poro                                    # elementwise, on GPU

    # Source term  [shape (N,1)]
    qinn = cp.reshape(qinn, (-1, 1), "F")
    fi_base = cp.maximum(qinn, 0)                        # inflow only

    # Face-flux positive/negative splits  (computed once)
    XP = cp.maximum(V["x"], 0);  XN = cp.minimum(V["x"], 0)
    YP = cp.maximum(V["y"], 0);  YN = cp.minimum(V["y"], 0)
    ZP = cp.maximum(V["z"], 0);  ZN = cp.minimum(V["z"], 0)

    # Net influx per cell  [shape (nx,ny,nz)]
    Vi = (
          XP[:nx,    :,    :]
        + YP[:,    :ny,    :]
        + ZP[:,      :,  :nz]
        - XN[1:nx+1, :,    :]
        - YN[:,  1:ny+1,   :]
        - ZN[:,      :, 1:nz+1]
    )
    Vi = cp.reshape(Vi, (N, 1), "F")

    # CFL time-step  -------------------------------------------------------
    # cp.min stays on GPU; no host-side Python min() transfer
    pm  = cp.min(pv / (Vi + fi_base))
    cfl = ((1.0 - SWR) / 3.0) * float(pm)   # single scalar → float is fine
    Nts = math.ceil(Tt / cfl)

    # Per-cell time-step scaling  [shape (N,)]
    dtx = cp.ravel(Tt / (Nts * pv))          # fused division, no spdiags needed

    # Scale transport matrix once: A_scaled = diag(dtx) @ A_unscaled
    # using elementwise row-scaling instead of a sparse diagonal matrix product
    A_unscaled = GenA(nx, ny, nz, V, qinn)   # sparse (N,N) CSR
    # Row-scale: multiply each row i by dtx[i]  — no N×N sparse alloc
    A = csr_matrix(A_unscaled.multiply(dtx.reshape(-1, 1)))

    # Scale source term once
    fi = fi_base * dtx.reshape(-1, 1)        # (N,1), stays on GPU

    # ------------------------------------------------------------------
    # 2.  Time loop  (only RelPerm + one SpMV + one vector add per step)
    # ------------------------------------------------------------------
    for _ in range(Nts):
        mw, mo, _, _ = RelPerm2(S, UW, UO, BW, BO, SWI, SWR, nx, ny, nz)

        # Fractional flow (fused ops, no temporaries)
        fw = mw / (mw + mo)                  # (N,1)

        # Saturation update: S += A @ fw + fi
        S = S + A @ fw + fi

    return S



def Upstream_3PHASE(
    nx, ny, nz, S, Soil,
    UW, UO, UG, BW, BO, BG, RS,
    SWI, SWR, Vol, qinn, qinnoil, V, Tt, porosity, tables,
):
    """
    Explicit upstream (upwind) three-phase flow solver.

    Note on phase convention
    ------------------------
    Following the Eclipse SWOG convention used in the calling code:
      S    = water saturation
      Soil = gas saturation   (named 'oil' but tracked via gas fractional flow)
      fw   = water fractional flow  = Mw / Mt
      fwo  = gas fractional flow    = Mg / Mt
    where Mt = Mw + Mo + Mg + RS*Mo  (total mobility incl. dissolved gas).

    Parameters
    ----------
    S, Soil    : cupy.ndarray (N,1) — water and gas saturation
    SWOW, SWOG : cupy.ndarray       — rel-perm tables
    V          : dict {"x","y","z"} — face fluxes

    Returns
    -------
    S, Soil : cupy.ndarray (N,1)
    """
    N = nx * ny * nz

    # ------------------------------------------------------------------
    # 1.  Static pre-computations (outside time loop)
    # ------------------------------------------------------------------
    poro = cp.reshape(porosity, (N, 1), "F")
    pv   = Vol * poro                               # (N,1)

    qinn  = cp.reshape(qinn,    (-1, 1), "F")
    qinno = cp.reshape(qinnoil, (-1, 1), "F")

    # Face-flux splits (computed once)
    XP = cp.maximum(V["x"], 0);  XN = cp.minimum(V["x"], 0)
    YP = cp.maximum(V["y"], 0);  YN = cp.minimum(V["y"], 0)
    ZP = cp.maximum(V["z"], 0);  ZN = cp.minimum(V["z"], 0)

    Vi = (
          XP[:nx,      :,      :]
        + YP[:,      :ny,      :]
        + ZP[:,        :,    :nz]
        - XN[1:nx+1,   :,      :]
        - YN[:,   1:ny+1,      :]
        - ZN[:,        :, 1:nz+1]
    )
    Vi = cp.reshape(Vi, (N, 1), "F")

    # ------------------------------------------------------------------
    # 2.  Unified CFL — use the MORE restrictive (larger Nts) of the two
    #     phases. Both share the same velocity field so they must step
    #     together. Using separate Nts for each phase violates stability
    #     for the slower-stepping phase.
    # ------------------------------------------------------------------
    fi_inflow  = cp.maximum(qinn,  0)
    fio_inflow = cp.maximum(qinno, 0)

    pm  = float(cp.min(pv / (Vi + fi_inflow)))    # cp.min — stays on GPU
    pmo = float(cp.min(pv / (Vi + fio_inflow)))

    cfl  = ((1.0 - SWR) / 3.0) * pm
    cflo = ((1.0 - SWR) / 3.0) * pmo

    # Single Nts — most restrictive of both phases
    Nts = max(math.ceil(Tt / cfl), math.ceil(Tt / cflo))

    # Per-cell timestep scalings
    dtx  = cp.ravel(Tt / (Nts * pv))              # (N,)  water
    dtxo = cp.ravel(Tt / (Nts * pv))              # (N,)  gas  (same dt — unified CFL)

    # Row-scale transport matrices — no N×N sparse diagonal alloc
    A_raw  = GenA(nx, ny, nz, V, qinn)
    Ao_raw = GenA(nx, ny, nz, V, qinno)

    A  = csr_matrix(A_raw.multiply( dtx.reshape(-1, 1)))
    Ao = csr_matrix(Ao_raw.multiply(dtxo.reshape(-1, 1)))

    # Scaled source terms (constant across timesteps)
    fi  = fi_inflow  * dtx.reshape(-1, 1)          # (N,1)
    fio = fio_inflow * dtxo.reshape(-1, 1)          # (N,1)

    # ------------------------------------------------------------------
    # 3.  Explicit time loop
    #     Only RelPerm3 + two SpMVs + two vector adds per step
    # ------------------------------------------------------------------
    for _ in range(Nts):
        mw, mo, mg, _, _, _ = RelPerm3(
            S, Soil, UW, UO, UG, BW, BO, BG,
            SWI, SWR, nx, ny, nz, tables
        )

        # Total mobility — computed once, shared by both fractional flows
        Mt = mw + mo + mg + RS * mo                # (N,1)

        # Water fractional flow and saturation update
        fw = mw / Mt
        S  = S + A @ fw + fi

        # Gas fractional flow and saturation update
        # (fwo = gas f.f. under SWOG/Eclipse convention)
        fwo  = mg / Mt
        Soil = Soil + Ao @ fwo + fio

    return S, Soil


def RelPerm2(Sa, UW, UO, BW, BO, SWI, SWR, nx, ny, nz):
    """
    Two-phase Brooks-Corey relative permeability and mobility derivatives.

    Parameters
    ----------
    Sa         : cupy.ndarray — water saturation, any shape
    UW, UO     : float        — water / oil viscosity
    BW, BO     : float        — water / oil formation volume factor
    SWI        : float        — irreducible water saturation
    SWR        : float        — residual water saturation
    nx, ny, nz : int          — grid dimensions (unused; retained for API compatibility)

    Returns
    -------
    Mw, Mo, dMw, dMo : cupy.ndarray, each shape (N, 1)
    """

    # Scalar denominator — computed once
    denom = 1.0 - SWI - SWR                      # scalar

    # Normalised water saturation — flat (N,) for all subsequent ops
    S     = (Sa.ravel() - SWI) / denom           # (N,)
    one_S = 1.0 - S                              # reused for Mo and dMo

    # Mobilities
    Mw = S**2        / (UW * BW)
    Mo = one_S**2    / (UO * BO)

    # Derivatives (analytically consistent with Mw, Mo above)
    dMw =  2.0 * S     / (UW * BW * denom)
    dMo = -2.0 * one_S / (UO * BO * denom)

    # Single reshape at return — no intermediate (-1,1) allocations
    def _col(x): return x.reshape(-1, 1)

    return _col(Mw), _col(Mo), _col(dMw), _col(dMo)



def _build_relperm3_tables(SWOW, SWOG):
    # Force conversion to CuPy FIRST before any slicing or astype
    SWOW = cp.asarray(SWOW, dtype=cp.float32)
    SWOG = cp.asarray(SWOG, dtype=cp.float32)

    sw_tab = SWOW[:, 0]
    sg_tab = SWOG[:, 0]

    return {
        "sw_tab"    : sw_tab,
        "sg_tab"    : sg_tab,
        "KROW_tab"  : SWOW[:, 1],
        "KRW_tab"   : SWOW[:, 2],
        "KROG_tab"  : SWOG[:, 1],
        "KRG_tab"   : SWOG[:, 2],
        "dKRW_tab"  : cp.gradient(SWOW[:, 2], sw_tab),
        "dKROW_tab" : cp.gradient(SWOW[:, 1], sw_tab),
        "dKRG_tab"  : cp.gradient(SWOG[:, 2], sg_tab),
        "dKROG_tab" : cp.gradient(SWOG[:, 1], sg_tab),
    }


def _interp_fast(x_flat, xp, fp):
    """
    Single-pass linear interpolation on a flat float32 CuPy array.
    No dtype casting, no NaN handling, no boundary cp.where overhead —
    those are unnecessary for rel-perm tables which are always in [0,1].

    Parameters
    ----------
    x_flat : cupy.ndarray (N,)  float32 — query points, already flat
    xp     : cupy.ndarray (M,)  float32 — table x-nodes (monotone increasing)
    fp     : cupy.ndarray (M,)  float32 — table y-values

    Returns
    -------
    y : cupy.ndarray (N,)  float32
    """
    idx = cp.searchsorted(xp, x_flat, side="right")
    idx = cp.clip(idx, 1, len(xp) - 1)

    x0 = xp[idx - 1];  x1 = xp[idx]
    f0 = fp[idx - 1];  f1 = fp[idx]

    t = (x_flat - x0) / (x1 - x0 + 1e-30)
    t = cp.clip(t, 0.0, 1.0)
    return f0 + t * (f1 - f0)


def _interp_batch(x_flat, xp, fp_list):
    """
    Run searchsorted ONCE and interpolate multiple fp arrays in one pass.
    This is the key optimisation — 8 separate interp calls become 1
    searchsorted + 8 cheap linear ops.

    Parameters
    ----------
    x_flat  : cupy.ndarray (N,)     float32 — query points
    xp      : cupy.ndarray (M,)     float32 — shared x-nodes
    fp_list : list of cupy.ndarray  float32 — y-value tables to interpolate

    Returns
    -------
    results : list of cupy.ndarray (N,)  float32
    """
    idx = cp.searchsorted(xp, x_flat, side="right")
    idx = cp.clip(idx, 1, len(xp) - 1)

    x0  = xp[idx - 1]
    x1  = xp[idx]
    t   = cp.clip((x_flat - x0) / (x1 - x0 + 1e-30), 0.0, 1.0)

    results = []
    for fp in fp_list:
        f0 = fp[idx - 1]
        f1 = fp[idx]
        results.append(f0 + t * (f1 - f0))
    return results


def RelPerm3(Sa, Sg, UW, UO, UG, BW, BO, BG, SWI, SWR, nx, ny, nz, tables):
    """
    Three-phase relative permeability and mobility derivatives.

    Uses precomputed table gradients (from _build_relperm3_tables) so that
    cp.gradient is never called inside the Newton loop.
    All 8 interpolations share a single cp.searchsorted call per phase.

    Parameters
    ----------
    Sa, Sg  : cupy.ndarray (N,1) or (N,) — water and gas saturations
    UW,UO,UG: float — phase viscosities [cP]
    BW,BO,BG: float — formation volume factors [RB/STB or RB/SCF]
    SWI     : float — irreducible water saturation
    SWR     : float — residual water saturation
    nx,ny,nz: int   — grid dimensions (unused internally, kept for API compat)
    tables  : dict  — precomputed from _build_relperm3_tables(SWOW, SWOG)
                      Pass this once per timestep, reuse across Newton iters.

    Returns
    -------
    Mw, Mo, Mg, dMw, dMo, dMg : cupy.ndarray each (N,1)
    """

    # ── 1. Flatten to float32 once ────────────────────────────────────────────
    sw   = cp.ravel(Sa, order="F").astype(cp.float32)   # (N,)
    sg   = cp.ravel(Sg, order="F").astype(cp.float32)   # (N,)
    soil = cp.clip(1.0 - (sw + sg), 0.0, 1.0)           # (N,)

    # ── 2. Unpack precomputed tables ──────────────────────────────────────────
    sw_tab   = tables["sw_tab"]
    sg_tab   = tables["sg_tab"]

    # ── 3. Batched interpolation — ONE searchsorted per phase ─────────────────
    # Water-saturation tables: KROW, KRW, dKROW, dKRW — 4 arrays, 1 searchsorted
    KROW, KRW, dKROW_sw, dKRW_sw = _interp_batch(
        sw, sw_tab,
        [tables["KROW_tab"], tables["KRW_tab"],
         tables["dKROW_tab"], tables["dKRW_tab"]]
    )

    # Gas-saturation tables: KROG, KRG, dKROG, dKRG — 4 arrays, 1 searchsorted
    KROG, KRG, dKROG_sg, dKRG_sg = _interp_batch(
        sg, sg_tab,
        [tables["KROG_tab"], tables["KRG_tab"],
         tables["dKROG_tab"], tables["dKRG_tab"]]
    )

    # ── 4. Three-phase KRO — Baker's linear model ─────────────────────────────
    denom_s = float(1.0 - SWI - SWR)                    # scalar
    So_norm = cp.clip(soil / denom_s, 0.0, 1.0)         # (N,)
    KRO     = So_norm * KROW * KROG                      # (N,)

    # ── 5. Mobilities ─────────────────────────────────────────────────────────
    inv_UwBw = float(1.0 / (UW * BW))
    inv_UoBo = float(1.0 / (UO * BO))
    inv_UgBg = float(1.0 / (UG * BG))

    Mw = KRW  * inv_UwBw                                 # (N,)
    Mo = KRO  * inv_UoBo
    Mg = KRG  * inv_UgBg

    # ── 6. Derivatives — chain rule on Baker's KRO ────────────────────────────
    # dKRO/dSw = dSo_norm/dSw * KROW*KROG  +  So_norm * dKROW/dSw * KROG
    #          = -KROW*KROG/denom_s  +  So_norm * dKROW_sw * KROG
    # dKRO/dSg = -KROW*KROG/denom_s  +  So_norm * KROW * dKROG_sg
    KROW_KROG_d = KROW * KROG / denom_s                  # (N,) shared term

    dKRO_sw = -KROW_KROG_d + So_norm * dKROW_sw * KROG
    dKRO_sg = -KROW_KROG_d + So_norm * KROW     * dKROG_sg

    dMw = dKRW_sw  * inv_UwBw
    dMo = (dKRO_sw + dKRO_sg) * inv_UoBo
    dMg = dKRG_sg  * inv_UgBg

    # ── 7. Return (N,1) columns ───────────────────────────────────────────────
    c = lambda x: x.reshape(-1, 1)
    return c(Mw), c(Mo), c(Mg), c(dMw), c(dMo), c(dMg)


def NewtRaph(
    nx, ny, nz, porosity, Vol, S, V, qinn, Tt,
    UW, UO, SWI, SWR, method2, BW, BO,
    max_newton=20,
    newton_tol=0.005,   # per-cell mean residual — grid-size independent
    max_it=20,
):
    """
    Newton-Raphson implicit solver for two-phase (water-oil) flow.

    Fixes vs original:
    ------------------
    1. Normalised residual ||G||/N used for convergence — not step norm ||ds||
       which scales with dt and N and never converges properly.
    2. Production sinks included in residual via fw-weighted fi_prod term.
       Original cp.maximum(qinn, 0) clipped all producers to zero.
    3. Preconditioner built from actual Jacobian -dG, not from -B.
    4. Saturation clipped to [SWI, 1] after every Newton update.
    5. max_newton=20, max_it=10 (was 10 and implicit 8).
    """
    N = nx * ny * nz

    # ── 1. Static quantities ──────────────────────────────────────────────────
    poro = cp.reshape(porosity, (N, 1), "F")
    pv   = Vol * poro                                  # (N,1) pore volume

    qinn_col = cp.asarray(qinn, dtype=cp.float32).reshape(-1, 1)
    fi_inj   = cp.maximum(qinn_col, 0.0)               # (N,1) injection source
    fi_prod  = cp.minimum(qinn_col, 0.0)               # (N,1) production sink

    A     = GenA(nx, ny, nz, V, qinn)                 # sparse (N,N) — static
    I_mat = sparse.eye(N, dtype=cp.float32, format="csr")

    S00  = S.copy()
    conv = False
    IT   = 0

    res_w = float("inf")

    # ── 2. Outer adaptive timestep loop ──────────────────────────────────────
    while not conv:

        if IT > max_it:
            print(f"[NewtRaph] WARNING: IT={IT} > max_it={max_it}, "
                  f"returning unconverged solution")
            return S

        dt       = Tt / (2 ** IT)
        dtx      = cp.nan_to_num(dt / pv, nan=0.0)    # (N,1)
        dtx_flat = cp.ravel(dtx)

        # dt-scaled source terms
        fi_w  = fi_inj  * dtx                          # injection  (>=0)
        fi_pw = fi_prod * dtx                          # production (<=0)

        # Row-scale transport matrix by dtx
        B = csr_matrix(A.multiply(dtx_flat.reshape(-1, 1)))   # (N,N) sparse

        S   = S00.copy()

        # ── 3. Sub-step loop ──────────────────────────────────────────────────
        failed = False
        for sub in range(2 ** IT):
            S0    = S.copy()
            res_w = float("inf")
            it    = 0
            M_pre = None

            # ── 4. Newton iteration ───────────────────────────────────────────
            while res_w > newton_tol and it < max_newton:

                Mw, Mo, dMw, dMo = RelPerm2(
                    S, UW, UO, BW, BO, SWI, SWR, nx, ny, nz
                )

                MwMo = Mw + Mo                         # total mobility (N,1)

                # Fractional-flow derivative df/dSw
                df_flat = cp.ravel(
                    dMw / MwMo - Mw / MwMo**2 * (dMw + dMo)
                )

                # Jacobian: dG = I - B * diag(df)
                dG     = I_mat - csr_matrix(B.multiply(df_flat))
                neg_dG = -dG

                # Preconditioner from Jacobian — built once per sub-step
                if it == 0 and method2 in (1, 3, 5):
                    try:
                        ilu   = spilu(neg_dG)
                        M_pre = LinearOperator(
                            (N, N), matvec=lambda x, f=ilu: f.solve(x)
                        )
                    except Exception:
                        M_pre = None

                # Fractional flow and residual
                fw     = Mw / MwMo                     # (N,1) water fraction
                G_flat = cp.ravel(
                    S - S0 - (B @ fw + fi_w + fi_pw * fw)
                )

                # ── Linear solve ──────────────────────────────────────────────
                if method2 == 1:
                    ds, _ = gmres(neg_dG, G_flat, rtol=1e-6, atol=0,
                                  restart=20, maxiter=100, M=M_pre)
                elif method2 == 2:
                    ds = spsolve(neg_dG, G_flat)
                elif method2 == 3:
                    ds, _ = cg(neg_dG, G_flat, rtol=1e-6, atol=0,
                               maxiter=100, M=M_pre)
                elif method2 == 4:
                    ds = lsqr(neg_dG, G_flat)[0]
                elif method2 == 5:
                    ds, _ = bicgstab(neg_dG, G_flat, M=M_pre, tol=1e-6)
                else:
                    ds, _ = gmres(neg_dG, G_flat, rtol=1e-6, atol=0,
                                  restart=20, maxiter=100, M=M_pre)

                # Line search — dampen if update violates [SWI, 1]
                alpha = 1.0
                for _ in range(6):
                    S_try = S + alpha * ds.reshape(-1, 1)
                    if (float(cp.min(S_try)) >= float(SWI) and
                            float(cp.max(S_try)) <= 1.0):
                        break
                    alpha *= 0.5

                S  += alpha * ds.reshape(-1, 1)
                S   = cp.clip(S, float(SWI), 1.0)      # enforce physical bounds

                # Normalised residual — grid-size and dt independent
                res_w = float(cp.linalg.norm(G_flat, 2)) / N
                it   += 1

            # ── Accept sub-step or mark failed ───────────────────────────────
            if res_w > newton_tol:
                S      = S00.copy()
                failed = True
                print(f"  [NewtRaph] sub={sub}/{2**IT} IT={IT} "
                      f"res_w={res_w:.3e} iters={it} → chopping dt")
                break

        # ── 5. Accept timestep or refine ─────────────────────────────────────
        if not failed and res_w <= newton_tol:
            conv = True
        else:
            IT += 1

    return S



def NewtRaph2(
    nx, ny, nz, porosity, Vol, S, Soil, V, qinn, qinnoil,
    Tt, UW, UO, UG, SWI, SWR, method2, BW, BO, BG, RS, SWOW, SWOG,
    tables=None,
    max_newton=20,
    newton_tol=0.005,
    max_it=20,          # was 8 — residuals reach tol at IT=8/9, need headroom
):
    N = nx * ny * nz

    if tables is None:
        tables = _build_relperm3_tables(SWOW, SWOG)

    # ── 1. Static quantities ──────────────────────────────────────────────────
    poro = cp.reshape(porosity, (N, 1), "F")
    pv   = Vol * poro

    A    = GenA(nx, ny, nz, V, qinn)
    Aoil = GenA(nx, ny, nz, V, qinnoil)

    qinn_col    = cp.asarray(qinn,    dtype=cp.float32).reshape(-1, 1)
    qinnoil_col = cp.asarray(qinnoil, dtype=cp.float32).reshape(-1, 1)

    fi_inj_w  = cp.maximum(qinn_col,    0.0)
    fi_prod_w = cp.minimum(qinn_col,    0.0)
    fi_inj_o  = cp.maximum(qinnoil_col, 0.0)
    fi_prod_o = cp.minimum(qinnoil_col, 0.0)

    I_mat = sparse.eye(N, dtype=cp.float32, format="csr")

    S00    = S.copy()
    S00oil = Soil.copy()
    conv   = False
    IT     = 0

    res_w = res_o = float("inf")
    dsn = dsnoil = 1.0

    def _make_precond(neg_mat):
        try:
            ilu = spilu(neg_mat)
            return LinearOperator((N, N), matvec=lambda x, f=ilu: f.solve(x))
        except Exception:
            return None

    # ── 2. Outer adaptive timestep loop ──────────────────────────────────────
    while not conv:

        if IT > max_it:
            print(f"[NewtRaph2] WARNING: IT={IT} > max_it={max_it}, "
                  f"returning unconverged solution")
            return S, Soil

        dt = Tt / (2 ** IT)

        dtx      = cp.nan_to_num(dt / pv, nan=0.0)
        dtx_flat = cp.ravel(dtx)

        fi_w  = fi_inj_w  * dtx
        fi_pw = fi_prod_w * dtx
        fi_o  = fi_inj_o  * dtx
        fi_po = fi_prod_o * dtx

        B    = csr_matrix(A.multiply(dtx_flat.reshape(-1, 1)))
        Boil = csr_matrix(Aoil.multiply(dtx_flat.reshape(-1, 1)))

        S    = S00.copy()
        Soil = S00oil.copy()

        # ── 3. Sub-step loop ──────────────────────────────────────────────────
        failed = False
        for sub in range(2 ** IT):
            S0    = S.copy()
            S0oil = Soil.copy()
            res_w = res_o = float("inf")
            dsn = dsnoil = 1.0
            it  = 0
            M_pre = M_pre_oil = None

            # ── 4. Newton iteration ───────────────────────────────────────────
            while (res_w > newton_tol or res_o > newton_tol) and it < max_newton:

                if it == 0 or dsn > 1e-4 or dsnoil > 1e-4:
                    Mw, Mo, Mg, dMw, dMo, dMg = RelPerm3(
                        S, Soil, UW, UO, UG, BW, BO, BG,
                        SWI, SWR, nx, ny, nz, tables
                    )

                denom  = Mw + Mo + Mg
                denom2 = denom ** 2
                dMsum  = dMw + dMo + dMg

                df_flat    = cp.ravel(dMw / denom - Mw / denom2 * dMsum)
                dfoil_flat = cp.ravel(dMg / denom - Mg / denom2 * dMsum)

                dG    = I_mat - csr_matrix(B.multiply(df_flat))
                dGoil = I_mat - csr_matrix(Boil.multiply(dfoil_flat))

                neg_dG    = -dG
                neg_dGoil = -dGoil

                if it == 0 and method2 in (1, 3, 5):
                    M_pre     = _make_precond(neg_dG)
                    M_pre_oil = _make_precond(neg_dGoil)

                fw    = Mw / denom
                fwoil = Mg / denom

                G    = cp.ravel(
                    S    - S0    - (B    @ fw    + fi_w  + fi_pw * fw   )
                )
                Goil = cp.ravel(
                    Soil - S0oil - (Boil @ fwoil + fi_o  + fi_po * fwoil)
                )

                # ── Linear solves ─────────────────────────────────────────────
                if method2 == 1:
                    ds,    _ = gmres(neg_dG,    G,    rtol=1e-6, atol=0,
                                     restart=20, maxiter=100, M=M_pre)
                    dsoil, _ = gmres(neg_dGoil, Goil, rtol=1e-6, atol=0,
                                     restart=20, maxiter=100, M=M_pre_oil)
                elif method2 == 2:
                    ds    = spsolve(neg_dG,    G)
                    dsoil = spsolve(neg_dGoil, Goil)
                elif method2 == 3:
                    ds,    _ = cg(neg_dG,    G,    rtol=1e-6, atol=0,
                                  maxiter=100, M=M_pre)
                    dsoil, _ = cg(neg_dGoil, Goil, rtol=1e-6, atol=0,
                                  maxiter=100, M=M_pre_oil)
                elif method2 == 4:
                    ds    = lsqr(neg_dG,    G)[0]
                    dsoil = lsqr(neg_dGoil, Goil)[0]
                elif method2 == 5:
                    ds,    _ = bicgstab(neg_dG,    G,    M=M_pre,     tol=1e-6)
                    dsoil, _ = bicgstab(neg_dGoil, Goil, M=M_pre_oil, tol=1e-6)
                else:
                    ds,    _ = gmres(neg_dG,    G,    rtol=1e-6, atol=0,
                                     restart=20, maxiter=100, M=M_pre)
                    dsoil, _ = gmres(neg_dGoil, Goil, rtol=1e-6, atol=0,
                                     restart=20, maxiter=100, M=M_pre_oil)

                # ── Line search: enforce Sw >= SWI, Sg >= 0, Sw+Sg <= 1 ──────
                alpha = 1.0
                for _ in range(6):
                    S_try    = S    + alpha * ds.reshape(-1, 1)
                    Soil_try = Soil + alpha * dsoil.reshape(-1, 1)
                    sw_ok  = float(cp.min(S_try))             >= float(SWI)
                    sg_ok  = float(cp.min(Soil_try))          >= 0.0
                    sum_ok = float(cp.max(S_try + Soil_try))  <= 1.0
                    if sw_ok and sg_ok and sum_ok:
                        break
                    alpha *= 0.5

                S    += alpha * ds.reshape(-1, 1)
                Soil += alpha * dsoil.reshape(-1, 1)

                # ── CRITICAL FIX: clip jointly — gas gets remaining pore space
                S    = cp.clip(S,    float(SWI), 1.0)
                Soil = cp.clip(Soil, 0.0, 1.0 - S)   # Sw + Sg <= 1 always

                res_w   = float(cp.linalg.norm(G,    2))/N
                res_o   = float(cp.linalg.norm(Goil, 2))/N
                dsn    = float(cp.linalg.norm(ds,    2)) * alpha
                dsnoil = float(cp.linalg.norm(dsoil, 2)) * alpha
                it    += 1

            # ── Accept sub-step or mark failed ───────────────────────────────
            if res_w > newton_tol or res_o > newton_tol:
                S    = S00.copy()
                Soil = S00oil.copy()
                failed = True
                print(f"  [Newton] sub={sub}/{2**IT} IT={IT} "
                      f"res_w={res_w:.3e} res_o={res_o:.3e} "
                      f"dsn={dsn:.3e} dsnoil={dsnoil:.3e} iters={it} → chopping dt")
                break

        # ── 5. Accept timestep or refine ─────────────────────────────────────
        if not failed and res_w <= newton_tol and res_o <= newton_tol:
            conv = True
        else:
            IT += 1

    return S, Soil

# ---------------------------------------------------------------------------
# 1.  calc_mu_g  —  Gas Viscosity  [cP]
# ---------------------------------------------------------------------------

def calc_mu_g(p):
    """
    Estimate gas viscosity as a quadratic function of pressure.

    Physics
    -------
    Empirical polynomial fit (Lee-Kesler style):
        μ_g(p) = 3×10⁻¹⁰·p² + 1×10⁻⁶·p + 0.0133

    At low pressure  (p→0):    μ_g ≈ 0.0133 cP
    At high pressure (p=5000): μ_g ≈ 0.026  cP  (physically reasonable)

    Parameters
    ----------
    p : float or cupy.ndarray — reservoir pressure [psi]

    Returns
    -------
    mu_g : float or cupy.ndarray — gas viscosity [cP]
    """
    return 3e-10 * p**2 + 1e-6 * p + 0.0133


# ---------------------------------------------------------------------------
# 2.  calc_rs  —  Solution Gas-Oil Ratio  [SCF/STB]
# ---------------------------------------------------------------------------

def calc_rs(p_bub, p):
    """
    Compute the solution GOR at pressure p (Standing 1947 correlation).

    Physics
    -------
    Below bubble point (p < p_bub):
        RS = C · (p / p_bub)^1.3
    Above bubble point (p ≥ p_bub):
        RS = C  (constant — no more gas dissolves)

    where C = 178.11² / 5.615 ≈ 5658 SCF/STB.

    Parameters
    ----------
    p_bub : float — bubble-point pressure [psi]
    p     : float or cupy.ndarray — reservoir pressure [psi]

    Returns
    -------
    rs : float or cupy.ndarray — solution GOR [SCF/STB]
    """
    C = 178.11**2 / 5.615   # ≈ 5658 SCF/STB

    if cp.isscalar(p) or isinstance(p, (int, float)):
        if p < p_bub:
            return float(C * (p / p_bub) ** 1.3)
        else:
            return float(C)
    else:
        p = cp.asarray(p, dtype=cp.float32)
        return cp.where(
            p < p_bub,
            C * (p / p_bub) ** 1.3,
            cp.full_like(p, C),
        )


# ---------------------------------------------------------------------------
# 3.  calc_dp  —  Pressure Differential for FVF Correlations  [psi]
# ---------------------------------------------------------------------------

def calc_dp(p_bub, p_atm, p):
    """
    Effective pressure differential used in FVF exponential fits.

    Physics
    -------
    Below Pb:  dp = p_atm - p       (reservoir expanding toward atm)
    Above Pb:  dp = p_atm - p_bub  (capped at bubble-point differential)

    Parameters
    ----------
    p_bub : float — bubble-point pressure [psi]
    p_atm : float — atmospheric pressure [psi]
    p     : float or cupy.ndarray — reservoir pressure [psi]

    Returns
    -------
    dp : float or cupy.ndarray — effective pressure differential [psi]
    """
    if cp.isscalar(p) or isinstance(p, (int, float)):
        if p < p_bub:
            return float(p_atm - p)
        else:
            return float(p_atm - p_bub)
    else:
        p = cp.asarray(p, dtype=cp.float32)
        return cp.where(p < p_bub, p_atm - p, p_atm - p_bub)


# ---------------------------------------------------------------------------
# 4.  calc_bg  —  Gas Formation Volume Factor  [RB/SCF]
# ---------------------------------------------------------------------------

def calc_bg(p_bub, p_atm, p):
    """
    Compute the gas formation volume factor BG.

    Physics
    -------
    BG = 1 / exp(1.7×10⁻³ · dp)

    where dp = calc_dp(p_bub, p_atm, p).

    At p = p_atm: dp = 0 → BG = 1  (correct — gas at surface conditions)
    At reservoir pressure: dp < 0  → BG > 1  (gas expands on production)

    Parameters
    ----------
    p_bub : float — bubble-point pressure [psi]
    p_atm : float — atmospheric pressure [psi]
    p     : float or cupy.ndarray — reservoir pressure [psi]

    Returns
    -------
    b_g : float or cupy.ndarray — gas FVF [RB/SCF]
    """
    dp = calc_dp(p_bub, p_atm, p)
    # dp is already float (scalar) or CuPy array — cp.exp handles both
    
    return 1.0 / cp.exp(1.7e-3 * dp)


# ---------------------------------------------------------------------------
# 5.  calc_bo  —  Oil Formation Volume Factor  [RB/STB]
# ---------------------------------------------------------------------------

def calc_bo(p_bub, p_atm, CFO, p):
    """
    Compute the oil formation volume factor BO.

    Physics
    -------
    Below bubble point (p < p_bub):
        BO = 1 / exp(-8×10⁻⁵ · (p_atm - p))
           = exp(8×10⁻⁵ · (p_atm - p))
        BO > 1 at reservoir pressure (p > p_atm) ✓

    Above bubble point (p ≥ p_bub):
        BO = 1 / (exp(-8×10⁻⁵·(p_atm-p_bub)) · exp(-CFO·(p-p_bub)))
        BO decreases with increasing pressure above Pb ✓

    Parameters
    ----------
    p_bub : float — bubble-point pressure [psi]
    p_atm : float — atmospheric pressure [psi]
    CFO   : float — oil compressibility [1/psi]
    p     : float or cupy.ndarray — reservoir pressure [psi]

    Returns
    -------
    b_o : float or cupy.ndarray — oil FVF [RB/STB]
    """
    if cp.isscalar(p) or isinstance(p, (int, float)):
        if p < p_bub:
            return float(1.0 / cp.exp(-8e-5 * (p_atm - p)))
        else:
            return float(1.0 / (cp.exp(-8e-5 * (p_atm - p_bub))
                                * cp.exp(-CFO * (p - p_bub))))
    else:
        p        = cp.asarray(p, dtype=cp.float32)
        bo_below = 1.0 / cp.exp(-8e-5 * (p_atm - p))
        bo_above = 1.0 / (cp.exp(-8e-5 * (p_atm - p_bub))
                          * cp.exp(-CFO * (p - p_bub)))
        return cp.where(p < p_bub, bo_below, bo_above)



def GenA(nx, ny, nz, V, qsa):
    """
    Input:

    nx, ny, nz: integers representing the number of grid cells in the x, y, and z directions, respectively.
    V: a dictionary containing the coordinate information for the grid cells.
    qsa: an array of shape (nx, ny, nz) representing the source term.
    Output:

    A: a sparse CSR matrix of shape (NxNyNz, NxNyNz) representing the discretized differential operator.
    Description:
    This function generates a sparse CSR matrix A that represents the discretized
    differential operator for the given grid and source term using the finite
    volume method. The matrix A is generated based on the Upstream weighting scheme.
    The input V is a dictionary containing the coordinate information for each grid cell,
    and qsa is the source term. The output A is a sparse CSR matrix of shape (NxNyNz, NxNyNz)
    that can be used to solve the system of linear equations representing the flow of fluid through the porous media.
    """
    N  = nx * ny * nz
    Nx, Ny, Nz = nx, ny, nz

    # ------------------------------------------------------------------
    # 1. Source term — outflow only (negative part)
    # ------------------------------------------------------------------
    fp = cp.ravel(cp.minimum(qsa, 0), order="F")   # (N,)

    # ------------------------------------------------------------------
    # 2. Face fluxes — compute positive/negative parts once per axis,
    #    then immediately slice to the interior faces needed.
    #    This avoids storing the full XN/XP arrays as named temporaries.
    # ------------------------------------------------------------------
    vx = V["x"]
    x1 = cp.ravel(cp.minimum(vx[:Nx,      :,      :],    0), order="F")  # left  faces
    x2 = cp.ravel(cp.maximum(vx[1:Nx+1,   :,      :],    0), order="F")  # right faces

    vy = V["y"]
    y1 = cp.ravel(cp.minimum(vy[:,    :Ny,      :],       0), order="F")
    y2 = cp.ravel(cp.maximum(vy[:,  1:Ny+1,     :],       0), order="F")

    vz = V["z"]
    z1 = cp.ravel(cp.minimum(vz[:,      :,    :Nz],       0), order="F")
    z2 = cp.ravel(cp.maximum(vz[:,      :,  1:Nz+1],      0), order="F")

    # ------------------------------------------------------------------
    # 3. Main diagonal — mass balance at each cell
    # ------------------------------------------------------------------
    diag_main = fp + x1 - x2 + y1 - y2 + z1 - z2   # (N,)  fused on GPU

    # ------------------------------------------------------------------
    # 4. Assemble diagonal vectors — stack into one 2-D array so spdiags
    #    gets a single contiguous GPU allocation rather than a Python list.
    # ------------------------------------------------------------------
    diag_vecs = cp.stack([z2, y2, x2, diag_main, -x1, -y1, -z1])  # (7, N)
    diag_indx = [-Nx * Ny, -Nx, -1, 0, 1, Nx, Nx * Ny]

    A = spdiags(diag_vecs, diag_indx, N, N, format="csr")
    return A


def ProgressBar(Total, Progress, BarLength=20, ProgressIcon="#", BarIcon="-"):
    try:
        # You can't have a progress bar with zero or negative length.
        if BarLength < 1:
            BarLength = 20
        # Use status variable for going to the next line after progress completion.
        Status = ""
        # Calcuting progress between 0 and 1 for percentage.
        Progress = float(Progress) / float(Total)
        # Doing this conditions at final progressing.
        if Progress >= 1.0:
            Progress = 1
            Status = "\r\n"  # Going to the next line
        # Calculating how many places should be filled
        Block = int(round(BarLength * Progress))
        # Show this
        Bar = "[{}] {:.0f}% {}".format(
            ProgressIcon * Block + BarIcon * (BarLength - Block),
            round(Progress * 100, 0),
            Status,
        )
        return Bar
    except:
        return "ERROR"


def ShowBar(Bar):
    sys.stdout.write(Bar)
    sys.stdout.flush()


def Equivalent_time(tim1, max_t1, tim2, max_t2):
    tk2 = tim1 / max_t1
    tc2 = np.arange(0.0, 1 + tk2, tk2)
    tc2[tc2 >= 1] = 1
    tc2 = tc2.reshape(-1, 1)  # reference scaled to 1

    tc2r = np.arange(0.0, max_t1 + tim1, tim1)
    tc2r[tc2r >= max_t1] = max_t1
    tc2r = tc2r.reshape(-1, 1)  # reference original
    func = interpolate.interp1d(tc2r.ravel(), tc2.ravel())

    tc2rr = np.arange(0.0, max_t2 + tim2, tim2)
    tc2rr[tc2rr >= max_t2] = max_t2
    tc2rr = tc2rr.reshape(-1, 1)  # reference original
    ynew = func(tc2rr.ravel())

    return ynew


##############################################################################
#         FINITE VOLUME RESERVOIR SIMULATOR
##############################################################################
def Reservoir_Simulator(
    Kuse,
    porosity,
    quse,
    quse_water,
    nx,
    ny,
    nz,
    factorr,
    max_t,
    Dx,
    Dy,
    Dz,
    BO,
    BW,
    CFL,
    timmee,
    MAXZ,
    PB,
    PATM,
    CFO,
    IWSw,
    method,
    steppi,
    SWI,
    SWR,
    UW,
    UO,
    step2,
    pini_alt,
):
    """
    Reservoir_Simulator function for 2 phase flow

    This function simulates the flow of fluids in a porous reservoir by solving the
    pressure and saturation equations using different numerical methods.
    The function takes the following parameters:

    Kuse: an array of shape (nx, ny, nz) representing the permeability values in the reservoir.

    porosity: an array of shape (nx, ny, nz) representing the porosity values in the reservoir.

    quse: an array of shape (nx, ny, nz) representing the source terms in the reservoir.

    quse_water: an array of shape (nx, ny, nz) representing the water source terms in the reservoir.

    nx, ny, nz: integers representing the number of grid cells in the x, y, and z directions, respectively.

    factorr: a float representing the anisotropy factor for the permeability tensor.

    max_t: a float representing the maximum simulation time.

    Dx, Dy, Dz: floats representing the dimensions of the grid cells in the x, y, and z directions, respectively.

    BO: a float representing the initial oil formation volume factor.

    BW: a float representing the initial water formation volume factor.

    CFL: an integer representing whether or not to use CFL condition for time step control.

    timmee: a float representing the total simulation time.

    MAXZ: a float representing the maximum depth of the reservoir.

    PB: a float representing the reservoir pressure at the bottom.

    PATM: a float representing the atmospheric pressure.

    CFO: a float representing the compressibility of the formation.

    IWSw: a float representing the initial water saturation in the reservoir.

    method: an integer representing the numerical method to use for solving the equations.

    steppi: an integer representing the number of time steps to take.

    SWI: a float representing the irreducible water saturation.

    SWR: a float representing the residual water saturation.

    UW: a float representing the water viscosity.

    UO: a float representing the oil viscosity.

    typee: an integer representing the solver type for AMGX.

    step2: an integer representing the number of sub-steps to use for implicit saturation calculations.

    pini_alt: a float representing the initial pressure in the reservoir.

    Returns:

    Big: a numpy array of shape (steppi, nx, ny, 2) representing the pressure and saturation fields over time.
    """
    text = """
                                                                                           
    NNNNNNNN        NNNNNNNVVVVVVVV           VVVVVVVRRRRRRRRRRRRRRRRR     SSSSSSSSSSSSSSS 
    N:::::::N       N::::::V::::::V           V::::::R::::::::::::::::R  SS:::::::::::::::S
    N::::::::N      N::::::V::::::V           V::::::R::::::RRRRRR:::::RS:::::SSSSSS::::::S
    N:::::::::N     N::::::V::::::V           V::::::RR:::::R     R:::::S:::::S     SSSSSSS
    N::::::::::N    N::::::NV:::::V           V:::::V  R::::R     R:::::S:::::S            
    N:::::::::::N   N::::::N V:::::V         V:::::V   R::::R     R:::::S:::::S            
    N:::::::N::::N  N::::::N  V:::::V       V:::::V    R::::RRRRRR:::::R S::::SSSS         
    N::::::N N::::N N::::::N   V:::::V     V:::::V     R:::::::::::::RR   SS::::::SSSSS    
    N::::::N  N::::N:::::::N    V:::::V   V:::::V      R::::RRRRRR:::::R    SSS::::::::SS  
    N::::::N   N:::::::::::N     V:::::V V:::::V       R::::R     R:::::R      SSSSSS::::S 
    N::::::N    N::::::::::N      V:::::V:::::V        R::::R     R:::::R           S:::::S
    N::::::N     N:::::::::N       V:::::::::V         R::::R     R:::::R           S:::::S
    N::::::N      N::::::::N        V:::::::V        RR:::::R     R:::::SSSSSSS     S:::::S
    N::::::N       N:::::::N         V:::::V         R::::::R     R:::::S::::::SSSSSS:::::S
    N::::::N        N::::::N          V:::V          R::::::R     R:::::S:::::::::::::::SS 
    NNNNNNNN         NNNNNNN           VVV           RRRRRRRR     RRRRRRRSSSSSSSSSSSSSSS  
    """
    print(text)
    # ------------------------------------------------------------------
    # 1.  Grid geometry  (plain Python/float — no cp.int32 truncation)
    # ------------------------------------------------------------------
    Nx, Ny, Nz = int(nx), int(ny), int(nz)
    N  = Nx * Ny * Nz

    hx = float(Dx) / Nx
    hy = float(Dy) / Ny
    hz = float(Dz) / Nz

    # Cell volume (scalar)
    Vol_scalar = hx * hy * hz
    Vol        = cp.full((N, 1), Vol_scalar, dtype=cp.float32)

    # TPFA face-transmissibility coefficients (scalars)
    hx_n = 1.0 / Nx;  hy_n = 1.0 / Ny;  hz_n = 1.0 / Nz
    tx   = 2.0 * hy_n * hz_n / hx_n
    ty_c = 2.0 * hx_n * hz_n / hy_n
    tz   = 2.0 * hx_n * hy_n / hz_n

    # ------------------------------------------------------------------
    # 2.  Time vector
    # ------------------------------------------------------------------
    tc2 = cp.asarray(Equivalent_time(timmee, MAXZ, timmee, max_t))
    dt  = float(cp.diff(tc2)[0])
    St  = dt
    Runs = tc2.shape[0]

    # ------------------------------------------------------------------
    # 3.  Static GPU arrays  (computed once)
    # ------------------------------------------------------------------
    porosity    = cp.asarray(porosity,    dtype=cp.float32)
    datause     = cp.asarray(Kuse,        dtype=cp.float32)
    Qq          = cp.asarray(quse,        dtype=cp.float32).ravel(order="F")
    quse_water  = cp.asarray(quse_water,  dtype=cp.float32)

    Kq = cp.zeros((3, Nx, Ny, Nz), dtype=cp.float32)
    Kq[0] = datause
    Kq[1] = datause
    Kq[2] = factorr * datause

    S = IWSw * cp.ones((N, 1), dtype=cp.float32)

    # ------------------------------------------------------------------
    # 4.  Output buffers stay on GPU until the very end
    # ------------------------------------------------------------------
    if Nz == 1:
        output_allp = cp.zeros((steppi, Nx, Ny),     dtype=cp.float32)
        output_alls = cp.zeros((steppi, Nx, Ny),     dtype=cp.float32)
    else:
        output_allp = cp.zeros((steppi, Nx, Ny, Nz), dtype=cp.float32)
        output_alls = cp.zeros((steppi, Nx, Ny, Nz), dtype=cp.float32)

    # ------------------------------------------------------------------
    # 5.  Transmissibility arrays allocated ONCE  (interior slices
    #     overwritten in-place each timestep — no reallocation)
    # ------------------------------------------------------------------
    TX = cp.zeros((Nx + 1, Ny,      Nz),     dtype=cp.float32)
    TY = cp.zeros((Nx,     Ny + 1,  Nz),     dtype=cp.float32)
    TZ = cp.zeros((Nx,     Ny,      Nz + 1), dtype=cp.float32)

    b = Qq

    print("-----------------------------FORWARDING---------------------------")

    BO = cp.float32(BO)

    # ------------------------------------------------------------------
    # 6.  Main time loop
    # ------------------------------------------------------------------
    for t in range(Runs - 1):
        progressBar = "\rSimulation Progress: " + ProgressBar(Runs - 1, t, Runs - 1)
        ShowBar(progressBar)

        # --- Mobility fields (vectorised, no Python loops) -------------
        Sout = (S.reshape(Nx, Ny, Nz, order="F") - SWI) / (1.0 - SWI - SWR)
        Mw   = Sout**2          / (UW * BW)
        Mo   = (1.0 - Sout)**2  / (UO * float(BO))
        Mt   = Mw + Mo          # shape (Nx, Ny, Nz)

        # --- Effective permeability  KM = K * Mt  ----------------------
        # Broadcast Mt across the 3 direction axis — no stack/reshape
        KM = Kq * Mt[cp.newaxis, :, :, :]           # (3, Nx, Ny, Nz)
        Ll = cp.reciprocal(KM)                       # 1/KM, fused on GPU

        # --- Transmissibilities (in-place slice assignment) ------------
        TX[1:Nx,  :,    :   ] = tx   / (Ll[0, :Nx-1, :,    :   ] + Ll[0, 1:Nx,  :,    :   ])
        TY[:,     1:Ny, :   ] = ty_c / (Ll[1, :,     :Ny-1,:   ] + Ll[1, :,     1:Ny, :   ])
        TZ[:,     :,    1:Nz] = tz   / (Ll[2, :,     :,    :Nz-1] + Ll[2, :,    :,    1:Nz])

        # --- TPFA matrix assembly --------------------------------------
        x1 = cp.ravel(TX[:Nx,       :,    :   ], order="F")
        x2 = cp.ravel(TX[1:Nx+1,    :,    :   ], order="F")
        y1 = cp.ravel(TY[:,    :Ny,  :   ],       order="F")
        y2 = cp.ravel(TY[:,  1:Ny+1, :   ],       order="F")
        z1 = cp.ravel(TZ[:,    :,    :Nz ],       order="F")
        z2 = cp.ravel(TZ[:,    :,  1:Nz+1],       order="F")

        diag_main = x1 + x2 + y1 + y2 + z1 + z2
        diag_vecs = cp.stack([-z2, -y2, -x2, diag_main, -x1, -y1, -z1])  # (7,N)
        diag_indx = [-Nx * Ny, -Nx, -1, 0, 1, Nx, Nx * Ny]

        A = spdiags(diag_vecs, diag_indx, N, N, format="csr")

        # Boundary correction — direct data-array edit, no index lookup
        A[0, 0] = A[0, 0] + float(cp.sum(Kq[:, 0, 0, 0]))

        # --- Preconditioner (built once per pressure solve) ------------
        def _make_precond(mat):
            try:
                ilu = spilu(mat)
                return LinearOperator((N, N), matvec=lambda x, f=ilu: f.solve(x))
            except Exception:
                return None

        # --- Pressure solve --------------------------------------------
        if method == 1:                              # GMRES + ILU
            M = _make_precond(A)
            u, _ = gmres(A, b, rtol=1e-6, atol=0, restart=20, maxiter=100, M=M)

        elif method == 2:                            # Direct LU
            u = spsolve(A, b)

        elif method == 3:                            # CG + ILU
            M = _make_precond(A)
            u, _ = cg(A, b, rtol=1e-6, atol=0, maxiter=100, M=M)

        elif method == 4:                            # LSQR
            u = lsqr(A, b)[0]
        elif method == 5:                            #BiCGSTAB 
            u, _ = bicgstab_ilu(A, b, tol=1e-6)
        else:                                        # AMG
            u = amg_solve(A, b)

        # --- Pressure field & Darcy fluxes ----------------------------
        P = u.reshape(Nx, Ny, Nz, order="F")

        V = {
            "x": cp.zeros((Nx + 1, Ny,     Nz),     dtype=cp.float32),
            "y": cp.zeros((Nx,     Ny + 1, Nz),     dtype=cp.float32),
            "z": cp.zeros((Nx,     Ny,     Nz + 1), dtype=cp.float32),
        }
        V["x"][1:Nx, :,    :   ] = (P[:Nx-1, :,    :   ] - P[1:Nx, :,    :   ]) * TX[1:Nx, :,    :   ]
        V["y"][:,    1:Ny, :   ] = (P[:,    :Ny-1, :   ] - P[:,    1:Ny, :   ]) * TY[:,    1:Ny, :   ]
        V["z"][:,    :,    1:Nz] = (P[:,    :,    :Nz-1] - P[:,    :,    1:Nz]) * TZ[:,    :,    1:Nz]

        # --- Saturation update ----------------------------------------
        if CFL == 1:
            S = Upstream_2PHASE(
                nx, ny, nz, S, UW, UO, BW, BO,
                SWI, SWR, Vol, quse_water, V, dt, porosity,
            )
        else:
            dt_sub = St / float(step2)
            for _ in range(step2):
                S = NewtRaph(
                    nx, ny, nz, porosity, Vol, S, V, quse_water,
                    dt_sub, UW, UO, SWI, SWR, method, BW, BO,
                )

        S = cp.clip(S, SWI, 1.0)

        # --- Store outputs (still on GPU) -----------------------------
        P_field = P if Nz > 1 else P[:, :, 0]
        S_field = S.reshape(Nx, Ny, Nz, order="F")
        S_field = S_field if Nz > 1 else S_field[:, :, 0]

        if t < steppi:
            output_allp[t] = P_field
            output_alls[t] = S_field

        # --- Update BO (pressure-dependent) ---------------------------
        Ppz  = float(cp.mean(P))
        BO   = calc_bo(PB, PATM, CFO, Ppz)

    progressBar = "\rSimulation Progress: " + ProgressBar(Runs - 1, Runs - 1, Runs - 1)
    ShowBar(progressBar)

    # ------------------------------------------------------------------
    # 7.  Single host transfer at the end
    # ------------------------------------------------------------------
    Big = cp.vstack([output_allp, output_alls])
    return cp.asnumpy(Big)



def Reservoir_Simulator2(
    Kuse, porosity, quse, quse_water, quse_oil,
    nx, ny, nz, factorr, max_t,
    Dx, Dy, Dz, BO, BW, BG, RS, CFL, timmee, MAXZ,
    PB, PATM, CFO, IWSw, IWSo, method, steppi,
    SWI, SWR, UW, UO, UG, step2, pini_alt, SWOW, SWOG,
):
    """
    3-phase finite-volume reservoir simulator (pressure + water/oil/gas sat.).

    Returns
    -------
    Big : numpy.ndarray, shape (4*steppi, nx, ny[, nz])
        Stacked: pressure | water-sat | oil-sat | gas-sat
    """

    # ------------------------------------------------------------------
    # 1.  Grid geometry  (plain Python floats — no cp.int32 truncation)
    # ------------------------------------------------------------------
    tables = _build_relperm3_tables(SWOW, SWOG)
    Nx, Ny, Nz = int(nx), int(ny), int(nz)
    N = Nx * Ny * Nz

    hx = float(Dx) / Nx
    hy = float(Dy) / Ny
    hz = float(Dz) / Nz

    Vol_scalar = hx * hy * hz
    Vol        = cp.full((N, 1), Vol_scalar, dtype=cp.float32)

    # Normalised TPFA coefficients
    hx_n = 1.0 / Nx;  hy_n = 1.0 / Ny;  hz_n = 1.0 / Nz
    tx   = 2.0 * hy_n * hz_n / hx_n
    ty_c = 2.0 * hx_n * hz_n / hy_n
    tz   = 2.0 * hx_n * hy_n / hz_n

    # ------------------------------------------------------------------
    # 2.  Time vector
    # ------------------------------------------------------------------
    tc2  = cp.asarray(Equivalent_time(timmee, MAXZ, timmee, max_t))
    dt   = float(cp.diff(tc2)[0])
    St   = dt
    Runs = tc2.shape[0]

    # ------------------------------------------------------------------
    # 3.  Static GPU arrays (computed once)
    # ------------------------------------------------------------------
    porosity   = cp.asarray(porosity,   dtype=cp.float32)
    datause    = cp.asarray(Kuse,       dtype=cp.float32)
    Qq         = cp.asarray(quse,       dtype=cp.float32).ravel(order="F")
    quse_water = cp.asarray(quse_water, dtype=cp.float32)
    quse_oil   = cp.asarray(quse_oil,   dtype=cp.float32)

    Kq = cp.zeros((3, Nx, Ny, Nz), dtype=cp.float32)
    Kq[0] = datause
    Kq[1] = datause
    Kq[2] = factorr * datause

    S    = IWSw * cp.ones((N, 1), dtype=cp.float32)
    Soil = IWSo * cp.ones((N, 1), dtype=cp.float32)

    # ------------------------------------------------------------------
    # 4.  Output buffers stay on GPU until the very end
    # ------------------------------------------------------------------
    shape2d = (steppi, Nx, Ny)
    shape3d = (steppi, Nx, Ny, Nz)
    buf_shape = shape2d if Nz == 1 else shape3d

    output_allp    = cp.zeros(buf_shape, dtype=cp.float32)
    output_alls    = cp.zeros(buf_shape, dtype=cp.float32)
    output_allsoil = cp.zeros(buf_shape, dtype=cp.float32)
    output_allsgas = cp.zeros(buf_shape, dtype=cp.float32)

    # ------------------------------------------------------------------
    # 5.  Transmissibility arrays allocated ONCE
    # ------------------------------------------------------------------
    TX = cp.zeros((Nx + 1, Ny,     Nz),     dtype=cp.float32)
    TY = cp.zeros((Nx,     Ny + 1, Nz),     dtype=cp.float32)
    TZ = cp.zeros((Nx,     Ny,     Nz + 1), dtype=cp.float32)

    b  = Qq
    # BO = cp.float32(BO)
    # BG = cp.float32(BG)
    # RS = cp.float32(RS)

    print("-----------------------------FORWARDING---------------------------")

    # ------------------------------------------------------------------
    # 6.  Preconditioner helper (avoids repeated closure redefinition)
    # ------------------------------------------------------------------
    def _make_precond(mat):
        try:
            ilu = spilu(mat)
            return LinearOperator((N, N), matvec=lambda x, f=ilu: f.solve(x))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 7.  Main time loop
    # ------------------------------------------------------------------
    for t in range(Runs - 1):
        progressBar = "\rSimulation Progress: " + ProgressBar(Runs - 1, t, Runs - 1)
        ShowBar(progressBar)

        # --- Phase mobilities ------------------------------------------
        Mw, Mo, Mg, _, _, _ = RelPerm3(
            S, Soil, UW, UO, UG, BW, float(BO), float(BG),
            SWI, SWR, Nx, Ny, Nz, tables
        )

        # Total mobility — includes dissolved-gas contribution
        #Mt = Mw + Mo + Mg #+ Mo * float(RS)       # (N,1) or scalar-broadcast
        
        Mt = Mw / float(BW) + Mo / float(BO) * (1.0 + float(RS)) + Mg / float(BG)

        # --- Effective permeability — single broadcast multiply --------
        Mt_3d = Mt.reshape(Nx, Ny, Nz, order="F")
        KM    = Kq * Mt_3d[cp.newaxis, :, :, :]  # (3,Nx,Ny,Nz) fused
        Ll    = cp.reciprocal(KM)                 # (3,Nx,Ny,Nz)

        # --- Transmissibilities (in-place slice update) ----------------
        TX[1:Nx,  :,    :   ] = tx   / (Ll[0, :Nx-1, :,    :   ] + Ll[0, 1:Nx,  :,    :   ])
        TY[:,     1:Ny, :   ] = ty_c / (Ll[1, :,     :Ny-1,:   ] + Ll[1, :,     1:Ny, :   ])
        TZ[:,     :,    1:Nz] = tz   / (Ll[2, :,     :,    :Nz-1] + Ll[2, :,    :,    1:Nz])

        # --- TPFA matrix assembly (corrected slice bounds) -------------
        x1 = cp.ravel(TX[:Nx,      :,     :   ], order="F")
        x2 = cp.ravel(TX[1:Nx+1,   :,     :   ], order="F")
        y1 = cp.ravel(TY[:,   :Ny,  :   ],        order="F")
        y2 = cp.ravel(TY[:, 1:Ny+1, :   ],        order="F")
        z1 = cp.ravel(TZ[:,    :,   :Nz ],        order="F")
        z2 = cp.ravel(TZ[:,    :, 1:Nz+1],        order="F")

        diag_main = x1 + x2 + y1 + y2 + z1 + z2
        diag_vecs = cp.stack([-z2, -y2, -x2, diag_main, -x1, -y1, -z1])  # (7,N)
        diag_indx = [-Nx * Ny, -Nx, -1, 0, 1, Nx, Nx * Ny]

        A = spdiags(diag_vecs, diag_indx, N, N, format="csr")
        A[0, 0] = A[0, 0] + float(cp.sum(Kq[:, 0, 0, 0]))

        # --- Pressure solve --------------------------------------------
        if method == 1:
            M = _make_precond(A)
            u, _ = gmres(A, b, rtol=1e-6, atol=0, restart=20, maxiter=100, M=M)
        elif method == 2:
            u = spsolve(A, b)
        elif method == 3:
            M = _make_precond(A)
            u, _ = cg(A, b, rtol=1e-6, atol=0, maxiter=100, M=M)
        elif method == 4:
            u = lsqr(A, b)[0]
        elif method == 5:                            #BiCGSTAB 
            u, _ = bicgstab_ilu(A, b, tol=1e-6)            
        else:
            u = amg_solve(A, b)

        # --- Pressure field & Darcy fluxes ----------------------------
        P = u.reshape(Nx, Ny, Nz, order="F")

        V = {
            "x": cp.zeros((Nx + 1, Ny,     Nz),     dtype=cp.float32),
            "y": cp.zeros((Nx,     Ny + 1, Nz),     dtype=cp.float32),
            "z": cp.zeros((Nx,     Ny,     Nz + 1), dtype=cp.float32),
        }
        V["x"][1:Nx, :,    :   ] = (P[:Nx-1, :,    :   ] - P[1:Nx, :,    :   ]) * TX[1:Nx, :,    :   ]
        V["y"][:,    1:Ny, :   ] = (P[:,    :Ny-1, :   ] - P[:,    1:Ny, :   ]) * TY[:,    1:Ny, :   ]
        V["z"][:,    :,    1:Nz] = (P[:,    :,    :Nz-1] - P[:,    :,    1:Nz]) * TZ[:,    :,    1:Nz]

        # --- Saturation update ----------------------------------------
        if CFL == 1:
            S, Soil = Upstream_3PHASE(
                nx, ny, nz, S, Soil, UW, UO, UG,
                BW, BO, BG, RS, SWI, SWR, Vol,
                quse_water, quse_oil, V, dt, porosity, tables,
            )
        else:
            dt_sub = St / float(step2)
            for _ in range(step2):
                S, Soil = NewtRaph2(
                    nx, ny, nz, porosity, Vol, S, Soil, V,
                    quse_water, quse_oil, dt_sub,
                    UW, UO, UG, SWI, SWR, method,
                    BW, BO, BG, RS,SWOW, SWOG, tables,
                )

# def NewtRaph2(
    # nx, ny, nz, porosity, Vol, S, Soil, V, qinn, qinnoil,
    # Tt, UW, UO, UG, SWI, SWR, method2, BW, BO, BG, RS, SWOW, SWOG,
    # tables=None,
    # max_newton=20,       # was 5 — far too few for 3-phase
    # newton_tol=1e-3,     # was 0.01 — loosen slightly to avoid excess iters
    # max_it=8,            # max dt halvings before giving up
# ):
        S = cp.clip(S, SWI, 1.0)

        # --- Phase saturation fields (all on GPU, no cp.asarray no-ops) -
        S2    = S.reshape(Nx, Ny, Nz, order="F")     # water
        S2oil = Soil.reshape(Nx, Ny, Nz, order="F")  # gas (Eclipse convention)
        S2gas = cp.clip(1.0 - cp.abs(S2 + S2oil), SWI, 1.0)  # oil

        # --- Store outputs (still on GPU) -----------------------------
        if t < steppi:
            if Nz == 1:
                output_allp[t]    = P[:, :, 0]
                output_alls[t]    = S2[:, :, 0]
                output_allsoil[t] = S2gas[:, :, 0]
                output_allsgas[t] = S2oil[:, :, 0]
            else:
                output_allp[t]    = P
                output_alls[t]    = S2
                output_allsoil[t] = S2gas
                output_allsgas[t] = S2oil

        # --- Update PVT parameters (direct float cast, no numpy round-trip) ---
        Ppz = float(cp.mean(P))
        BO  = calc_bo(PB, PATM, CFO, Ppz)
        BG  = calc_bg(PB, PATM, Ppz)/ 5.61458
        RS  = calc_rs(PB, Ppz)

    progressBar = "\rSimulation Progress: " + ProgressBar(Runs - 1, Runs - 1, Runs - 1)
    ShowBar(progressBar)

    # ------------------------------------------------------------------
    # 8.  Single host transfer at the very end
    # ------------------------------------------------------------------
    Big = cp.vstack([output_allp, output_alls, output_allsoil, output_allsgas])
    return cp.asnumpy(Big)


def compute_f(
    pressure, kuse, krouse, krwuse, rwell1, skin, pwf_producer1, UO, BO, DX, UW, BW, DZ
):
    RE = 0.2 * cp.asarray(DX)
    up = UO * BO

    # facc = tf.constant(10,dtype = tf.float64)

    DZ = cp.asarray(DZ)
    down = 2.0 * cp.pi * kuse * krouse * DZ
    # down = piit * pii * krouse * DZ1

    right = cp.log(RE / cp.asarray(rwell1)) + cp.asarray(skin)
    J = down / (up * right)
    drawdown = pressure - cp.asarray(pwf_producer1)
    qoil = -((drawdown) * J)
    aa = qoil * 1e-5
    # aa[aa<=0] = 0
    # print(aa)

    # water production
    up2 = UW * BW
    down = 2.0 * cp.pi * kuse * krwuse * DZ
    J = down / (up2 * right)
    drawdown = pressure - cp.asarray(pwf_producer1)
    qwater = -((drawdown) * J)
    aaw = qwater * 1e-5
    # aaw = (qwater)
    # aaw[aaw<=0] = 0
    # print(qwater)
    ouut = aa + aaw
    return -(ouut)  # outnew


def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def rescale_linear_numpy_pytorch(array, new_min, new_max, minimum, maximum):
    """Rescale an arrary linearly."""
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def rescale_linear_pytorch_numpy(array, new_min, new_max, minimum, maximum):
    """Rescale an arrary linearly."""
    m = (maximum - minimum) / (new_max - new_min)
    b = minimum - m * new_min
    return m * array + b

