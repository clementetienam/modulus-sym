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

import numpy as np


import cupy as cp


# ─────────────────────────────────────────────────────────────────────────────
#  masked_clip_2d
# ─────────────────────────────────────────────────────────────────────────────

def masked_clip_2d(data, mask, lower, upper):
    """
    Clip elements of a 2-D array between lower and upper bounds in-place,
    leaving masked elements (mask == True) unchanged.

    Replaces the original masked_clip_2d_kernel (@cuda.jit).

    Parameters
    ----------
    data  : cupy.ndarray  — 2-D float array, modified in-place
    mask  : cupy.ndarray  — boolean array, same shape as data
                            True  → skip (leave unchanged)
                            False → clip
    lower : float         — lower clip bound
    upper : float         — upper clip bound

    Returns
    -------
    None  (data is modified in-place)
    """
    clipped = cp.clip(data, lower, upper)
    # Only write back where mask is False
    cp.copyto(data, clipped, where=~mask)


# ─────────────────────────────────────────────────────────────────────────────
#  interp  —  drop-in replacement for np.interp on CuPy arrays
# ─────────────────────────────────────────────────────────────────────────────

def interp(x, xp, fp, left=None, right=None):
    """
    CuPy implementation of np.interp for 1-D or 2-D input arrays.

    Replaces the original interp2d_kernel (@cuda.jit) with a pure CuPy
    searchsorted-based approach.  No Numba / libnvvm required.

    Behaviour matches np.interp exactly:
      - Values below xp[0]  → left  (default: fp[0])
      - Values above xp[-1] → right (default: fp[-1])
      - NaN inputs          → NaN output

    Parameters
    ----------
    x    : cupy.ndarray  — query points (any shape)
    xp   : cupy.ndarray  — 1-D array of reference x-coordinates (must be
                           monotonically increasing)
    fp   : cupy.ndarray  — 1-D array of reference y-values, same length as xp
    left : float, optional  — value for x < xp[0]   (default: fp[0])
    right: float, optional  — value for x >= xp[-1]  (default: fp[-1])

    Returns
    -------
    output_y : cupy.ndarray — interpolated values, same shape as x, float64
    """
    xp = cp.asarray(xp, dtype=cp.float64)
    fp = cp.asarray(fp, dtype=cp.float64)
    x  = cp.asarray(x,  dtype=cp.float64)

    left_val  = float(fp[0])  if left  is None else float(left)
    right_val = float(fp[-1]) if right is None else float(right)

    shape    = x.shape
    x_flat   = x.ravel()

    # searchsorted gives index i such that xp[i-1] <= x < xp[i]
    idx = cp.searchsorted(xp, x_flat, side="right")
    idx = cp.clip(idx, 1, len(xp) - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    f0 = fp[idx - 1]
    f1 = fp[idx]

    # Linear interpolation — guard against zero-width intervals
    denom = x1 - x0
    denom = cp.where(denom == 0.0, 1.0, denom)        # avoid divide-by-zero
    t     = cp.clip((x_flat - x0) / denom, 0.0, 1.0)
    y     = f0 + t * (f1 - f0)

    # Apply boundary conditions
    y = cp.where(x_flat <  xp[0],  left_val,  y)
    y = cp.where(x_flat >= xp[-1], right_val, y)

    # Propagate NaN
    y = cp.where(cp.isnan(x_flat), cp.nan, y)

    return y.reshape(shape)