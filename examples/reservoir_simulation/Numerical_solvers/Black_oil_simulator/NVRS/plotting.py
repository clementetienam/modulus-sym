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
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os.path

# import torch
from scipy import interpolate
import multiprocessing
import numpy.matlib
import matplotlib.colors
from matplotlib import cm
from shutil import rmtree
import numpy

# os.environ['KERAS_BACKEND'] = 'tensorflow'
import os.path
import time
import random
import numpy.ma as ma
import logging
import os
cores = multiprocessing.cpu_count()
import math
logger = logging.getLogger(__name__)
# numpy.random.seed(99)



def Add_marker(plt, XX, YY, locc):
    """
    Function to add marker to given coordinates on a matplotlib plot

    less
    Copy code
    Parameters:
        plt: a matplotlib.pyplot object to add the markers to
        XX: a numpy array of X coordinates
        YY: a numpy array of Y coordinates
        locc: a numpy array of locations where markers need to be added

    Return:
        None
    """
    # iterate through each location
    for i in range(locc.shape[0]):
        a = locc[i, :]
        xloc = int(a[0])
        yloc = int(a[1])

        # if the location type is 2, add an upward pointing marker
        if a[2] == 2:
            plt.scatter(
                XX.T[xloc - 1, yloc - 1] + 0.5,
                YY.T[xloc - 1, yloc - 1] + 0.5,
                s=100,
                marker="^",
                color="white",
            )
        # otherwise, add a downward pointing marker
        else:
            plt.scatter(
                XX.T[xloc - 1, yloc - 1] + 0.5,
                YY.T[xloc - 1, yloc - 1] + 0.5,
                s=100,
                marker="v",
                color="white",
            )


def Add_marker2(plt, XX, YY, injectors, producers):
    """
    Function to add marker to given coordinates on a matplotlib plot

    less
    Copy code
    Parameters:
        plt: a matplotlib.pyplot object to add the markers to
        XX: a numpy array of X coordinates
        YY: a numpy array of Y coordinates
        locc: a numpy array of locations where markers need to be added

    Return:
        None
    """

    n_inj = len(injectors)  # Number of injectors
    n_prod = len(producers)  # Number of producers

    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])

        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=200,
            marker="v",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )

    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=200,
            marker="^",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )


def Plot_RSM_percentile(True_mat, Namesz):
    timezz = True_mat[:, 0].reshape(-1, 1)

    plt.figure(figsize=(40, 40))

    plt.subplot(4, 4, 1)
    plt.plot(timezz, True_mat[:, 1], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 2)
    plt.plot(timezz, True_mat[:, 2], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 3)
    plt.plot(timezz, True_mat[:, 3], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 4)
    plt.plot(timezz, True_mat[:, 4], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 5)
    plt.plot(timezz, True_mat[:, 5], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 6)
    plt.plot(timezz, True_mat[:, 6], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 7)
    plt.plot(timezz, True_mat[:, 7], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 8)
    plt.plot(timezz, True_mat[:, 8], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 9)
    plt.plot(timezz, True_mat[:, 9], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 10)
    plt.plot(timezz, True_mat[:, 10], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 11)
    plt.plot(timezz, True_mat[:, 11], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 12)
    plt.plot(timezz, True_mat[:, 12], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 13)
    plt.plot(timezz, True_mat[:, 13], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT(\%)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 14)
    plt.plot(timezz, True_mat[:, 14], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT(\%)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 15)
    plt.plot(timezz, True_mat[:, 15], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT(\%)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 16)
    plt.plot(timezz, True_mat[:, 16], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT(\%)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()
    plt.suptitle("FIELD PRODUCTION PROFILE", fontsize=25)

    # os.chdir('RESULTS')
    plt.savefig(
        Namesz
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def Plot_RSM_percentile2(True_mat, Namesz):
    timezz = True_mat[:, 0].reshape(-1, 1)

    plt.figure(figsize=(40, 40))

    plt.subplot(5, 4, 1)
    plt.plot(timezz, True_mat[:, 1], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 2)
    plt.plot(timezz, True_mat[:, 2], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 3)
    plt.plot(timezz, True_mat[:, 3], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 4)
    plt.plot(timezz, True_mat[:, 4], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 5)
    plt.plot(timezz, True_mat[:, 5], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 6)
    plt.plot(timezz, True_mat[:, 6], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 7)
    plt.plot(timezz, True_mat[:, 7], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 8)
    plt.plot(timezz, True_mat[:, 8], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 9)
    plt.plot(timezz, True_mat[:, 9], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 10)
    plt.plot(timezz, True_mat[:, 10], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 11)
    plt.plot(timezz, True_mat[:, 11], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 12)
    plt.plot(timezz, True_mat[:, 12], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 13)
    plt.plot(timezz, True_mat[:, 13], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 14)
    plt.plot(timezz, True_mat[:, 14], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 15)
    plt.plot(timezz, True_mat[:, 15], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 16)
    plt.plot(timezz, True_mat[:, 16], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 17)
    plt.plot(timezz, True_mat[:, 17], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT(\%)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 18)
    plt.plot(timezz, True_mat[:, 18], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT(\%)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 19)
    plt.plot(timezz, True_mat[:, 19], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT(\%)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 20)
    plt.plot(timezz, True_mat[:, 20], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT(\%)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.suptitle("FIELD PRODUCTION PROFILE", fontsize=25)
    # os.chdir('RESULTS')
    plt.savefig(
        Namesz
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def plot_properties(perm, poro, nx, ny, nz, wells):
    if nz == 1:
        permeability = np.reshape(perm, (nx, ny), "F")
        porosity = np.reshape(poro, (nx, ny), "F")

        permeability = cp.asnumpy(permeability)
        porosity = cp.asnumpy(porosity)

        XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

        plt.figure(figsize=(12, 12))
        plt.subplot(2, 2, 1)
        plt.pcolormesh(XX.T, YY.T, permeability, cmap="jet")

        plt.title("permeability ", fontsize=15)
        plt.ylabel("Y", fontsize=13)
        plt.xlabel("X", fontsize=13)
        plt.axis([0, (nx - 1), 0, (ny - 1)])
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        cbar1 = plt.colorbar()
        cbar1.ax.set_ylabel(" K (mD)", fontsize=13)
        plt.clim(min(cp.ravel(permeability)), max(cp.ravel(permeability)))
        Add_marker(plt, XX, YY, wells)

        plt.subplot(2, 2, 2)
        plt.pcolormesh(XX.T, YY.T, porosity, cmap="jet")
        # Add_marker(plt,XX,YY,wells)
        plt.title("porosity ", fontsize=15)
        plt.ylabel("Y", fontsize=13)
        plt.xlabel("X", fontsize=13)
        plt.axis([0, (nx - 1), 0, (ny - 1)])
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        cbar1 = plt.colorbar()
        cbar1.ax.set_ylabel(" K (mD)", fontsize=13)
        plt.clim(min(cp.ravel(porosity)), max(cp.ravel(porosity)))
        Add_marker(plt, XX, YY, wells)
        plt.savefig(os.path.join(path_save, "properties.png"))
        plt.clf()
        plt.close()
    else:
        permeability = np.reshape(perm, (nx, ny, nz), "F")

        porosity = np.reshape(poro, (nx, ny, nz), "F")

        permeability = cp.asnumpy(permeability)
        porosity = cp.asnumpy(porosity)

        XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

        plt.figure(figsize=(12, 12))

        for i in range(nz):
            plt.subplot(2, 3, i + 1)
            plt.pcolormesh(XX.T, YY.T, permeability[:, :, i], cmap="jet")
            title = "Perm_Layer_" + str(i + 1)
            plt.title(title, fontsize=15)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (nx - 1), 0, (ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" K (mD)", fontsize=13)
            plt.clim(min(cp.ravel(permeability)), max(cp.ravel(permeability)))
            Add_marker(plt, XX, YY, wells)
        plt.savefig(os.path.join(path_save, "properties_perm.png"))
        plt.clf()
        plt.close()

        plt.figure(figsize=(12, 12))
        for i in range(nz):
            plt.subplot(2, 3, i + 1)
            plt.pcolormesh(XX.T, YY.T, porosity[:, :, i], cmap="jet")
            title = "Perm_Layer_" + str(i + 1)
            plt.title(title, fontsize=15)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (nx - 1), 0, (ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" units", fontsize=13)
            plt.clim(min(cp.ravel(porosity)), max(cp.ravel(porosity)))
            Add_marker(plt, XX, YY, wells)
        plt.savefig(os.path.join(path_save, "properties_porosity.png"))
        plt.clf()
        plt.close()


def Plot_performance(trueF, nx, ny, namet, itt, dt, MAXZ, steppi, wells):
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, itt - 1, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)

    lookf = trueF[itt, :, :]
    lookf_sat = trueF[itt + steppi, :, :]
    lookf_oil = 1 - lookf_sat

    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.pcolormesh(XX.T, YY.T, lookf, cmap="jet")
    plt.title("Pressure CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(2, 2, 2)
    plt.pcolormesh(XX.T, YY.T, lookf_sat, cmap="jet")
    plt.title("water_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" water sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(2, 2, 3)
    plt.pcolormesh(XX.T, YY.T, lookf_oil, cmap="jet")
    plt.title("oil_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" oil sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(int((itt + 1) * dt * MAXZ)) + " days"

    plt.suptitle(tita, fontsize=16)

    # name = namet + str(int(itt)) + '.png'

    name = namet + "{:03d}.png".format(int(itt))

    plt.savefig(name)

    # plt.show()
    plt.clf()


def Plot_impedance(trueF1, nx, ny, namet, itt, dt, MAXZ, steppi, injectors, producers):
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, itt - 1, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)

    Ip = trueF1

    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.figure(figsize=(12, 12))

    plt.pcolormesh(XX.T, YY.T, Ip, cmap="jet")
    plt.title(r"$I_{p}$", fontsize=16, weight="bold")
    plt.ylabel("Y", fontsize=16)
    plt.xlabel("X", fontsize=16)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(r"$I_{p}$", fontsize=16)
    # Add_marker2(plt,XX,YY,wells)

    Add_marker2(plt, XX, YY, injectors, producers)

    # plt.tight_layout(rect = [0,0,1,0.95])

    tita = "Seismic survey timestep --" + str(int((itt + 1) * dt * MAXZ)) + " days"

    plt.suptitle(tita, fontsize=16)

    # name = namet + str(int(itt)) + '.png'

    name = namet + "{:03d}.png".format(int(itt))
    plt.savefig(name)
    # plt.show()
    plt.clf()


def Plot_performance2(trueF, nx, ny, namet, itt, dt, MAXZ, steppi, wells):
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, itt - 1, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)

    lookf = trueF[itt, :, :]
    lookf_sat = trueF[itt + steppi, :, :]
    lookf_oil = trueF[itt + 2 * steppi, :, :]
    lookf_gas = 1 - (lookf_sat + lookf_oil)

    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.pcolormesh(XX.T, YY.T, lookf, cmap="jet")
    plt.title("Pressure CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(2, 2, 2)
    plt.pcolormesh(XX.T, YY.T, lookf_sat, cmap="jet")
    plt.title("water_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" water sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(2, 2, 3)
    plt.pcolormesh(XX.T, YY.T, lookf_oil, cmap="jet")
    plt.title("oil_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" oil sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(2, 2, 4)
    plt.pcolormesh(XX.T, YY.T, lookf_gas, cmap="jet")
    plt.title("gas_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" gas sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(int((itt + 1) * dt * MAXZ)) + " days"

    plt.suptitle(tita, fontsize=16)

    name = namet + str(int(itt)) + ".png"
    plt.savefig(name)
    # plt.show()
    plt.clf()



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



def plot3d2static(arr_3d, nx, ny, nz, namet, titti, maxii, minii, injectors, producers):
    """
    Plot a 3D array with matplotlib and annotate specific points on the plot.

    Args:
    arr_3d (np.ndarray): 3D array to plot.
    nx (int): number of cells in the x direction.
    ny (int): number of cells in the y direction.
    nz (int): number of cells in the z direction.
    itt (int): current iteration number.
    dt (float): time step.
    MAXZ (int): maximum number of iterations in the z direction.
    namet (str): name of the file to save the plot.
    titti (str): title of the plot.
    maxii (float): maximum value of the colorbar.
    minii (float): minimum value of the colorbar.

    Returns:
    None.
    """
    fig = plt.figure(figsize=(15, 15), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    # Shift the coordinates to center the points at the voxel locations
    x, y, z = np.indices((arr_3d.shape))
    x = x + 0.5
    y = y + 0.5
    z = z + 0.5

    # Set the colors of each voxel using a jet colormap
    colors = plt.cm.jet(arr_3d)
    norm = matplotlib.colors.Normalize(vmin=minii, vmax=maxii)

    # Plot each voxel and save the mappable object
    ax.voxels(arr_3d, facecolors=colors, alpha=0.5, edgecolor="none", shade=True)
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])

    plt.colorbar(m, ax=ax, fraction=0.02, pad=0.1, label=" Log 10 - K(mD)")

    # Add a colorbar for the mappable object
    # plt.colorbar(mappable)
    # Set the axis labels and title
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    # ax.set_title(titti,fontsize= 14)

    # Set axis limits to reflect the extent of each axis of the matrix
    ax.set_xlim(0, arr_3d.shape[0])
    ax.set_ylim(0, arr_3d.shape[1])
    ax.set_zlim(0, arr_3d.shape[2])
    # ax.set_zlim(0, 60)

    # Remove the grid
    ax.grid(False)

    # Set lighting to bright
    ax.set_facecolor("white")
    # Set the aspect ratio of the plot

    ax.set_box_aspect([nx, ny, nz * 2])

    # Set the projection type to orthogonal
    ax.set_proj_type("ortho")

    # Remove the tick labels on each axis
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Remove the tick lines on each axis
    ax.xaxis._axinfo["tick"]["inward_factor"] = 0
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.4

    # Set the azimuth and elevation to make the plot brighter
    ax.view_init(elev=30, azim=45)

    n_inj = len(injectors)  # Number of injectors
    n_prod = len(producers)  # Number of producers

    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 2)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "black", linewidth=2)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="black",
            weight="bold",
            fontsize=16,
        )

    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 2)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "r", linewidth=2)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="r",
            weight="bold",
            fontsize=16,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.suptitle(titti, fontsize=16)

    name = namet + ".png"
    plt.savefig(name)
    # plt.show()
    plt.clf()


def plot3d2(
    arr_3d, nx, ny, nz, itt, dt, MAXZ, namet, titti, maxii, minii, injectors, producers
):
    """
    Plot a 3D array with matplotlib and annotate specific points on the plot.

    Args:
    arr_3d (np.ndarray): 3D array to plot.
    nx (int): number of cells in the x direction.
    ny (int): number of cells in the y direction.
    nz (int): number of cells in the z direction.
    itt (int): current iteration number.
    dt (float): time step.
    MAXZ (int): maximum number of iterations in the z direction.
    namet (str): name of the file to save the plot.
    titti (str): title of the plot.
    maxii (float): maximum value of the colorbar.
    minii (float): minimum value of the colorbar.

    Returns:
    None.
    """
    fig = plt.figure(figsize=(12, 12), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    # Shift the coordinates to center the points at the voxel locations
    x, y, z = np.indices((arr_3d.shape))
    x = x + 0.5
    y = y + 0.5
    z = z + 0.5

    # Set the colors of each voxel using a jet colormap
    colors = plt.cm.jet(arr_3d)
    norm = matplotlib.colors.Normalize(vmin=minii, vmax=maxii)

    # Plot each voxel and save the mappable object
    ax.voxels(arr_3d, facecolors=colors, alpha=0.5, edgecolor="none", shade=True)
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])

    if titti == "Pressure":
        plt.colorbar(m,ax=ax, fraction=0.02, pad=0.1, label="Pressure [psia]")
    elif titti == "water_sat":
        plt.colorbar(m,ax=ax, fraction=0.02, pad=0.1, label="water_sat [units]")
    else:
        plt.colorbar(m,ax=ax, fraction=0.02, pad=0.1, label="oil_sat [psia]")

    # Add a colorbar for the mappable object
    # plt.colorbar(mappable)
    # Set the axis labels and title
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    # Set axis limits to reflect the extent of each axis of the matrix
    ax.set_xlim(0, arr_3d.shape[0])
    ax.set_ylim(0, arr_3d.shape[1])
    ax.set_zlim(0, arr_3d.shape[2])
    # ax.set_zlim(0, 10)

    # Remove the grid
    ax.grid(False)

    # Set lighting to bright
    ax.set_facecolor("white")
    # Set the aspect ratio of the plot

    ax.set_box_aspect([nx, ny, nz * 2])

    # Set the projection type to orthogonal
    ax.set_proj_type("ortho")

    # Remove the tick labels on each axis
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Remove the tick lines on each axis
    ax.xaxis._axinfo["tick"]["inward_factor"] = 0
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.4

    # Set the azimuth and elevation to make the plot brighter
    ax.view_init(elev=30, azim=45)

    n_inj = len(injectors)  # Number of injectors
    n_prod = len(producers)  # Number of producers

    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 2)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "black", linewidth=2)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="black",
            weight="bold",
            fontsize=16,
        )

    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 2)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "r", linewidth=2)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="r",
            weight="bold",
            fontsize=16,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = str(titti) + "- Timestep --" + str(int((itt + 1) * dt * MAXZ)) + " days"

    plt.suptitle(tita, fontsize=16)

    # name = namet + str(int(itt)) + '.png'
    name = namet + "{:03d}.png".format(int(itt))
    plt.savefig(name)
    # plt.show()
    plt.close(fig)
