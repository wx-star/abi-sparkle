# Copyright (c) 2021-2023.

# Author(s):

#   Harry Dove-Robinson <admin@wx-star.com>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Entrypoint to the Sparkle object for the Sparkle Detection and Characterization Algorithm (SDCA)"""

import importlib
import time

import cv2
import numpy as np
from heregoes import ancillary, image, load
from heregoes.util import njit

from abisparkle import (
    cloud,
    nirrefl,
    sparklealgo,
    sparkledebug,
    sparkleflags,
    sparkleimage,
    sparklemask,
    sparklemeta,
    sparklenav,
    sparkleparams,
    sparklestats,
)

importlib.reload(sparkleparams)


class Sparkle:
    def __init__(self, c02_nc, c05_nc, c07_nc, c14_nc, water_mask=None, nav=None):
        self.c02_nc = c02_nc
        self.c05_nc = c05_nc
        self.c07_nc = c07_nc
        self.c14_nc = c14_nc
        self.water_mask = water_mask
        self.nav = nav

        # sets C02 as the "source" image - all datasets will be resized to the size of C02
        self.source_abi_data = load(self.c02_nc)
        self.source_shape = (
            self.source_abi_data.dimensions["y"].size,
            self.source_abi_data.dimensions["x"].size,
        )

        #############################################################################
        ##########################setup algo params and flags########################
        s_time = time.time()
        self.SDCAFlags = sparkleflags.SDCAFlags(self.source_shape)
        self.SDCAParams = sparkleparams.SDCAParams()
        self.SDCAStats = sparklestats.SDCAStats()
        print("setup params and flags:", time.time() - s_time)
        #############################################################################
        #############################################################################

        #############################################################################
        ##############################setup water and nav############################
        if self.water_mask is None:
            s_time = time.time()
            self.water_mask = ancillary.WaterMask(
                self.source_abi_data, gshhs_scale="intermediate", rivers=True
            ).data["water_mask"]
            print("setup water:", time.time() - s_time)

        if self.nav is None:
            s_time = time.time()
            self.nav = sparklenav.SparkleNavigation(
                self.source_abi_data, precise_sun=False
            )
            print("setup nav:", time.time() - s_time)
        #############################################################################
        #############################################################################

        #############################################################################
        ###########################test for daylit image#############################
        s_time = time.time()
        if self.check_daylit_land_portion():
            self.is_daylit = True

        else:
            self.is_daylit = False
            print("not operating on a nighttime image")
            print("daylit check:", time.time() - s_time)
            return None
        print("daylit check:", time.time() - s_time)
        #############################################################################
        #############################################################################

        #############################################################################
        ################################setup datasets###############################
        s_time = time.time()
        self.setup_datasets()
        print("setup datasets:", time.time() - s_time)
        #############################################################################
        #############################################################################

        #############################################################################
        #################################setup masking###############################
        s_time = time.time()

        self.SDCAMask = sparklemask.SDCAMask(self)

        print("setup masking:", time.time() - s_time)
        #############################################################################
        #############################################################################

        #############################################################################
        ###############################run algorithm#################################
        s_time = time.time()
        self.SDCAMask.validated_mask = sparklealgo.sparkle(
            c02_rf=self.c02_image.cmi,
            c05_rf=self.c05_image.cmi,
            c07_rf=self.c07_nirrefl.rf,
            c14_bt=self.c14_image.cmi,
            validated_mask=self.SDCAMask.validated_mask,
            discard_mask=self.SDCAMask.discard_mask,
            skip_mask=self.SDCAMask.skip_mask,
            bad_dqf_mask=self.SDCAMask.bad_dqf_mask,
            algo_params=self.SDCAParams.algo_params,
            algo_flags=self.SDCAFlags,
            algo_stats=self.SDCAStats,
        )
        self.valid_sparkles = self.SDCAMask.validated_mask
        print("sparkle algo:", time.time() - s_time)
        #############################################################################
        #############################################################################

        #############################################################################
        ###############set up algorithm meta and post-algo flaging###################
        s_time = time.time()

        self.SDCAMeta = sparklemeta.SDCAMeta(self)
        self.SDCAImage = sparkleimage.SDCAImage(self)
        self.SDCADebug = sparkledebug.SDCADebug(self)

        print("algorithm meta:", time.time() - s_time)

        print("")
        print("Found sparkle pixels:", np.count_nonzero(self.valid_sparkles))
        #############################################################################
        #############################################################################

    def check_daylit_land_portion(self):
        """This is meant to quickly test whether enough of an image is "daylit land" to be worth running the full algorithm on"""

        @njit.heregoes_njit
        def _daylit_land(sun_za, water_mask, subsample_factor, algo_params):
            sun_za_subsampled = sun_za[::subsample_factor, ::subsample_factor]
            water_mask_subsampled = water_mask[::subsample_factor, ::subsample_factor]

            daylit_land_bool = (
                sun_za_subsampled <= np.deg2rad(algo_params["max_sun_za_threshold"])
            ) & (water_mask_subsampled)
            daylit_portion_of_land = np.count_nonzero(
                daylit_land_bool
            ) / np.count_nonzero(water_mask_subsampled)

            if daylit_portion_of_land > algo_params["min_daylit_portion_of_land"]:
                return True

            else:
                return False

        # if the sun navigation and water mask arrays are the same size as the source CMI image, subsample to make the calculation faster
        subsample_factor = 1
        if (
            self.nav.sun_za.shape == self.source_shape
            and self.water_mask.shape == self.source_shape
        ):
            subsample_factor = 10

        return _daylit_land(
            self.nav.sun_za,
            self.water_mask,
            subsample_factor,
            self.SDCAParams.algo_params,
        )

    def norm_shape(self, arr):
        """normalizes arrays to be the same size as self.source_shape"""
        y_factor = self.source_shape[0] / arr.shape[0]
        x_factor = self.source_shape[1] / arr.shape[1]

        if y_factor != x_factor:
            raise Exception(
                "Aspect ratio mismatch when attempting to normalize array shape"
            )

        original_dtype = arr.dtype
        if y_factor != 1 and x_factor != 1:
            if original_dtype == bool:
                arr = arr.astype(np.uint8)

            arr = cv2.resize(
                arr, None, fx=x_factor, fy=y_factor, interpolation=cv2.INTER_NEAREST
            )

        return arr.astype(original_dtype)

    def setup_datasets(self):
        self.c02_image = image.ABIImage(self.c02_nc)
        self.c02_image.rad = self.norm_shape(self.c02_image.rad)
        self.c02_image.dqf = self.norm_shape(self.c02_image.dqf)

        self.c05_image = image.ABIImage(self.c05_nc)
        self.c05_image.rad = self.norm_shape(self.c05_image.rad)
        self.c05_image.dqf = self.norm_shape(self.c05_image.dqf)

        self.c07_image = image.ABIImage(self.c07_nc)
        self.c07_image.rad = self.norm_shape(self.c07_image.rad)
        self.c07_image.dqf = self.norm_shape(self.c07_image.dqf)

        self.c14_image = image.ABIImage(self.c14_nc)
        self.c14_image.rad = self.norm_shape(self.c14_image.rad)
        self.c14_image.dqf = self.norm_shape(self.c14_image.dqf)

        self.c07_nirrefl = nirrefl.ABINIRRefl(self.c07_image, self.c14_image)

        self.nav.glint_angle = self.norm_shape(self.nav.glint_angle)
        self.nav.sun_za = self.norm_shape(self.nav.sun_za)
        self.nav.sat_za = self.norm_shape(self.nav.sat_za)

        self.water_mask = self.norm_shape(self.water_mask)

        self.cloud_mask = self.norm_shape(
            cloud.CloudMask(self.c07_image, self.c14_image).cloud_mask
        )
