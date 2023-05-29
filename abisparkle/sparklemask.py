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

"""Creates boolean masks used to filter the SDCA before the windowed deviation detection stage"""

import numpy as np
from heregoes import heregoes_njit, util


class SDCAMask:
    # the Sparkle class cannot currently be a @jitclass, so there are lots of hidden static methods in here
    def __init__(self, sparkle):
        self.sparkle = sparkle

        self._bad_dqf_mask = None
        self._validated_mask = None
        self._invalidated_mask = None
        self._skip_mask = None

        (
            self.validated_mask,
            self.invalidated_mask,
            self.discard_mask,
            self.skip_mask,
        ) = self._finalize(
            self.validated_mask,
            self.invalidated_mask,
            self.skip_mask,
            self.sparkle.SDCAFlags,
        )

    @staticmethod
    @heregoes_njit
    def _finalize(validated_mask, invalidated_mask, skip_mask, algo_flags):
        # make sure pre-validated pixels don't contain any pre-invalidated ones
        validated_mask.ravel()[np.nonzero(invalidated_mask.ravel())] = False

        # make a convenience mask for discarding validated and invalidated pixels from the statistical background of the moving window function
        discard_mask = validated_mask | invalidated_mask

        # make skip_mask contain validated_mask and invalidated_mask pixels
        skip_mask.ravel()[np.nonzero(discard_mask.ravel())] = True

        # set flags for pre-algo masking
        algo_flags.set_mask_flag(
            validated_mask,
            algo_flags.algo_flag_def["pixel_validated_by_pre_algo_masking"],
        )
        algo_flags.set_mask_flag(
            invalidated_mask,
            algo_flags.algo_flag_def["pixel_invalidated_by_pre_algo_masking"],
        )
        algo_flags.set_mask_flag(
            skip_mask, algo_flags.algo_flag_def["pixel_skipped_by_pre_algo_masking"]
        )

        return validated_mask, invalidated_mask, discard_mask, skip_mask

    @property
    def bad_dqf_mask(self):
        @heregoes_njit
        def _bad_dqf(c02_dqf, c05_dqf, c07_dqf, c14_dqf):
            bad_dqf_mask = (
                ((c02_dqf != 0) & (c02_dqf != 2))
                | ((c05_dqf != 0) & (c05_dqf != 2))
                | ((c07_dqf != 0) & (c07_dqf != 2))
                | ((c14_dqf != 0) & (c14_dqf != 2))
            )

            return bad_dqf_mask

        if self._bad_dqf_mask is None:
            self._bad_dqf_mask = _bad_dqf(
                c02_dqf=self.sparkle.c02_image.dqf,
                c05_dqf=self.sparkle.c05_image.dqf,
                c07_dqf=self.sparkle.c07_image.dqf,
                c14_dqf=self.sparkle.c14_image.dqf,
            )

        return self._bad_dqf_mask

    @property
    def validated_mask(self):
        @heregoes_njit
        def _validate(source_shape, c02_rf, c05_rf, c07_rf, algo_params, algo_flags):
            validated_mask = np.full(source_shape, False)

            max_rf_mask = (
                (c02_rf > algo_params["c02_rf_max_threshold"])
                & (c05_rf > algo_params["c05_rf_max_threshold"])
                & (c07_rf > algo_params["c07_rf_max_threshold"])
            )
            max_rf_idx = np.nonzero(max_rf_mask.ravel())
            validated_mask.ravel()[max_rf_idx] = True
            algo_flags.set_mask_flag(
                max_rf_mask,
                algo_flags.algo_flag_def["pixel_prevalidated_by_max_rf_thresholds"],
            )

            return validated_mask

        if self._validated_mask is None:
            self._validated_mask = _validate(
                source_shape=self.sparkle.source_shape,
                c02_rf=self.sparkle.c02_image.cmi,
                c05_rf=self.sparkle.c05_image.cmi,
                c07_rf=self.sparkle.c07_nirrefl.rf,
                algo_params=self.sparkle.SDCAParams.algo_params,
                algo_flags=self.sparkle.SDCAFlags,
            )

        return self._validated_mask

    @validated_mask.setter
    def validated_mask(self, value):
        self._validated_mask = value

    @property
    def invalidated_mask(self):
        @heregoes_njit
        def _invalidate(
            source_shape,
            c02_rf,
            c05_rf,
            c07_rf,
            c07_bt,
            c14_bt,
            bad_dqf_mask,
            water_mask,
            sat_za,
            sun_za,
            glint_angle,
            algo_params,
            algo_flags,
        ):
            invalidated_mask = np.full(source_shape, False)

            # bad dqfs
            bad_dqf_idx = np.nonzero(bad_dqf_mask.ravel())
            invalidated_mask.ravel()[bad_dqf_idx] = True
            algo_flags.set_mask_flag(
                bad_dqf_mask,
                algo_flags.algo_flag_def["pixel_preinvalidated_by_bad_dqf"],
            )

            # missing/bad data
            bad_data_mask = (
                ((c02_rf <= 0) | (c02_rf == np.nan))
                | ((c05_rf <= 0) | (c05_rf == np.nan))
                | ((c07_rf <= 0) | (c07_rf == np.nan))
                | ((c07_bt <= 0) | (c07_bt == np.nan))
                | ((c14_bt <= 0) | (c14_bt == np.nan))
            )
            bad_data_idx = np.nonzero(bad_data_mask.ravel())
            invalidated_mask.ravel()[bad_data_idx] = True
            algo_flags.set_mask_flag(
                bad_data_mask,
                algo_flags.algo_flag_def["pixel_preinvalidated_by_bad_data"],
            )

            # WaterMask has water as False and land as True, but we want to find water pixels as True so we invert it with ~
            water_idx = np.nonzero(~water_mask.ravel())
            invalidated_mask.ravel()[water_idx] = True
            algo_flags.set_mask_flag(
                ~water_mask,
                algo_flags.algo_flag_def["pixel_preinvalidated_by_water_mask"],
            )

            # exclude by satellite zenith angle
            sat_za_mask = sat_za > np.deg2rad(algo_params["max_sat_za_threshold"])
            sat_za_idx = np.nonzero(sat_za_mask.ravel())
            invalidated_mask.ravel()[sat_za_idx] = True
            algo_flags.set_mask_flag(
                sat_za_mask,
                algo_flags.algo_flag_def[
                    "pixel_preinvalidated_by_max_sat_za_threshold"
                ],
            )

            # exclude by max sun zenith angle - day/night terminator
            sun_za_max_mask = sun_za > np.deg2rad(algo_params["max_sun_za_threshold"])
            sun_za_max_idx = np.nonzero(sun_za_max_mask.ravel())
            invalidated_mask.ravel()[sun_za_max_idx] = True
            algo_flags.set_mask_flag(
                sun_za_max_mask,
                algo_flags.algo_flag_def[
                    "pixel_preinvalidated_by_max_sun_za_threshold"
                ],
            )

            # exclude by min sun zenith angle - subsolar point as in FDCA
            sun_za_min_mask = sun_za <= np.deg2rad(algo_params["min_sun_za_threshold"])
            sun_za_min_idx = np.nonzero(sun_za_min_mask.ravel())
            invalidated_mask.ravel()[sun_za_min_idx] = True
            algo_flags.set_mask_flag(
                sun_za_min_mask,
                algo_flags.algo_flag_def[
                    "pixel_preinvalidated_by_min_sun_za_threshold"
                ],
            )

            # exclude by glint angle as in FDCA
            glint_angle_mask = glint_angle <= np.deg2rad(
                algo_params["min_glint_angle_threshold"]
            )
            glint_angle_idx = np.nonzero(glint_angle_mask.ravel())
            invalidated_mask.ravel()[glint_angle_idx] = True
            algo_flags.set_mask_flag(
                glint_angle_mask,
                algo_flags.algo_flag_def[
                    "pixel_preinvalidated_by_min_glint_angle_threshold"
                ],
            )

            return invalidated_mask

        if self._invalidated_mask is None:
            self._invalidated_mask = _invalidate(
                source_shape=self.sparkle.source_shape,
                c02_rf=self.sparkle.c02_image.cmi,
                c05_rf=self.sparkle.c05_image.cmi,
                c07_rf=self.sparkle.c07_nirrefl.rf,
                c07_bt=self.sparkle.c07_image.cmi,
                c14_bt=self.sparkle.c14_image.cmi,
                bad_dqf_mask=self.bad_dqf_mask,
                water_mask=self.sparkle.water_mask,
                sat_za=self.sparkle.nav.sat_za,
                sun_za=self.sparkle.nav.sun_za,
                glint_angle=self.sparkle.nav.glint_angle,
                algo_params=self.sparkle.SDCAParams.algo_params,
                algo_flags=self.sparkle.SDCAFlags,
            )

        return self._invalidated_mask

    @invalidated_mask.setter
    def invalidated_mask(self, value):
        self._invalidated_mask = value

    @property
    def skip_mask(self):
        @heregoes_njit
        def _skip(
            source_shape,
            cloud_mask,
            c02_rf,
            c05_rf,
            c07_rf,
            c07_bt,
            c14_bt,
            algo_params,
            algo_flags,
        ):
            skip_mask = np.full(source_shape, False)

            cloud_idx = np.nonzero(cloud_mask.ravel())
            skip_mask.ravel()[cloud_idx] = True
            algo_flags.set_mask_flag(
                cloud_mask, algo_flags.algo_flag_def["pixel_skipped_by_cloud_mask"]
            )

            # exclude border
            border_mask = np.full(source_shape, False)
            util.fill_border(
                border_mask,
                width=algo_params["exclude_border_width"],
                fill=True,
                copy=False,
            )
            border_idx = np.nonzero(border_mask.ravel())
            skip_mask.ravel()[border_idx] = True
            algo_flags.set_mask_flag(
                border_mask, algo_flags.algo_flag_def["pixel_skipped_by_border_mask"]
            )

            # exclude by min c02 rf threshold
            c02_rf_min_mask = c02_rf <= algo_params["c02_rf_min_threshold"]
            c02_rf_min_idx = np.nonzero(c02_rf_min_mask.ravel())
            skip_mask.ravel()[c02_rf_min_idx] = True
            algo_flags.set_mask_flag(
                c02_rf_min_mask,
                algo_flags.algo_flag_def["pixel_skipped_by_min_c02_rf_threshold"],
            )

            # exclude by min c05 rf threshold
            c05_rf_min_mask = c05_rf <= algo_params["c05_rf_min_threshold"]
            c05_rf_min_idx = np.nonzero(c05_rf_min_mask.ravel())
            skip_mask.ravel()[c05_rf_min_idx] = True
            algo_flags.set_mask_flag(
                c05_rf_min_mask,
                algo_flags.algo_flag_def["pixel_skipped_by_min_c05_rf_threshold"],
            )

            # exclude by min c07 rf threshold
            c07_rf_min_mask = c07_rf <= algo_params["c07_rf_min_threshold"]
            c07_rf_min_idx = np.nonzero(c07_rf_min_mask.ravel())
            skip_mask.ravel()[c07_rf_min_idx] = True
            algo_flags.set_mask_flag(
                c07_rf_min_mask,
                algo_flags.algo_flag_def["pixel_skipped_by_min_c07_rf_threshold"],
            )

            # exclude by min c07 bt threshold
            c07_bt_min_mask = c07_bt <= algo_params["c07_bt_min_threshold"]
            c07_bt_min_idx = np.nonzero(c07_bt_min_mask.ravel())
            skip_mask.ravel()[c07_bt_min_idx] = True
            algo_flags.set_mask_flag(
                c07_bt_min_mask,
                algo_flags.algo_flag_def["pixel_skipped_by_min_c07_bt_threshold"],
            )

            # exclude by min c14 bt threshold
            c14_bt_min_mask = c14_bt <= algo_params["c14_bt_min_threshold"]
            c14_bt_min_idx = np.nonzero(c14_bt_min_mask.ravel())
            skip_mask.ravel()[c14_bt_min_idx] = True
            algo_flags.set_mask_flag(
                c14_bt_min_mask,
                algo_flags.algo_flag_def["pixel_skipped_by_min_c14_bt_threshold"],
            )

            return skip_mask

        if self._skip_mask is None:
            self._skip_mask = _skip(
                source_shape=self.sparkle.source_shape,
                cloud_mask=self.sparkle.cloud_mask,
                c02_rf=self.sparkle.c02_image.cmi,
                c05_rf=self.sparkle.c05_image.cmi,
                c07_rf=self.sparkle.c07_nirrefl.rf,
                c07_bt=self.sparkle.c07_image.cmi,
                c14_bt=self.sparkle.c14_image.cmi,
                algo_params=self.sparkle.SDCAParams.algo_params,
                algo_flags=self.sparkle.SDCAFlags,
            )

        return self._skip_mask

    @skip_mask.setter
    def skip_mask(self, value):
        self._skip_mask = value
