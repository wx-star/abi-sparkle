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

"""Stores algorithm decisions per pixel index in an int64 bitfield"""

import numpy as np
from numba.core import types as ntypes
from numba.experimental import jitclass
from numba.typed import Dict as ndict

kv_ty = (ntypes.unicode_type, ntypes.int64)
spec = [("algo_flag_def", ntypes.DictType(*kv_ty)), ("algo_flags", ntypes.int64[:, :])]


@jitclass(spec)
class SDCAFlags:
    def __init__(self, source_shape):
        #############################################################################
        ###########################set up algorithm flag definitions#################
        self.algo_flag_def = ndict.empty(*kv_ty)

        self.algo_flag_def["unprocessed_pixel"] = ntypes.int64(0)

        self.algo_flag_def["flag_offset_pre_validated_mask"] = ntypes.int64(1)
        self.algo_flag_def["pixel_validated_by_pre_algo_masking"] = ntypes.int64(2)
        self.algo_flag_def["pixel_prevalidated_by_max_rf_thresholds"] = ntypes.int64(3)

        self.algo_flag_def["flag_offset_skip_mask"] = ntypes.int64(10)
        self.algo_flag_def["pixel_skipped_by_pre_algo_masking"] = ntypes.int64(11)
        self.algo_flag_def["pixel_skipped_by_cloud_mask"] = ntypes.int64(12)
        self.algo_flag_def["pixel_skipped_by_border_mask"] = ntypes.int64(13)
        self.algo_flag_def["pixel_skipped_by_min_c02_rf_threshold"] = ntypes.int64(14)
        self.algo_flag_def["pixel_skipped_by_min_c05_rf_threshold"] = ntypes.int64(15)
        self.algo_flag_def["pixel_skipped_by_min_c07_rf_threshold"] = ntypes.int64(16)
        self.algo_flag_def["pixel_skipped_by_min_c07_bt_threshold"] = ntypes.int64(17)
        self.algo_flag_def["pixel_skipped_by_min_c14_bt_threshold"] = ntypes.int64(18)

        self.algo_flag_def["flag_offset_pre_invalidated_mask"] = ntypes.int64(20)
        self.algo_flag_def["pixel_invalidated_by_pre_algo_masking"] = ntypes.int64(21)
        self.algo_flag_def["pixel_preinvalidated_by_bad_dqf"] = ntypes.int64(22)
        self.algo_flag_def["pixel_preinvalidated_by_bad_data"] = ntypes.int64(23)
        self.algo_flag_def["pixel_preinvalidated_by_water_mask"] = ntypes.int64(24)
        self.algo_flag_def[
            "pixel_preinvalidated_by_max_sat_za_threshold"
        ] = ntypes.int64(25)
        self.algo_flag_def[
            "pixel_preinvalidated_by_max_sun_za_threshold"
        ] = ntypes.int64(26)
        self.algo_flag_def[
            "pixel_preinvalidated_by_min_sun_za_threshold"
        ] = ntypes.int64(27)
        self.algo_flag_def[
            "pixel_preinvalidated_by_min_glint_angle_threshold"
        ] = ntypes.int64(28)

        self.algo_flag_def["flag_offset_algo_passes"] = ntypes.int64(30)
        self.algo_flag_def["pixel_considered_on_first_pass"] = ntypes.int64(31)
        self.algo_flag_def["pixel_considered_on_second_pass"] = ntypes.int64(32)

        self.algo_flag_def["flag_offset_window_iterations"] = ntypes.int64(40)
        self.algo_flag_def["pixel_had_1_window_iterations"] = ntypes.int64(41)
        self.algo_flag_def["pixel_had_2_window_iterations"] = ntypes.int64(42)
        self.algo_flag_def["pixel_had_3_window_iterations"] = ntypes.int64(43)

        self.algo_flag_def["flag_offset_algo_failure_states"] = ntypes.int64(50)
        self.algo_flag_def["pixel_invalidated_by_dqf_neighbor"] = ntypes.int64(51)
        self.algo_flag_def["pixel_invalidated_by_window_sizing"] = ntypes.int64(52)

        self.algo_flag_def["flag_offset_algo_success_states"] = ntypes.int64(60)
        self.algo_flag_def["pixel_validated_by_window_deviation"] = ntypes.int64(61)
        #############################################################################
        #############################################################################

        self.algo_flags = np.zeros(source_shape, dtype=np.int64)

    def set_flag(self, idx, flag):
        self.algo_flags[idx] |= np.int64(1) << np.int64(flag)

    def set_mask_flag(self, arr, flag):
        # sets True pixels in arr with the provided flag value
        for idx in np.argwhere(arr):
            idx = tuple((idx[0], idx[1]))
            self.set_flag(idx, flag)

    def has_flag(self, bitfield, flag):
        return bitfield | np.int64(1) << np.int64(flag) == bitfield

    def bitfield_decode(self, bitfield):
        flag_dict = {}
        for key in self.algo_flag_def.keys():
            if self.has_flag(bitfield, self.algo_flag_def[key]):
                flag_dict[self.algo_flag_def[key]] = key

        return flag_dict

    def idx_decode(self, idx):
        bitfield = self.algo_flags[idx]
        return self.bitfield_decode(bitfield)
