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

"""Sparkle Detection and Characterization Algorithm (SDCA) parameters"""

from numba.core import types as ntypes
from numba.experimental import jitclass
from numba.typed import Dict as ndict

kv_ty = (ntypes.unicode_type, ntypes.float32)
spec = [
    ("algo_params", ntypes.DictType(*kv_ty)),
]


@jitclass(spec)
class SDCAParams:
    def __init__(self):
        #############################################################################
        ###########################set up algorithm parameters#######################
        self.algo_params = ndict.empty(*kv_ty)

        self.algo_params["min_daylit_portion_of_land"] = ntypes.float32(0.1)
        self.algo_params["max_algo_passes"] = ntypes.float32(2.0)

        self.algo_params["first_window_radius"] = ntypes.float32(15.0)
        self.algo_params["max_window_radius_iter"] = ntypes.float32(3.0)
        self.algo_params["min_window_clean_proportion_threshold"] = ntypes.float32(0.75)

        self.algo_params["exclude_border_width"] = ntypes.float32(15.0)
        self.algo_params["exclude_dqf_radius"] = ntypes.float32(10.0)

        self.algo_params["max_sat_za_threshold"] = ntypes.float32(80.0)
        self.algo_params["max_sun_za_threshold"] = ntypes.float32(85.0)
        self.algo_params["min_sun_za_threshold"] = ntypes.float32(10.0)
        self.algo_params["min_glint_angle_threshold"] = ntypes.float32(10.0)

        self.algo_params["c02_rf_max_threshold"] = ntypes.float32(1.0)
        self.algo_params["c05_rf_max_threshold"] = ntypes.float32(1.0)
        self.algo_params["c07_rf_max_threshold"] = ntypes.float32(1.0)

        self.algo_params["c02_rf_min_threshold"] = ntypes.float32(0.475)
        self.algo_params["c05_rf_min_threshold"] = ntypes.float32(0.55)
        self.algo_params["c07_rf_min_threshold"] = ntypes.float32(0.1)
        self.algo_params["c07_bt_min_threshold"] = ntypes.float32(300)
        self.algo_params["c14_bt_min_threshold"] = ntypes.float32(275)

        self.algo_params["c02_rf_deviation_min_threshold"] = ntypes.float32(0.425)
        self.algo_params["c05_rf_deviation_min_threshold"] = ntypes.float32(0.50)
        self.algo_params["c07_rf_deviation_min_threshold"] = ntypes.float32(0.05)
        self.algo_params["c14_bt_deviation_min_threshold"] = ntypes.float32(-3.0)
        self.algo_params["c14_bt_standard_deviation_max_threshold"] = ntypes.float32(
            8.0
        )
        #############################################################################
        #############################################################################
