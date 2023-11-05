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

"""Calculates 3.9 μm reflectance factor on ABI"""

import numpy as np
from heregoes.goesr import abi
from heregoes.util import make_8bit, njit
from pyspectral.rsr_reader import RelativeSpectralResponse
from pyspectral.solar import TOTAL_IRRADIANCE_SPECTRUM_2000ASTM, SolarIrradianceSpectrum


class ABINIRRefl:
    def __init__(self, c07_image, c14_image):
        self.c07_image = c07_image
        self.c14_image = c14_image
        self._rf = None
        self._bv = None

        # calculate equivalent 3.9 μm radiance from 11.2 μm brightness temperature
        self.c07_emissive_rad = abi.bt2rad(
            self.c14_image.cmi,
            self.c07_image.abi_data["planck_fk1"][...].item(),
            self.c07_image.abi_data["planck_fk2"][...].item(),
            self.c07_image.abi_data["planck_bc1"][...].item(),
            self.c07_image.abi_data["planck_bc2"][...].item(),
        )

        self.abi_rsr = RelativeSpectralResponse(
            self.c07_image.abi_data.platform_ID_safe, "abi"
        )
        self.c07_solar_irradiance = np.atleast_1d(
            (
                SolarIrradianceSpectrum(
                    TOTAL_IRRADIANCE_SPECTRUM_2000ASTM
                ).inband_solarirradiance(self.abi_rsr.rsr["ch7"])
            )
        )

        # convert from W/m^2/μm to mW/m^2/cm^-1
        self.c07_solar_irradiance = abi.rad_wvl2wvn(
            self.c07_solar_irradiance,
            *self.c07_image.abi_data.instrument_coefficients.eqw,
        )

        self.c07_solar_radiance = self.c07_solar_irradiance / np.pi

    @staticmethod
    @njit.heregoes_njit
    def _calc_rf(c07_rad, c07_emissive_rad, c07_esd, c07_solar_radiance):
        # following https://doi.org/10.1016/0169-8095(94)90096-5
        return (
            ((c07_rad - c07_emissive_rad) * np.square(c07_esd))
            / (c07_solar_radiance - c07_emissive_rad)
        ).astype(np.float32)

    @property
    def rf(self):
        if self._rf is None:
            self._rf = self._calc_rf(
                self.c07_image.rad,
                self.c07_emissive_rad,
                self.c07_image.abi_data["earth_sun_distance_anomaly_in_AU"][...].item(),
                self.c07_solar_radiance.item(),
            )

        return self._rf

    @rf.setter
    def rf(self, value):
        self._rf = value

    @property
    def bv(self):
        if self._bv is None:
            self._bv = make_8bit(self.rf * 255)

        return self._bv

    @bv.setter
    def bv(self, value):
        self._bv = value
