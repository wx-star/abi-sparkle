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

"""Runs the standard ABI navigation routines as well as calculations for specular reflection vectors"""

import numpy as np
from heregoes import navigation
from heregoes.util import njit


class SparkleNavigation(navigation.ABINavigation):
    """Calculate the full navigation of an ABI image, including specular reflection vectors"""

    def __init__(self, *args, **kwargs):
        super(SparkleNavigation, self).__init__(*args, **kwargs)

        self._omega = None
        self._beta = None
        self._gamma = None
        self._glint_angle = None

    @property
    def omega(self):
        if self._omega is None:
            self._omega, self._beta, self._gamma = self.calc_reflections(
                self.sun_az, self.sun_za, self.sat_az, self.sat_za
            )

        return self._omega

    @omega.setter
    def omega(self, value):
        self._omega = value

    @property
    def beta(self):
        if self._beta is None:
            self._omega, self._beta, self._gamma = self.calc_reflections(
                self.sun_az, self.sun_za, self.sat_az, self.sat_za
            )

        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    @property
    def gamma(self):
        if self._gamma is None:
            self._omega, self._beta, self._gamma = self.calc_reflections(
                self.sun_az, self.sun_za, self.sat_az, self.sat_za
            )

        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    @property
    def glint_angle(self):
        if self._glint_angle is None:
            self._glint_angle = self.calc_glint_angle(
                self.sun_az, self.sun_za, self.sat_az, self.sat_za
            )

        return self._glint_angle

    @glint_angle.setter
    def glint_angle(self, value):
        self._glint_angle = value

    @staticmethod
    @njit.heregoes_njit
    def calc_glint_angle(sun_az_rad, sun_za_rad, sat_az_rad, sat_za_rad):
        # angle between Sun and satellite vectors
        return np.atleast_1d(
            np.arccos(
                np.cos(sun_za_rad) * np.cos(sat_za_rad)
                - np.sin(sun_za_rad)
                * np.sin(sat_za_rad)
                * np.cos(sun_az_rad - sat_az_rad)
            )
        ).astype(np.float32)

    @staticmethod
    @njit.heregoes_njit
    def calc_reflections(sun_az_rad, sun_za_rad, sat_az_rad, sat_za_rad):
        # unit vector of sun
        s_x = np.sin(sun_za_rad) * np.cos(sun_az_rad)
        s_y = np.sin(sun_za_rad) * np.sin(sun_az_rad)
        s_z = np.cos(sun_za_rad)

        # unit vector of satellite
        r_x = np.sin(sat_za_rad) * np.cos(sat_az_rad)
        r_y = np.sin(sat_za_rad) * np.sin(sat_az_rad)
        r_z = np.cos(sat_za_rad)

        omega = np.arccos((s_x * r_x) + (s_y * r_y) + (s_z * r_z)) / 2.0
        beta = np.arccos((s_z + r_z) / (2.0 * np.cos(omega)))
        gamma = (np.arctan2(s_y + r_y, s_x + r_x) + (2.0 * np.pi)) % (2.0 * np.pi)

        return (
            np.atleast_1d(omega).astype(np.float32),
            np.atleast_1d(beta).astype(np.float32),
            np.atleast_1d(gamma).astype(np.float32),
        )


class FastSparkleNavigation(SparkleNavigation):
    """Calculate just the Sun and specular reflection vectors of an ABI image from provided lat/lon and satellite vector, with optional subsampling"""

    def __init__(
        self,
        abi_data,
        lat_deg,
        lon_deg,
        sat_za,
        sat_az,
        precise_sun=False,
        subsample_factor=1,
    ):
        self.abi_data = abi_data
        self.lat_deg = lat_deg[::subsample_factor, ::subsample_factor]
        self.lon_deg = lon_deg[::subsample_factor, ::subsample_factor]
        self.sat_za = sat_za[::subsample_factor, ::subsample_factor]
        self.sat_az = sat_az[::subsample_factor, ::subsample_factor]
        self.hae_m = np.atleast_1d(0.0)
        self.precise_sun = precise_sun
        self.subsample_factor = subsample_factor
        self.degrees = False

        self._omega = None
        self._beta = None
        self._gamma = None
        self._glint_angle = None

        self._sun_za = None
        self._sun_az = None
        self._area_m = None

        self.time = self.abi_data.midpoint_time

        self.x_rad, self.y_rad = np.meshgrid(
            self.abi_data["x"][...],
            self.abi_data["y"][...],
        )
        self.x_rad = self.x_rad[::subsample_factor, ::subsample_factor]
        self.y_rad = self.y_rad[::subsample_factor, ::subsample_factor]

        if self.hae_m.shape != self.lat_deg.shape:
            self.hae_m = np.full(self.lat_deg.shape, self.hae_m, dtype=np.float32)
