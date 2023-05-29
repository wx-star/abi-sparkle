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

"""Cloud masking for ABI"""

from heregoes import heregoes_njit


class CloudMask:
    def __init__(self, c07_image, c14_image):

        self.cloud_mask = self.wfabba_clouds(c07_image.cmi, c14_image.cmi)

    @staticmethod
    @heregoes_njit
    def wfabba_clouds(c07_bt, c14_bt):
        # https://www.star.nesdis.noaa.gov/goesr/docs/ATBD/Fire.pdf
        # pg 21-22
        wfabba_mask1 = c14_bt < 270.0  # also in menzel chapter 6
        wfabba_mask2 = (c07_bt - c14_bt) < -4.0
        wfabba_mask3 = ((c07_bt - c14_bt) > 20.0) & (c07_bt < 285.0)

        return wfabba_mask1 | wfabba_mask2 | wfabba_mask3
