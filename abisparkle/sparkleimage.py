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

"""Creates color-coded imagery for interpreting algorithm decisions and input data quality flags (DQF)"""

import cv2
import numpy as np
from heregoes import util


class SDCAImage:
    def __init__(self, sparkle):
        self.sparkle = sparkle

        self._debug_image = None

        self._masked_c02_rf_sparkle = None
        self._c02_rf_sparkle = None
        self._c02_rf_dqf = None

        self._c05_rf_sparkle = None
        self._c05_rf_dqf = None

        self._c07_rf_sparkle = None
        self._c07_rf_dqf = None

        self._c07_bt_sparkle = None
        self._c07_bt_dqf = None

        self._c14_bt_sparkle = None
        self._c14_bt_dqf = None

        self.color_map = {
            "valid_sparkle": (91, 225, 241, 255),
            "unprocessed": (0, 128, 255),
            "land": (108, 140, 117),
            "bad_data": (255, 255, 255),
            "water": (181, 88, 31),
            "cloud": (252, 199, 167),
            "bad_dqf": (0, 145, 255),
            "bad_geometry": (69, 18, 29),
            "border": (69, 16, 32),
            "processed_non_sparkle": (0, 0, 0),
        }

        self.sparkle_image = np.zeros(self.sparkle.source_shape + (4,), dtype=np.uint8)

        for cluster_id, cluster_idx in self.sparkle.SDCAMeta.cluster_map.items():
            max_radius = max(
                self.sparkle.SDCAParams.algo_params["first_window_radius"],
                max(
                    [
                        i["debug"]["window_radius"]
                        for i in self.sparkle.SDCAMeta.get_cluster_members(cluster_id)
                    ]
                ),
            )

            # draw a yellow rectangle that borders the actual window of the algorithm
            rectangle_ul = tuple(
                (
                    int(cluster_idx[1] - 2 - max_radius),
                    int(cluster_idx[0] - 2 - max_radius),
                )
            )
            rectangle_lr = tuple(
                (
                    int(cluster_idx[1] + 2 + max_radius),
                    int(cluster_idx[0] + 2 + max_radius),
                )
            )
            self.debug_image = cv2.rectangle(
                self.debug_image,
                pt1=rectangle_ul,
                pt2=rectangle_lr,
                color=self.color_map["valid_sparkle"][0:3],
                thickness=2,
            )
            self.sparkle_image = cv2.rectangle(
                self.sparkle_image,
                pt1=rectangle_ul,
                pt2=rectangle_lr,
                color=self.color_map["valid_sparkle"],
                thickness=2,
            )

    @property
    def debug_image(self):
        if self._debug_image is None:
            self._debug_image = np.zeros(
                self.sparkle.source_shape + (3,), dtype=np.uint8
            )

            # unprocessed
            self._debug_image[
                np.nonzero(
                    self.sparkle.SDCAFlags.has_flag(
                        self.sparkle.SDCAFlags.algo_flags,
                        self.sparkle.SDCAFlags.algo_flag_def["unprocessed_pixel"],
                    )
                )
            ] = np.array(self.color_map["unprocessed"], dtype=np.uint8)

            # min thresholds, likely land
            self._debug_image[
                np.nonzero(
                    (
                        self.sparkle.SDCAFlags.has_flag(
                            self.sparkle.SDCAFlags.algo_flags,
                            self.sparkle.SDCAFlags.algo_flag_def[
                                "pixel_skipped_by_min_c02_rf_threshold"
                            ],
                        )
                        | self.sparkle.SDCAFlags.has_flag(
                            self.sparkle.SDCAFlags.algo_flags,
                            self.sparkle.SDCAFlags.algo_flag_def[
                                "pixel_skipped_by_min_c05_rf_threshold"
                            ],
                        )
                        | self.sparkle.SDCAFlags.has_flag(
                            self.sparkle.SDCAFlags.algo_flags,
                            self.sparkle.SDCAFlags.algo_flag_def[
                                "pixel_skipped_by_min_c07_rf_threshold"
                            ],
                        )
                        | self.sparkle.SDCAFlags.has_flag(
                            self.sparkle.SDCAFlags.algo_flags,
                            self.sparkle.SDCAFlags.algo_flag_def[
                                "pixel_skipped_by_min_c07_bt_threshold"
                            ],
                        )
                        | self.sparkle.SDCAFlags.has_flag(
                            self.sparkle.SDCAFlags.algo_flags,
                            self.sparkle.SDCAFlags.algo_flag_def[
                                "pixel_skipped_by_min_c14_bt_threshold"
                            ],
                        )
                    )
                )
            ] = np.array(self.color_map["land"], dtype=np.uint8)

            # bad data
            self._debug_image[
                np.nonzero(
                    self.sparkle.SDCAFlags.has_flag(
                        self.sparkle.SDCAFlags.algo_flags,
                        self.sparkle.SDCAFlags.algo_flag_def[
                            "pixel_preinvalidated_by_bad_data"
                        ],
                    )
                )
            ] = np.array(self.color_map["bad_data"], dtype=np.uint8)

            # water
            self._debug_image[
                np.nonzero(
                    self.sparkle.SDCAFlags.has_flag(
                        self.sparkle.SDCAFlags.algo_flags,
                        self.sparkle.SDCAFlags.algo_flag_def[
                            "pixel_preinvalidated_by_water_mask"
                        ],
                    )
                )
            ] = np.array(self.color_map["water"], dtype=np.uint8)

            # cloud
            self._debug_image[
                np.nonzero(
                    self.sparkle.SDCAFlags.has_flag(
                        self.sparkle.SDCAFlags.algo_flags,
                        self.sparkle.SDCAFlags.algo_flag_def[
                            "pixel_skipped_by_cloud_mask"
                        ],
                    )
                )
            ] = np.array(self.color_map["cloud"], dtype=np.uint8)

            # bad dqf
            self._debug_image[
                np.nonzero(
                    self.sparkle.SDCAFlags.has_flag(
                        self.sparkle.SDCAFlags.algo_flags,
                        self.sparkle.SDCAFlags.algo_flag_def[
                            "pixel_preinvalidated_by_bad_dqf"
                        ],
                    )
                )
            ] = np.array(self.color_map["bad_dqf"], dtype=np.uint8)

            # geometry
            self._debug_image[
                np.nonzero(
                    (
                        self.sparkle.SDCAFlags.has_flag(
                            self.sparkle.SDCAFlags.algo_flags,
                            self.sparkle.SDCAFlags.algo_flag_def[
                                "pixel_preinvalidated_by_max_sat_za_threshold"
                            ],
                        )
                        | self.sparkle.SDCAFlags.has_flag(
                            self.sparkle.SDCAFlags.algo_flags,
                            self.sparkle.SDCAFlags.algo_flag_def[
                                "pixel_preinvalidated_by_max_sun_za_threshold"
                            ],
                        )
                        | self.sparkle.SDCAFlags.has_flag(
                            self.sparkle.SDCAFlags.algo_flags,
                            self.sparkle.SDCAFlags.algo_flag_def[
                                "pixel_preinvalidated_by_min_glint_angle_threshold"
                            ],
                        )
                    )
                )
            ] = np.array(self.color_map["bad_geometry"], dtype=np.uint8)

            # border
            self._debug_image[
                np.nonzero(
                    self.sparkle.SDCAFlags.has_flag(
                        self.sparkle.SDCAFlags.algo_flags,
                        self.sparkle.SDCAFlags.algo_flag_def[
                            "pixel_skipped_by_border_mask"
                        ],
                    )
                )
            ] = np.array(self.color_map["border"], dtype=np.uint8)

            # pixels that went through iteration loop at least once and were not validated as sparkles
            self._debug_image[
                np.nonzero(
                    (
                        self.sparkle.SDCAFlags.has_flag(
                            self.sparkle.SDCAFlags.algo_flags,
                            self.sparkle.SDCAFlags.algo_flag_def[
                                "pixel_had_1_window_iterations"
                            ],
                        )
                        & ~self.sparkle.SDCAFlags.has_flag(
                            self.sparkle.SDCAFlags.algo_flags,
                            self.sparkle.SDCAFlags.algo_flag_def[
                                "pixel_validated_by_window_deviation"
                            ],
                        )
                    )
                )
            ] = np.array(self.color_map["processed_non_sparkle"], dtype=np.uint8)

            # valid sparkles
            self._debug_image[np.nonzero(self.sparkle.valid_sparkles)] = np.array(
                self.color_map["valid_sparkle"][0:3], dtype=np.uint8
            )

        return self._debug_image

    @debug_image.setter
    def debug_image(self, value):
        self._debug_image = value

    @property
    def masked_c02_rf_sparkle(self):
        if self._masked_c02_rf_sparkle is None:
            self._masked_c02_rf_sparkle = overlay(
                np.where(
                    self.sparkle.SDCAMask.discard_mask == True,
                    np.uint8(0),
                    self.sparkle.c02_image.bv,
                ),
                self.sparkle_image,
            )

        return self._masked_c02_rf_sparkle

    @property
    def c02_rf_sparkle(self):
        if self._c02_rf_sparkle is None:
            self._c02_rf_sparkle = overlay(
                self.sparkle.c02_image.bv, self.sparkle_image
            )

        return self._c02_rf_sparkle

    @property
    def c02_rf_dqf(self):
        if self._c02_rf_dqf is None:
            self._c02_rf_dqf = overlay(
                self.sparkle.c02_image.bv, dqf_image(self.sparkle.c02_image.dqf)
            )

        return self._c02_rf_dqf

    @property
    def c05_rf_sparkle(self):
        if self._c05_rf_sparkle is None:
            self._c05_rf_sparkle = overlay(
                self.sparkle.c05_image.bv, self.sparkle_image
            )

        return self._c05_rf_sparkle

    @property
    def c05_rf_dqf(self):
        if self._c05_rf_dqf is None:
            self._c05_rf_dqf = overlay(
                self.sparkle.c05_image.bv, dqf_image(self.sparkle.c05_image.dqf)
            )

        return self._c05_rf_dqf

    @property
    def c07_rf_sparkle(self):
        if self._c07_rf_sparkle is None:
            self._c07_rf_sparkle = overlay(
                self.sparkle.c07_nirrefl.bv, self.sparkle_image
            )

        return self._c07_rf_sparkle

    @property
    def c07_rf_dqf(self):
        if self._c07_rf_dqf is None:
            self._c07_rf_dqf = overlay(
                self.sparkle.c07_nirrefl.bv, dqf_image(self.sparkle.c07_image.dqf)
            )

        return self._c07_rf_dqf

    @property
    def c07_bt_sparkle(self):
        if self._c07_bt_sparkle is None:
            self._c07_bt_sparkle = overlay(
                self.sparkle.c07_image.bv, self.sparkle_image
            )

        return self._c07_bt_sparkle

    @property
    def c07_bt_dqf(self):
        if self._c07_bt_dqf is None:
            self._c07_bt_dqf = overlay(
                self.sparkle.c07_image.bv, dqf_image(self.sparkle.c07_image.dqf)
            )

        return self._c07_bt_dqf

    @property
    def c14_bt_sparkle(self):
        if self._c14_bt_sparkle is None:
            self._c14_bt_sparkle = overlay(
                self.sparkle.c14_image.bv, self.sparkle_image
            )

        return self._c14_bt_sparkle

    @property
    def c14_bt_dqf(self):
        if self._c14_bt_dqf is None:
            self._c14_bt_dqf = overlay(
                self.sparkle.c14_image.bv, dqf_image(self.sparkle.c14_image.dqf)
            )

        return self._c14_bt_dqf

    def get_crops(self, img):
        crops = {}
        for cluster_id, cluster_idx in self.sparkle.SDCAMeta.cluster_map.items():
            crops[cluster_id] = util.crop_center(
                getattr(self, img), cluster_idx, (201, 201)
            )

        return crops


def dqf_image(src):
    img = np.stack((src,) * 4, axis=-1).astype(np.uint8)

    img[img[:, :, 3] == 1] = [0, 0, 255, 255]
    img[img[:, :, 3] == 2] = [0, 255, 0, 255]
    img[img[:, :, 3] == 3] = [255, 0, 0, 255]
    img[img[:, :, 3] == 4] = [255, 0, 255, 255]

    return img


def overlay(background, foreground):
    # overlays a 4-channel transparent "foreground" on a 3-channel image "background"
    if background.shape[-1] != 3:
        img = np.stack((background,) * 3, axis=-1)
    else:
        img = background.copy()
    img[foreground[:, :, 3] == 255, :] = foreground[foreground[:, :, 3] == 255, :][
        :, :3
    ]

    return img
