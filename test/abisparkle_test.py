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

from pathlib import Path

import cv2
import numpy as np
from abisparkle import sdca

SCRIPT_PATH = Path(__file__).parent.resolve()
input_dir = SCRIPT_PATH.joinpath("input")
input_dir.mkdir(parents=True, exist_ok=True)
output_dir = SCRIPT_PATH.joinpath("output")
output_dir.mkdir(parents=True, exist_ok=True)

for output_file in output_dir.glob("*"):
    output_file.unlink()


c02_nc = input_dir.joinpath(
    "OR_ABI-L1b-RadM1-M6C02_G17_s20191631836275_e20191631836333_c20191631836362.nc"
)
c05_nc = input_dir.joinpath(
    "OR_ABI-L1b-RadM1-M6C05_G17_s20191631836275_e20191631836333_c20191631836368.nc"
)
c07_nc = input_dir.joinpath(
    "OR_ABI-L1b-RadM1-M6C07_G17_s20191631836275_e20191631836344_c20191631836375.nc"
)
c14_nc = input_dir.joinpath(
    "OR_ABI-L1b-RadM1-M6C14_G17_s20191631836275_e20191631836333_c20191631836377.nc"
)


sparkle = sdca.Sparkle(c02_nc, c05_nc, c07_nc, c14_nc)
cluster_centroid_idx_1 = (1568, 1138)
cluster_centroid_idx_2 = (1406, 1087)
border_and_water_and_cloud_idx = (747, 1999)
cloud_idx = (882, 925)
water_idx = (852, 70)
num_sparkle_pixels = 95
num_sparkle_clusters = 2


def test_sparkle_image():
    cv2.imwrite(
        str(output_dir.joinpath("c02_rf_sparkle.png")), sparkle.SDCAImage.c02_rf_sparkle
    )
    cv2.imwrite(
        str(output_dir.joinpath("c05_rf_sparkle.png")), sparkle.SDCAImage.c05_rf_sparkle
    )
    cv2.imwrite(
        str(output_dir.joinpath("c07_bt_sparkle.png")), sparkle.SDCAImage.c07_bt_sparkle
    )
    cv2.imwrite(
        str(output_dir.joinpath("c07_rf_sparkle.png")), sparkle.SDCAImage.c07_rf_sparkle
    )
    cv2.imwrite(
        str(output_dir.joinpath("c14_bt_sparkle.png")), sparkle.SDCAImage.c14_bt_sparkle
    )


def test_sparkle_meta():
    # test that records sum to the expected number of sparkle pixels
    assert len(sparkle.SDCAMeta.algo_meta) == num_sparkle_pixels
    cluster_sizes = set([i["cluster"]["size"] for i in sparkle.SDCAMeta.algo_meta])
    assert len(cluster_sizes) == num_sparkle_clusters
    assert np.sum(tuple(cluster_sizes)) == num_sparkle_pixels


def test_sparkle_nav():
    # test that the cluster centroid in the meta has the expected sparklenav data
    cluster_centroid_meta_sun_za_deg = sparkle.SDCAMeta.get_idx(cluster_centroid_idx_1)[
        "nav"
    ]["sun_za_deg"]
    cluster_centroid_nav_sun_za_deg = float(
        np.round(np.rad2deg(sparkle.nav.sun_za[cluster_centroid_idx_1].item()), 6)
    )
    assert (
        cluster_centroid_meta_sun_za_deg == cluster_centroid_nav_sun_za_deg == 21.87163
    )

    cluster_centroid_meta_sat_za_deg = sparkle.SDCAMeta.get_idx(cluster_centroid_idx_1)[
        "nav"
    ]["sat_za_deg"]
    cluster_centroid_nav_sat_za_deg = float(
        np.round(np.rad2deg(sparkle.nav.sat_za[cluster_centroid_idx_1].item()), 6)
    )
    assert (
        cluster_centroid_meta_sat_za_deg == cluster_centroid_nav_sat_za_deg == 44.928855
    )

    cluster_centroid_meta_beta_deg = sparkle.SDCAMeta.get_idx(cluster_centroid_idx_1)[
        "nav"
    ]["beta_deg"]
    cluster_centroid_nav_beta_deg = float(
        np.round(np.rad2deg(sparkle.nav.beta[cluster_centroid_idx_1].item()), 6)
    )
    assert cluster_centroid_meta_beta_deg == cluster_centroid_nav_beta_deg == 26.050493

    cluster_centroid_meta_glint_angle_deg = sparkle.SDCAMeta.get_idx(
        cluster_centroid_idx_1
    )["nav"]["glint_angle_deg"]
    cluster_centroid_nav_glint_angle_deg = float(
        np.round(np.rad2deg(sparkle.nav.glint_angle[cluster_centroid_idx_1].item()), 5)
    )
    assert (
        cluster_centroid_meta_glint_angle_deg
        == cluster_centroid_nav_glint_angle_deg
        == 49.0032
    )


def test_sparkle_flags():
    # test that the cluster centroid in the meta has the expected sparkleflags
    cluster_centroid_meta_flags = sparkle.SDCAMeta.get_idx(cluster_centroid_idx_1)[
        "flags"
    ]
    cluster_centroid_sparkleflags = list(
        sparkle.SDCAFlags.idx_decode(cluster_centroid_idx_1).values()
    )
    assert (
        cluster_centroid_meta_flags
        == cluster_centroid_sparkleflags
        == [
            "pixel_validated_by_pre_algo_masking",
            "pixel_prevalidated_by_max_rf_thresholds",
            "pixel_skipped_by_pre_algo_masking",
        ]
    )
    for flag in cluster_centroid_meta_flags:
        assert (
            sparkle.SDCAFlags.has_flag(
                sparkle.SDCAFlags.algo_flags[cluster_centroid_idx_1],
                sparkle.SDCAFlags.algo_flag_def[flag],
            )
            == True
        )

    sparkle.SDCADebug.idx_debug(cluster_centroid_idx_1)

    # test that a smaller cluster passed through the window deviation tests according to sparkleflags
    cluster_centroid_meta_flags = sparkle.SDCAMeta.get_idx(cluster_centroid_idx_2)[
        "flags"
    ]
    cluster_centroid_sparkleflags = list(
        sparkle.SDCAFlags.idx_decode(cluster_centroid_idx_2).values()
    )
    assert (
        cluster_centroid_meta_flags
        == cluster_centroid_sparkleflags
        == [
            "pixel_considered_on_first_pass",
            "pixel_had_1_window_iterations",
            "pixel_validated_by_window_deviation",
        ]
    )
    for flag in cluster_centroid_meta_flags:
        assert (
            sparkle.SDCAFlags.has_flag(
                sparkle.SDCAFlags.algo_flags[cluster_centroid_idx_2],
                sparkle.SDCAFlags.algo_flag_def[flag],
            )
            == True
        )

    sparkle.SDCADebug.idx_debug(cluster_centroid_idx_2)


def test_sparkle_stats():
    # check deviations
    cluster_centroid_meta_c02_rf_dev = sparkle.SDCAMeta.get_idx(cluster_centroid_idx_1)[
        "devs"
    ]["c02_rf"]
    cluster_centroid_stats_c02_rf_dev = np.round(
        sparkle.SDCAStats.get_deviation(cluster_centroid_idx_1, "c02_rf_deviation"), 7
    )
    assert cluster_centroid_meta_c02_rf_dev == cluster_centroid_stats_c02_rf_dev == 0.0

    cluster_centroid_meta_c05_rf_dev = sparkle.SDCAMeta.get_idx(cluster_centroid_idx_2)[
        "devs"
    ]["c05_rf"]
    cluster_centroid_stats_c05_rf_dev = np.round(
        sparkle.SDCAStats.get_deviation(cluster_centroid_idx_2, "c05_rf_deviation"), 7
    )
    assert (
        cluster_centroid_meta_c05_rf_dev
        == cluster_centroid_stats_c05_rf_dev
        == 0.5045494
    )

    cluster_centroid_meta_c07_rf_dev = sparkle.SDCAMeta.get_idx(cluster_centroid_idx_2)[
        "devs"
    ]["c07_rf"]
    cluster_centroid_stats_c07_rf_dev = np.round(
        sparkle.SDCAStats.get_deviation(cluster_centroid_idx_2, "c07_rf_deviation"), 7
    )
    assert (
        cluster_centroid_meta_c07_rf_dev
        == cluster_centroid_stats_c07_rf_dev
        == 0.1003152
    )

    cluster_centroid_meta_c14_bt_dev = sparkle.SDCAMeta.get_idx(cluster_centroid_idx_2)[
        "devs"
    ]["c14_bt"]
    cluster_centroid_stats_c14_bt_dev = np.round(
        sparkle.SDCAStats.get_deviation(cluster_centroid_idx_2, "c14_bt_deviation"), 5
    )
    assert (
        cluster_centroid_meta_c14_bt_dev == cluster_centroid_stats_c14_bt_dev == 1.05938
    )

    # check standard deviations
    cluster_centroid_meta_c02_rf_dev = sparkle.SDCAMeta.get_idx(cluster_centroid_idx_1)[
        "stdevs"
    ]["c02_rf"]
    cluster_centroid_stats_c02_rf_dev = np.round(
        sparkle.SDCAStats.get_deviation(cluster_centroid_idx_1, "c02_rf_stdev"), 7
    )
    assert cluster_centroid_meta_c02_rf_dev == cluster_centroid_stats_c02_rf_dev == 0.0

    cluster_centroid_meta_c05_rf_dev = sparkle.SDCAMeta.get_idx(cluster_centroid_idx_2)[
        "stdevs"
    ]["c05_rf"]
    cluster_centroid_stats_c05_rf_dev = np.round(
        sparkle.SDCAStats.get_deviation(cluster_centroid_idx_2, "c05_rf_stdev"), 7
    )
    assert (
        cluster_centroid_meta_c05_rf_dev
        == cluster_centroid_stats_c05_rf_dev
        == 0.0600242
    )

    cluster_centroid_meta_c07_rf_dev = sparkle.SDCAMeta.get_idx(cluster_centroid_idx_2)[
        "stdevs"
    ]["c07_rf"]
    cluster_centroid_stats_c07_rf_dev = np.round(
        sparkle.SDCAStats.get_deviation(cluster_centroid_idx_2, "c07_rf_stdev"), 7
    )
    assert (
        cluster_centroid_meta_c07_rf_dev
        == cluster_centroid_stats_c07_rf_dev
        == 0.0418818
    )

    cluster_centroid_meta_c14_bt_dev = sparkle.SDCAMeta.get_idx(cluster_centroid_idx_2)[
        "stdevs"
    ]["c14_bt"]
    cluster_centroid_stats_c14_bt_dev = np.round(
        sparkle.SDCAStats.get_deviation(cluster_centroid_idx_2, "c14_bt_stdev"), 5
    )
    assert (
        cluster_centroid_meta_c14_bt_dev == cluster_centroid_stats_c14_bt_dev == 2.88706
    )

    # check debug
    cluster_centroid_meta_algo_passes = sparkle.SDCAMeta.get_idx(
        cluster_centroid_idx_1
    )["debug"]["algo_passes"]
    cluster_centroid_stats_algo_passes = int(
        sparkle.SDCAStats.get_debug(cluster_centroid_idx_1, "algo_passes")
    )
    assert cluster_centroid_meta_algo_passes == cluster_centroid_stats_algo_passes == 0

    cluster_centroid_meta_window_radius = sparkle.SDCAMeta.get_idx(
        cluster_centroid_idx_2
    )["debug"]["window_radius"]
    cluster_centroid_stats_window_radius = int(
        sparkle.SDCAStats.get_debug(cluster_centroid_idx_2, "window_radius")
    )
    assert (
        cluster_centroid_meta_window_radius
        == cluster_centroid_stats_window_radius
        == 15
    )

    cluster_centroid_meta_window_iterations = sparkle.SDCAMeta.get_idx(
        cluster_centroid_idx_2
    )["debug"]["window_iterations"]
    cluster_centroid_stats_window_iterations = int(
        sparkle.SDCAStats.get_debug(cluster_centroid_idx_2, "window_iterations")
    )
    assert (
        cluster_centroid_meta_window_iterations
        == cluster_centroid_stats_window_iterations
        == 1
    )

    cluster_centroid_meta_window_valid_proportion = sparkle.SDCAMeta.get_idx(
        cluster_centroid_idx_2
    )["debug"]["window_valid_proportion"]
    cluster_centroid_stats_window_valid_proportion = np.round(
        sparkle.SDCAStats.get_debug(cluster_centroid_idx_2, "window_valid_proportion"),
        7,
    )
    assert (
        cluster_centroid_meta_window_valid_proportion
        == cluster_centroid_stats_window_valid_proportion
        == 0.9979188
    )


def test_border_and_water_and_cloud():
    # assert that a pixel can have multiple simultaneous flag classifications and mask states
    assert (
        sparkle.SDCAFlags.has_flag(
            sparkle.SDCAFlags.algo_flags[border_and_water_and_cloud_idx],
            sparkle.SDCAFlags.algo_flag_def["pixel_skipped_by_border_mask"],
        )
        == sparkle.SDCAMask.discard_mask[border_and_water_and_cloud_idx]
        == True
    )
    assert (
        sparkle.SDCAFlags.has_flag(
            sparkle.SDCAFlags.algo_flags[border_and_water_and_cloud_idx],
            sparkle.SDCAFlags.algo_flag_def["pixel_skipped_by_cloud_mask"],
        )
        == sparkle.cloud_mask[border_and_water_and_cloud_idx]
        == sparkle.SDCAMask.skip_mask[border_and_water_and_cloud_idx]
        == True
    )
    assert (
        sparkle.SDCAFlags.has_flag(
            sparkle.SDCAFlags.algo_flags[border_and_water_and_cloud_idx],
            sparkle.SDCAFlags.algo_flag_def["pixel_preinvalidated_by_water_mask"],
        )
        == ~sparkle.water_mask[border_and_water_and_cloud_idx]
        == sparkle.SDCAMask.invalidated_mask[border_and_water_and_cloud_idx]
        == True
    )

    sparkle.SDCADebug.idx_debug(border_and_water_and_cloud_idx)


def test_cloud():
    # test that a cloud pixel is declared as cloud by sparkleflags and sparklemask
    assert (
        sparkle.SDCAFlags.has_flag(
            sparkle.SDCAFlags.algo_flags[cloud_idx],
            sparkle.SDCAFlags.algo_flag_def["pixel_skipped_by_cloud_mask"],
        )
        == sparkle.cloud_mask[cloud_idx]
        == True
    )

    sparkle.SDCADebug.idx_debug(cloud_idx)


def test_water():
    # test that a water pixel is declared as water by sparkleflags and sparklemask
    assert (
        sparkle.SDCAFlags.has_flag(
            sparkle.SDCAFlags.algo_flags[water_idx],
            sparkle.SDCAFlags.algo_flag_def["pixel_preinvalidated_by_water_mask"],
        )
        == ~sparkle.water_mask[water_idx]
        == True
    )

    sparkle.SDCADebug.idx_debug(water_idx)


def test_nirrefl():
    assert sparkle.c07_nirrefl.rf.dtype == np.float32
    assert sparkle.c07_nirrefl.rf[cluster_centroid_idx_1].item() == 10.882692337036133
    assert sparkle.c07_nirrefl.rf[cluster_centroid_idx_2].item() == 0.23989807069301605
