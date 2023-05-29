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

"""Windowed deviation detection algorithm"""

import numpy as np
from heregoes import heregoes_njit_noparallel, util
from numba.core import types as ntypes


@heregoes_njit_noparallel
def sparkle(
    c02_rf,
    c05_rf,
    c07_rf,
    c14_bt,
    validated_mask,
    discard_mask,
    skip_mask,
    bad_dqf_mask,
    algo_params,
    algo_flags,
    algo_stats,
):
    def validate(idx):
        # when we find a valid sparkle pixel:
        validated_mask[idx] = True  # mark as valid
        skip_mask[idx] = True  # remove from the iteration loop
        discard_mask[
            idx
        ] = True  # remove from the statistical background of other sparkles

    def invalidate(idx):
        # when we find an invalid pixel that shouldn't be considered again:
        skip_mask[idx] = True  # remove from the iteration loop

    algo_passes = 1
    while algo_passes <= algo_params["max_algo_passes"]:

        # loop over every pixel marked "False" in skip_mask
        for idx in np.argwhere(~skip_mask):
            idx = tuple((ntypes.int64(idx[0]), ntypes.int64(idx[1])))

            algo_flags.set_flag(
                idx,
                algo_flags.algo_flag_def["flag_offset_algo_passes"] + algo_passes,
            )

            # skip pixels that are within exclude_dqf_radius of a bad DQF
            bad_dqf_window = util.window_slice(
                bad_dqf_mask,
                idx,
                outer_radius=algo_params["exclude_dqf_radius"],
                replace_inner=False,
            )
            if bad_dqf_window.any():
                invalidate(idx)
                algo_flags.set_flag(
                    idx,
                    algo_flags.algo_flag_def["pixel_invalidated_by_dqf_neighbor"],
                )
                continue

            # determine the appropriate size of the background window based on clean proportions of discard_mask
            (
                window_valid,
                window_radius,
                window_iter,
                window_valid_proportion,
            ) = window_sizer(
                discard_mask,
                idx,
                first_radius=algo_params["first_window_radius"],
                max_window_iter=algo_params["max_window_radius_iter"],
                min_window_clean_proportion_threshold=algo_params[
                    "min_window_clean_proportion_threshold"
                ],
            )
            algo_flags.set_flag(
                idx,
                algo_flags.algo_flag_def["flag_offset_window_iterations"]
                + np.int64(window_iter),
            )

            algo_stats.set_debug(idx, "algo_passes", algo_passes)
            algo_stats.set_debug(idx, "window_radius", window_radius)
            algo_stats.set_debug(idx, "window_iterations", window_iter)
            algo_stats.set_debug(
                idx, "window_valid_proportion", window_valid_proportion
            )

            # skip pixels where we couldn't get a clean window
            if not window_valid:
                # the clean window proportion will never increase on subsequent passes, so invalidate this pixel
                invalidate(idx)
                algo_flags.set_flag(
                    idx,
                    algo_flags.algo_flag_def["pixel_invalidated_by_window_sizing"],
                )
                continue

            # take windows of each image and discard validated and invalidated pixels from the statistical background
            discard_mask_window = util.window_slice(
                discard_mask, idx, outer_radius=window_radius, replace_inner=True
            )

            c02_rf_window = util.window_slice(
                c02_rf, idx, outer_radius=window_radius, replace_inner=True
            )
            c02_rf_window.ravel()[np.nonzero(discard_mask_window.ravel())] = np.nan

            c05_rf_window = util.window_slice(
                c05_rf, idx, outer_radius=window_radius, replace_inner=True
            )
            c05_rf_window.ravel()[np.nonzero(discard_mask_window.ravel())] = np.nan

            c07_rf_window = util.window_slice(
                c07_rf, idx, outer_radius=window_radius, replace_inner=True
            )
            c07_rf_window.ravel()[np.nonzero(discard_mask_window.ravel())] = np.nan

            c14_bt_window = util.window_slice(
                c14_bt, idx, outer_radius=window_radius, replace_inner=True
            )
            c14_bt_window.ravel()[np.nonzero(discard_mask_window.ravel())] = np.nan

            # record the window statistics
            algo_stats.set_deviation(
                idx, "c02_rf_deviation", c02_rf[idx] - np.nanmean(c02_rf_window)
            )
            algo_stats.set_deviation(
                idx, "c05_rf_deviation", c05_rf[idx] - np.nanmean(c05_rf_window)
            )
            algo_stats.set_deviation(
                idx, "c07_rf_deviation", c07_rf[idx] - np.nanmean(c07_rf_window)
            )
            algo_stats.set_deviation(
                idx, "c14_bt_deviation", c14_bt[idx] - np.nanmean(c14_bt_window)
            )

            # record the window statistics
            algo_stats.set_deviation(idx, "c02_rf_stdev", np.nanstd(c02_rf_window))
            algo_stats.set_deviation(idx, "c05_rf_stdev", np.nanstd(c05_rf_window))
            algo_stats.set_deviation(idx, "c07_rf_stdev", np.nanstd(c07_rf_window))
            algo_stats.set_deviation(idx, "c14_bt_stdev", np.nanstd(c14_bt_window))

            # sparkle validation parameters based on window statistics
            if (
                (
                    algo_stats.get_deviation(idx, "c02_rf_deviation")
                    > algo_params["c02_rf_deviation_min_threshold"]
                )
                and (
                    algo_stats.get_deviation(idx, "c05_rf_deviation")
                    > algo_params["c05_rf_deviation_min_threshold"]
                )
                and (
                    algo_stats.get_deviation(idx, "c07_rf_deviation")
                    > algo_params["c07_rf_deviation_min_threshold"]
                )
                and (
                    algo_stats.get_deviation(idx, "c14_bt_deviation")
                    > algo_params["c14_bt_deviation_min_threshold"]
                )
                and (
                    algo_stats.get_deviation(idx, "c14_bt_stdev")
                    <= algo_params["c14_bt_standard_deviation_max_threshold"]
                )
            ):
                validate(idx)
                algo_flags.set_flag(
                    idx,
                    algo_flags.algo_flag_def["pixel_validated_by_window_deviation"],
                )

        algo_passes += 1

        # do not do another pass if nothing was found in the first place
        if np.count_nonzero(validated_mask) == 0:
            break

    return validated_mask


@heregoes_njit_noparallel
def window_sizer(
    arr,
    idx,
    first_radius=15,
    max_window_iter=3,
    min_window_clean_proportion_threshold=0.75,
):
    """
    Determines the appropriate window size by checking whether a majority of pixels in the window are valid numbers (not nans).
    Returns the radius of the window and whether the window is valid (majority of pixels are not nans, and the window size iterated less than max_window_iter times)
    or invalid (majority of pixels are nans after max_window_iter iterations)
    """
    window_iter = 1
    while window_iter <= max_window_iter:
        window_radius = first_radius * window_iter
        window = util.window_slice(
            arr, idx, outer_radius=window_radius, replace_inner=True
        )

        if window.size == int(np.square(2 * window_radius + 1)):
            window_valid_proportion = np.count_nonzero(~window) / window.size

            if window_valid_proportion > min_window_clean_proportion_threshold:
                window_valid = True
                return window_valid, window_radius, window_iter, window_valid_proportion

        window_iter += 1

    # invalidate the window if we exceed max_window_iter iterations without getting a clean window
    window_valid = False
    return window_valid, window_radius, window_iter, window_valid_proportion
