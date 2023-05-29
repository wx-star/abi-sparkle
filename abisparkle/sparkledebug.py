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

"""Debug tool for evaluating algorithm decisions by pixel index"""


class SDCADebug:
    def __init__(self, sparkle):
        self.sparkle = sparkle

    def idx_debug(self, idx):
        deviation_default = 999.0
        debug_default = 0.0

        print("---------------------")
        print("Index:", idx)
        print("---------------------")
        print("")

        # pixel values
        print("---------------------")
        print("Pixel values:")
        print("---------------------")
        print("")

        print("C02 RF:", self.sparkle.c02_image.cmi[idx], end="")
        if (
            self.sparkle.c02_image.cmi[idx]
            > self.sparkle.SDCAParams.algo_params["c02_rf_min_threshold"]
        ):
            print(" | VALID")
        else:
            print(" | FAIL")

        print("C05 RF:", self.sparkle.c05_image.cmi[idx], end="")
        if (
            self.sparkle.c05_image.cmi[idx]
            > self.sparkle.SDCAParams.algo_params["c05_rf_min_threshold"]
        ):
            print(" | VALID")
        else:
            print(" | FAIL")

        print("C07 RF:", self.sparkle.c07_nirrefl.rf[idx], end="")
        if (
            self.sparkle.c07_nirrefl.rf[idx]
            > self.sparkle.SDCAParams.algo_params["c07_rf_min_threshold"]
        ):
            print(" | VALID")
        else:
            print(" | FAIL")

        print("C07 BT:", self.sparkle.c07_image.cmi[idx], end="")
        if (
            self.sparkle.c07_image.cmi[idx]
            > self.sparkle.SDCAParams.algo_params["c07_bt_min_threshold"]
        ):
            print(" | VALID")
        else:
            print(" | FAIL")

        print("C14 BT:", self.sparkle.c14_image.cmi[idx], end="")
        if (
            self.sparkle.c14_image.cmi[idx]
            > self.sparkle.SDCAParams.algo_params["c14_bt_min_threshold"]
        ):
            print(" | VALID")
        else:
            print(" | FAIL")

        # window deviation statistics
        print("")
        print("---------------------")
        print("Deviation statistics:")
        print("---------------------")
        print("")

        c02_rf_deviation = self.sparkle.SDCAStats.get_deviation(
            idx, "c02_rf_deviation", deviation_default
        )
        print("C02 RF deviation:", c02_rf_deviation, end="")
        if c02_rf_deviation != deviation_default:
            if (
                c02_rf_deviation
                > self.sparkle.SDCAParams.algo_params["c02_rf_deviation_min_threshold"]
            ):
                print(" | VALID")
            else:
                print(" | FAIL")
        else:
            print(" | N/A")

        c05_rf_deviation = self.sparkle.SDCAStats.get_deviation(
            idx, "c05_rf_deviation", deviation_default
        )
        print("C05 RF deviation:", c05_rf_deviation, end="")
        if c05_rf_deviation != deviation_default:
            if (
                c05_rf_deviation
                > self.sparkle.SDCAParams.algo_params["c05_rf_deviation_min_threshold"]
            ):
                print(" | VALID")
            else:
                print(" | FAIL")
        else:
            print(" | N/A")

        c07_rf_deviation = self.sparkle.SDCAStats.get_deviation(
            idx, "c07_rf_deviation", deviation_default
        )
        print("C07 RF deviation:", c07_rf_deviation, end="")
        if c07_rf_deviation != deviation_default:
            if (
                c07_rf_deviation
                > self.sparkle.SDCAParams.algo_params["c07_rf_deviation_min_threshold"]
            ):
                print(" | VALID")
            else:
                print(" | FAIL")
        else:
            print(" | N/A")

        c14_bt_deviation = self.sparkle.SDCAStats.get_deviation(
            idx, "c14_bt_deviation", deviation_default
        )
        print("C14 BT deviation:", c14_bt_deviation, end="")
        if c14_bt_deviation != deviation_default:
            if (
                c14_bt_deviation
                > self.sparkle.SDCAParams.algo_params["c14_bt_deviation_min_threshold"]
            ):
                print(" | VALID")
            else:
                print(" | FAIL")
        else:
            print(" | N/A")

        c02_rf_stdev = self.sparkle.SDCAStats.get_deviation(
            idx, "c02_rf_stdev", deviation_default
        )
        # unused and not currently defined in algo_params
        print("C02 RF stdev:", c02_rf_stdev, "| N/A")
        # if c02_rf_stdev != deviation_default:
        #     if (
        #         c02_rf_stdev
        #         > self.sparkle.SDCAParams.algo_params["c02_rf_stdev_max_threshold"]
        #     ):
        #         print(" | VALID")
        #     else:
        #         print(" | FAIL")
        # else:
        #     print(" | N/A")

        c05_rf_stdev = self.sparkle.SDCAStats.get_deviation(
            idx, "c05_rf_stdev", deviation_default
        )
        # unused and not currently defined in algo_params
        print("C05 RF stdev:", c05_rf_stdev, "| N/A")
        # if c05_rf_stdev != deviation_default:
        #     if (
        #         c05_rf_stdev
        #         > self.sparkle.SDCAParams.algo_params["c05_rf_stdev_max_threshold"]
        #     ):
        #         print(" | VALID")
        #     else:
        #         print(" | FAIL")
        # else:
        #     print(" | N/A")

        c07_rf_stdev = self.sparkle.SDCAStats.get_deviation(
            idx, "c07_rf_stdev", deviation_default
        )
        # unused and not currently defined in algo_params
        print("C07 RF stdev:", c07_rf_stdev, "| N/A")
        # if c07_rf_stdev != deviation_default:
        #     if (
        #         c07_rf_stdev
        #         > self.sparkle.SDCAParams.algo_params["c07_rf_stdev_max_threshold"]
        #     ):
        #         print(" | VALID")
        #     else:
        #         print(" | FAIL")
        # else:
        #     print(" | N/A")

        c14_bt_stdev = self.sparkle.SDCAStats.get_deviation(
            idx, "c14_bt_stdev", deviation_default
        )
        print("C14 BT stdev:", c14_bt_stdev, end="")
        if c14_bt_stdev != deviation_default:
            if (
                c14_bt_stdev
                <= self.sparkle.SDCAParams.algo_params[
                    "c14_bt_standard_deviation_max_threshold"
                ]
            ):
                print(" | VALID")
            else:
                print(" | FAIL")
        else:
            print(" | N/A")

        # flags
        print("")
        print("---------------------")
        print("Pixel flags:")
        print("---------------------")
        print("")
        [print(i) for i in self.sparkle.SDCAFlags.idx_decode(idx).values()]

        # debug_statistics
        print("")
        print("---------------------")
        print("Debug statistics:")
        print("---------------------")
        print("")

        algo_passes = self.sparkle.SDCAStats.get_debug(
            idx, "algo_passes", debug_default
        )
        print("Algorithm passes:", algo_passes, end="")
        if algo_passes != debug_default:
            if algo_passes <= self.sparkle.SDCAParams.algo_params["max_algo_passes"]:
                print(" | VALID")
            else:
                print(" | FAIL")
        else:
            print(" | N/A")

        window_radius = self.sparkle.SDCAStats.get_debug(
            idx, "window_radius", debug_default
        )
        print("Window radius:", window_radius, end="")
        if window_radius != debug_default:
            if (
                window_radius
                <= self.sparkle.SDCAParams.algo_params["first_window_radius"]
                * self.sparkle.SDCAParams.algo_params["max_window_radius_iter"]
            ):
                print(" | VALID")
            else:
                print(" | FAIL")
        else:
            print(" | N/A")

        window_iterations = self.sparkle.SDCAStats.get_debug(
            idx, "window_iterations", debug_default
        )
        print("Window iterations:", window_iterations, end="")
        if window_iterations != debug_default:
            if (
                window_iterations
                <= self.sparkle.SDCAParams.algo_params["max_window_radius_iter"]
            ):
                print(" | VALID")
            else:
                print(" | FAIL")
        else:
            print(" | N/A")

        window_valid_proportion = self.sparkle.SDCAStats.get_debug(
            idx, "window_valid_proportion", debug_default
        )
        print(
            "Window valid proportion:",
            window_valid_proportion,
            end="",
        )
        if window_valid_proportion != debug_default:
            if (
                window_valid_proportion
                >= self.sparkle.SDCAParams.algo_params[
                    "min_window_clean_proportion_threshold"
                ]
            ):
                print(" | VALID")
            else:
                print(" | FAIL")
        else:
            print(" | N/A")
