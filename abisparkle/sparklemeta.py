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

"""Stores indices and associated algorithm metadata for detected sparkle pixels"""

import uuid

import numpy as np
from heregoes.instrument.abi import rad_wvn2wvl
from scipy import ndimage

from abisparkle.sparklenav import SparkleNavigation

db_time_format = "%Y-%m-%dT%H:%M:%SZ"
safe_time_format = "%Y-%m-%dT%H%M%SZ"


class SDCAMeta:
    def __init__(self, sparkle):
        self.sparkle = sparkle

        self.algo_meta = []

        self.valid_clusters, self.num_clusters = ndimage.label(
            sparkle.valid_sparkles, structure=np.ones((3, 3))
        )

        # maps cluster UUIDs to cluster centroid indices
        self.cluster_map = {}

        # for each cluster of adjacent True pixels in the valid_sparkle image:
        for cluster in range(1, self.num_clusters + 1):

            # get some information about each cluster
            cluster_id = (
                sparkle.source_meta.time_coverage_start.strftime(safe_time_format)
                + "_"
                + str(uuid.uuid4())
            )
            cluster_centroid_idx = tuple(
                np.floor(
                    np.mean(np.nonzero(self.valid_clusters == cluster), axis=1)
                ).astype(np.uint16)
            )

            cluster_centroid_lat = float(
                np.round(sparkle.nav.lat_deg[cluster_centroid_idx].item(), 5)
            )
            cluster_centroid_lon = float(
                np.round(sparkle.nav.lon_deg[cluster_centroid_idx].item(), 5)
            )

            (
                cluster_centroid_omega,
                cluster_centroid_beta,
                cluster_centroid_gamma,
            ) = SparkleNavigation.calc_reflections(
                np.atleast_1d(sparkle.nav.sun_az[cluster_centroid_idx]),
                np.atleast_1d(sparkle.nav.sun_za[cluster_centroid_idx]),
                np.atleast_1d(sparkle.nav.sat_az[cluster_centroid_idx]),
                np.atleast_1d(sparkle.nav.sat_za[cluster_centroid_idx]),
            )

            self.cluster_map[str(cluster_id)] = cluster_centroid_idx
            num_in_cluster = np.count_nonzero(self.valid_clusters == cluster)

            # for each index in each cluster:
            for idx in np.argwhere(self.valid_clusters == cluster):
                idx = tuple((int((idx[0])), int(idx[1])))

                idx_lat = float(np.round(sparkle.nav.lat_deg[idx].item(), 5))
                idx_lon = float(np.round(sparkle.nav.lon_deg[idx].item(), 5))

                # these values are calculated explicitly rather than creating a single-index SparkleNavigation object for each index, or computing on an entire image
                idx_omega, idx_beta, idx_gamma = SparkleNavigation.calc_reflections(
                    np.atleast_1d(sparkle.nav.sun_az[idx]),
                    np.atleast_1d(sparkle.nav.sun_za[idx]),
                    np.atleast_1d(sparkle.nav.sat_az[idx]),
                    np.atleast_1d(sparkle.nav.sat_za[idx]),
                )
                idx_area_m = super(type(sparkle.nav), sparkle.nav).pixel_area(
                    np.atleast_1d(sparkle.nav.y_rad[idx]),
                    np.atleast_1d(sparkle.nav.x_rad[idx]),
                    np.atleast_1d(sparkle.nav.abi_meta.instrument_meta.semi_major_axis),
                    np.atleast_1d(
                        sparkle.nav.abi_meta.instrument_meta.perspective_point_height
                    ),
                    np.atleast_1d(sparkle.nav.abi_meta.instrument_meta.ifov),
                )

                # store the emissive radiance in wavelength space to match reflective radiance
                c07_rad_wvl = rad_wvn2wvl(
                    np.atleast_1d(sparkle.c07_image.rad[idx]),
                    *sparkle.c07_image.meta.instrument_meta.coefficients.eqw,
                )
                c14_rad_wvl = rad_wvn2wvl(
                    np.atleast_1d(sparkle.c14_image.rad[idx]),
                    *sparkle.c14_image.meta.instrument_meta.coefficients.eqw,
                )

                idx_meta = {
                    "event": "valid_sparkle",
                    "time_coverage_start": sparkle.c02_image.meta.time_coverage_start.strftime(
                        db_time_format
                    ),
                    "time_coverage_end": sparkle.c02_image.meta.time_coverage_end.strftime(
                        db_time_format
                    ),
                    "y": int(idx[0]),
                    "x": int(idx[1]),
                    "lat": idx_lat,
                    "lon": idx_lon,
                    "google_maps": f"https://www.google.com/maps/@?api=1&map_action=map&center={idx_lat},{idx_lon}&zoom=14&basemap=satellite",
                    "cluster": {
                        "id": str(cluster_id),
                        "centroid_y": int(cluster_centroid_idx[0]),
                        "centroid_x": int(cluster_centroid_idx[1]),
                        "centroid_lat": cluster_centroid_lat,
                        "centroid_lon": cluster_centroid_lon,
                        "centroid_google_maps": f"https://www.google.com/maps/@?api=1&map_action=map&center={cluster_centroid_lat},{cluster_centroid_lon}&zoom=14&basemap=satellite",
                        "centroid_omega_deg": float(
                            np.round(np.rad2deg(cluster_centroid_omega.item()), 5)
                        ),
                        "centroid_beta_deg": float(
                            np.round(np.rad2deg(cluster_centroid_beta.item()), 6)
                        ),
                        "centroid_gamma_deg": float(
                            np.round(np.rad2deg(cluster_centroid_gamma.item()), 5)
                        ),
                        "size": num_in_cluster,
                    },
                    "files": {
                        "c02": sparkle.c02_image.meta.dataset_name,
                        "c05": sparkle.c05_image.meta.dataset_name,
                        "c07": sparkle.c07_image.meta.dataset_name,
                        "c14": sparkle.c14_image.meta.dataset_name,
                    },
                    "dqfs": {
                        "c02": int(sparkle.c02_image.dqf[idx].item()),
                        "c05": int(sparkle.c05_image.dqf[idx].item()),
                        "c07": int(sparkle.c07_image.dqf[idx].item()),
                        "c14": int(sparkle.c14_image.dqf[idx].item()),
                    },
                    "rads": {
                        "c02": float(np.round(sparkle.c02_image.rad[idx].item(), 5)),
                        "c05": float(np.round(sparkle.c05_image.rad[idx].item(), 5)),
                        "c07": float(np.round(c07_rad_wvl.item(), 5)),
                        "c14": float(np.round(c14_rad_wvl.item(), 5)),
                    },
                    "rfs": {
                        "c02": float(np.round(sparkle.c02_image.cmi[idx].item(), 7)),
                        "c05": float(np.round(sparkle.c05_image.cmi[idx].item(), 7)),
                        "c07": float(np.round(sparkle.c07_nirrefl.rf[idx].item(), 7)),
                    },
                    "bts": {
                        "c07": float(np.round(sparkle.c07_image.cmi[idx].item(), 5)),
                        "c14": float(np.round(sparkle.c14_image.cmi[idx].item(), 5)),
                    },
                    "devs": {
                        "c02_rf": float(
                            np.round(
                                self.sparkle.SDCAStats.get_deviation(
                                    idx, "c02_rf_deviation"
                                ),
                                7,
                            )
                        ),
                        "c05_rf": float(
                            np.round(
                                self.sparkle.SDCAStats.get_deviation(
                                    idx, "c05_rf_deviation"
                                ),
                                7,
                            )
                        ),
                        "c07_rf": float(
                            np.round(
                                self.sparkle.SDCAStats.get_deviation(
                                    idx, "c07_rf_deviation"
                                ),
                                7,
                            )
                        ),
                        "c14_bt": float(
                            np.round(
                                self.sparkle.SDCAStats.get_deviation(
                                    idx, "c14_bt_deviation"
                                ),
                                5,
                            )
                        ),
                    },
                    "stdevs": {
                        "c02_rf": float(
                            np.round(
                                self.sparkle.SDCAStats.get_deviation(
                                    idx, "c02_rf_stdev"
                                ),
                                7,
                            )
                        ),
                        "c05_rf": float(
                            np.round(
                                self.sparkle.SDCAStats.get_deviation(
                                    idx, "c05_rf_stdev"
                                ),
                                7,
                            )
                        ),
                        "c07_rf": float(
                            np.round(
                                self.sparkle.SDCAStats.get_deviation(
                                    idx, "c07_rf_stdev"
                                ),
                                7,
                            )
                        ),
                        "c14_bt": float(
                            np.round(
                                self.sparkle.SDCAStats.get_deviation(
                                    idx, "c14_bt_stdev"
                                ),
                                5,
                            )
                        ),
                    },
                    "nav": {
                        # these angles are already calculated in SDCA for an entire image
                        "sun_za_deg": float(
                            np.round(np.rad2deg(sparkle.nav.sun_za[idx].item()), 6)
                        ),
                        "sun_az_deg": float(
                            np.round(np.rad2deg(sparkle.nav.sun_az[idx].item()), 5)
                        ),
                        "sat_za_deg": float(
                            np.round(np.rad2deg(sparkle.nav.sat_za[idx].item()), 6)
                        ),
                        "sat_az_deg": float(
                            np.round(np.rad2deg(sparkle.nav.sat_az[idx].item()), 5)
                        ),
                        "glint_angle_deg": float(
                            np.round(np.rad2deg(sparkle.nav.glint_angle[idx].item()), 5)
                        ),
                        # these are calculated on a per-pixel basis for sparklemeta
                        "omega_deg": float(np.round(np.rad2deg(idx_omega.item()), 5)),
                        "beta_deg": float(np.round(np.rad2deg(idx_beta.item()), 6)),
                        "gamma_deg": float(np.round(np.rad2deg(idx_gamma.item()), 5)),
                        "area_m": float(np.round(idx_area_m.item(), 2)),
                    },
                    "flags": list(sparkle.SDCAFlags.idx_decode(idx).values()),
                    "debug": {
                        "algo_passes": int(
                            self.sparkle.SDCAStats.get_debug(idx, "algo_passes")
                        ),
                        "window_radius": int(
                            self.sparkle.SDCAStats.get_debug(idx, "window_radius")
                        ),
                        "window_iterations": int(
                            self.sparkle.SDCAStats.get_debug(idx, "window_iterations")
                        ),
                        "window_valid_proportion": float(
                            np.round(
                                self.sparkle.SDCAStats.get_debug(
                                    idx, "window_valid_proportion"
                                ),
                                7,
                            )
                        ),
                    },
                }

                self.algo_meta.append(idx_meta)

    def get_idx(self, idx):
        y, x = idx
        for i in self.algo_meta:
            if i["y"] == y and i["x"] == x:
                return i

        return None

    def get_cluster_members(self, cluster_id):
        cluster_members = []
        for i in self.algo_meta:
            if i["cluster"]["id"] == cluster_id:
                cluster_members.append(i)

        return cluster_members

    def get_clusters(self):
        clusters = []

        for cluster_id in self.cluster_map.keys():
            clusters.append(self.get_cluster_members(cluster_id)[0]["cluster"])

        return clusters
