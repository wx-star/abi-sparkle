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

"""Stores algorithm-generated statistics and debug information per pixel index"""

from numba.core import types as ntypes
from numba.experimental import jitclass
from numba.typed import Dict as ndict

ndict_inner_kv_ty = (ntypes.unicode_type, ntypes.float32)
ndict_outer_kv_ty = (
    ntypes.UniTuple(ntypes.int64, 2),
    ntypes.DictType(*ndict_inner_kv_ty),
)
spec = [
    ("deviation_statistics", ntypes.DictType(*ndict_outer_kv_ty)),
    ("debug_statistics", ntypes.DictType(*ndict_outer_kv_ty)),
    ("_empty_child", ntypes.DictType(*ndict_inner_kv_ty)),
]


@jitclass(spec)
class SDCAStats:
    def __init__(self):
        self.deviation_statistics = ndict.empty(*ndict_outer_kv_ty)
        self.debug_statistics = ndict.empty(*ndict_outer_kv_ty)
        self._empty_child = ndict.empty(*ndict_inner_kv_ty)

    def _init_child(self, _dict, idx):
        _dict[idx] = self._empty_child.copy()

    def _is_empty_child(self, _dict, idx):
        try:
            val = _dict.get(idx)

            return val is None or val == self._empty_child

        except:
            return True

    def _set(self, _dict, idx, key, value):
        _dict[idx][key] = ntypes.float32(value)

    def _get(self, _dict, idx, key, default=ntypes.float32(0.0)):
        try:
            if self._is_empty_child(_dict, idx):
                return default

            else:
                return _dict[idx][key]

        except:
            return default

    # @jitclass is experimental so these are concrete methods for now
    def set_deviation(self, idx, key, value):
        if self._is_empty_child(self.deviation_statistics, idx):
            self._init_child(self.deviation_statistics, idx)
        self._set(self.deviation_statistics, idx, key, value)

    def set_debug(self, idx, key, value):
        if self._is_empty_child(self.debug_statistics, idx):
            self._init_child(self.debug_statistics, idx)
        self._set(self.debug_statistics, idx, key, value)

    def get_deviation(self, idx, key, default=ntypes.float32(0.0)):
        return self._get(self.deviation_statistics, idx, key, default=default)

    def get_debug(self, idx, key, default=ntypes.float32(0.0)):
        return self._get(self.debug_statistics, idx, key, default=default)
