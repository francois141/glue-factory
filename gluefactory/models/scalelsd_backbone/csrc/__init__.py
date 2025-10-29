import glob
import os.path as osp

from torch.utils.cpp_extension import load

__this__ = osp.dirname(__file__)

try:
    _C = load(
        name="_C",
        sources=[
            osp.join(__this__, "binding.cpp"),
            osp.join(__this__, "linesegment.cu"),
        ],
    )
except:
    _C = None

__all__ = ["_C"]
