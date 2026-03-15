#  utils.jl
#  Master entry point — include all utility modules in dependency order.
#
#  Usage:
#      include("utils/utils.jl")

include("parsing_utils.jl")   # string/tuple/optimizer arg helpers — no deps
include("img_utils.jl")     # MedImage geometry & intensity ops
include("io_utils.jl")        # HDF5 r/w, NIfTI conversion, JSON split helpers
include("model_utils.jl")     # Lux inference, tensor inspection
include("debug_utils.jl")     # HDF5 tree printer, other diagnostics