using Test

@testset "MedPipe3D Tests" begin

include("dataset_to_hdf5.jl")
include("hdf5_to_nifti.jl")
include("training_test.jl")

end