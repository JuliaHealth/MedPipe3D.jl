using Test

@testset "MedPipe3D Tests" begin

include("test_dataset_to_hdf5.jl")
include("test_hdf5_to_nifti.jl")
include("test_training.jl")

end