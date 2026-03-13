using Test

try
    using MedPipe3D
catch
    include(joinpath(@__DIR__, "..", "src", "MedPipe3D.jl"))
    using .MedPipe3D
end

@testset "HDF5 → NIfTI conversion" begin

repo_root = joinpath(@__DIR__, "..")

pathToHDF5 = joinpath(repo_root,"dataset","HDF5","heart_dataset.hdf5")
output_dir = joinpath(repo_root,"dataset","nifti_output")

@test isfile(pathToHDF5)

mkpath(output_dir)

MedPipe3D.convert_hdf5_to_medimages(pathToHDF5,output_dir)

@test isdir(output_dir)

files = readdir(output_dir)

@test length(files) > 0

end