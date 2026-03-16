ENV["JULIA_CUDA_MEMORY_POOL"] = "none"
ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "3500MiB"
using Test
using JSON
using CUDA
# Limit CUDA memory pool to leave room for training

CUDA.reclaim()
GC.gc()
# CUDA.pool_reclaim()

# Tell CUDA not to reserve all memory upfront
ENV["JULIA_CUDA_MEMORY_POOL"] = "none"  # disable pool, allocate directly
try
    using MedPipe3D
catch
    include(joinpath(@__DIR__, "..", "src", "MedPipe3D.jl"))
    using .MedPipe3D
end

@testset "Training pipeline" begin

repo_root = joinpath(@__DIR__, "..")

pathToHDF5 = joinpath(repo_root,"dataset","HDF5","heart_dataset.hdf5")

@test isfile(pathToHDF5)

config_dir = joinpath(repo_root,"dataset","config")
mkpath(config_dir)

config_path = joinpath(config_dir,"config.json")

config = Dict(

"data"=>Dict(
"batch_size"=>1,
"batch_complete"=>false,
"channel_size_imgs"=>1,
"channel_size_masks"=>1,
"normalization"=>true,
"has_mask"=>true
),

"augmentation"=>Dict(
"order"=>[],
"p_rand"=>0.5,
"augmentations"=>Dict(),
"processing_unit"=>"GPU"
),

"learning"=>Dict(
"test_train_validation"=>(0.6,0.2,0.2),
"shuffle"=>true,
"patch_size"=>[32,32,32],
"patch_probabilistic_oversampling"=>true,
"oversampling_probability"=>0.5,
"metric"=>"dice"
),

"model"=>Dict(
"optimizer_name"=>"Adam",
"optimizer_args"=>"lr=0.001",
"num_epochs"=>10,
"loss_function_name"=>"dice"
)

)

open(config_path,"w") do f
    print(f,JSON.json(config,4))
end

@test isfile(config_path)

rng_seed = 42

result = MedPipe3D.main_loop(pathToHDF5,config_path,rng_seed)

@test result !== nothing

@test CUDA.functional() isa Bool

end