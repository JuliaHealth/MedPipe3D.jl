using Test
include("../src/utils/model_utils.jl")
include("../src/Configuration/configuration.jl")
include("../src/utils/img_utils.jl")
include("../src/utils/parsing_utils.jl")
include("../src/utils/io_utils.jl")
include("../src/Data_loading_to_HDF5_with_pre-processing/batch_main.jl")
include("../src/batch_loader.jl")

@testset "MedPipe3D Tests" begin

	include("test_dataset_to_hdf5.jl")
	include("test_hdf5_to_nifti.jl")
	include("test_training.jl")
	include("utils/test_parsing_utils.jl")
	include("utils/test_img_utils.jl")
	include("utils/test_io_utils.jl")
	include("utils/test_model_utils.jl")
	include("test_config.jl")
	include("test_batch_loader.jl")
end
