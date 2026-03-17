using Test
using LinearAlgebra
using JLD2
include("../src/dependencies.jl")
include("../src/utils/model_utils.jl")
include("../src/Configuration/configuration.jl")
include("../src/utils/img_utils.jl")
include("../src/utils/parsing_utils.jl")
include("../src/utils/io_utils.jl")
include("../src/Data_loading_to_HDF5_with_pre-processing/batch_main.jl")
include("../src/batch_loader.jl")
include("../src/Train_Validation_Test/validation.jl")
include("../src/Train_Validation_Test/test.jl")
include("../src/Train_Validation_Test/train.jl")
include("../src/Train_Validation_Test/get_loss_function.jl")
include("../src/Train_Validation_Test/get_optimiser.jl")
include("../src/Train_Validation_Test/model.jl")
include("../src/Train_Validation_Test/splits.jl")
include("../src/Train_Validation_Test/main_loop.jl")


@testset "MedPipe3D Tests" begin

	# include("test_dataset_to_hdf5.jl")
	# include("test_hdf5_to_nifti.jl")
	# include("test_training.jl")
	# include("utils/test_parsing_utils.jl")
	# include("utils/test_img_utils.jl")
	# include("utils/test_io_utils.jl")
	# include("utils/test_model_utils.jl")
	# include("test_config.jl")
	# include("test_batch_loader.jl")
	include("test_integration.jl")
end
