__precompile__(false)

module MedPipe3D

include("dependencies.jl")

# Utilities (uses HDF5, Statistics, JSON — loaded inside the included files)
include("utils/utils.jl")

# Configuration
include("Configuration/configuration.jl")

# Data loading / preprocessing into HDF5
include("Data_loading_to_HDF5_with_pre-processing/batch_main.jl")

# Get data (batches)
include("batch_loader.jl")

# Training components
include("Train_Validation_Test/get_loss_function.jl")
include("Train_Validation_Test/get_optimiser.jl")
include("Train_Validation_Test/model.jl")
include("Train_Validation_Test/splits.jl")
include("Train_Validation_Test/validation.jl")
include("Train_Validation_Test/train.jl")
include("Train_Validation_Test/test.jl")
include("Train_Validation_Test/main_loop.jl")


# Augmentations
include("Augmentations/augmentation.jl")
include("Augmentations/apply.jl")

# Morphology operations
include("Morphology_operations/dilation.jl")
include("Morphology_operations/erosion.jl")

# Post-processing
include("Post-processing/post-processing.jl")

# Batching / channeling
include("Batching_channeling/batching_channeling.jl")

export main_loop, batch_main, print_hdf5_contents

end # module MedPipe3D
