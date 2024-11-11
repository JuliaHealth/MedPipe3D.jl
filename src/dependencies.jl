using HDF5
using JSON
using Statistics
using Zygote
using CUDA
using KernelAbstractions
using Lux, LuxCUDA, Lux.Training
using Lux.Training: single_train_step!, TrainState
using Random,Optimisers
using ADTypes
using ComputerVisionMetrics
CUDA.allowscalar(true)