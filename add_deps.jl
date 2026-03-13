using Pkg
Pkg.add(["JSON", "Zygote", "CUDA", "KernelAbstractions", "Lux", "LuxCUDA", "Optimisers", "ADTypes", "ComputerVisionMetrics", "Distributions", "Interpolations", "ImageFiltering"])
Pkg.instantiate()
