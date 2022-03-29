# MedPipe
Package that enable working on GPU accelerated medical image segmentation using Julia. The tech stack includes :


MedEye3d.jl - OpenGL based tool for viewing and annotation of 3D medical imagiing


MedEval3D.jl - CUDA accelerated package with 3D medical image segmentation algorithms


HDF5.jl - Julia interface to HDF5 file system which is proven to give higher performance than native medical imagiing formats


MONAI - Python package called in Julia using PythonCall - used for preprocessing

