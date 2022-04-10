# MedPipe
Package that enable working on GPU accelerated medical image segmentation using Julia. The tech stack includes :


MedEye3d.jl - OpenGL based tool for viewing and annotation of 3D medical imagiing


MedEval3D.jl - CUDA accelerated package with 3D medical image segmentation algorithms


HDF5.jl - Julia interface to HDF5 file system which is proven to give higher performance than native medical imagiing formats


MONAI - Python package called in Julia using PythonCall - used for preprocessing

Tutorial can be found on https://github.com/jakubMitura14/MedPipe3DTutorial/tree/master


If You will find usefull my work please cite it

```
@Article{Mitura2021,
  author   = {Mitura, Jakub and Chrapko, Beata E.},
  journal  = {Zeszyty Naukowe WWSI},
  title    = {{3D Medical Segmentation Visualization in Julia with MedEye3d}},
  year     = {2021},
  number   = {25},
  pages    = {57--67},
  volume   = {15},
  doi      = {10.26348/znwwsi.25.57},
  keywords = {OpenGl, Computer Tomagraphy, PET/CT, medical image annotation, medical image visualization},
}

```
