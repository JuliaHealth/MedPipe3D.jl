using Revise
includet("D:\\projects\\vsCode\\MedPipe\\MedPipe\\src\\includeAll.jl")

using Main.LoadFromMonai, Main.HDF5saveUtils,Main.visualizationFromHdf5, Main.distinctColorsSaved
using PythonCall
#only for finding paths in folder better to do it with pure Julia
glob=pyimport("glob")
os=pyimport("os")

#representing number that is the patient id in this dataset
patienGroupName="0"
z=7# how big is the area from which we collect data to construct probability distributions
klusterNumb = 5# number of clusters - number of probability distributions we will use


#directory of folder with files in this directory all of the image files should be in subfolder volumes 0-49 and labels labels if one ill use lines below
data_dir = "D:\\dataSets\\CTORGmini\\"
train_images = sort(pyconvert(Array,glob.glob(os.path.join(data_dir, "volumes 0-49", "*.nii.gz"))))
train_labels = sort(pyconvert(Array,glob.glob(os.path.join(data_dir, "labels", "*.nii.gz"))))
#zipping so we will have tuples with image and label names
zipped= collect(zip(train_images,train_labels))
tupl=zipped[1]
#proper loading
loaded = LoadFromMonai.loadByMonaiFromImageAndLabelPaths(tupl[1],tupl[2])
#now 






using Revise,CUDA

using MedEye3d
import MedEye3d.ForDisplayStructs.TextureSpec
using Distributions
using Clustering
using ParallelStencil


import MedEye3d
import MedEye3d.ForDisplayStructs
import MedEye3d.ForDisplayStructs.TextureSpec
using ColorTypes
import MedEye3d.SegmentationDisplay

import MedEye3d.DataStructs.ThreeDimRawDat
import MedEye3d.DataStructs.DataToScrollDims
import MedEye3d.DataStructs.FullScrollableDat
import MedEye3d.ForDisplayStructs.KeyboardStruct
import MedEye3d.ForDisplayStructs.MouseStruct
import MedEye3d.ForDisplayStructs.ActorWithOpenGlObjects
import MedEye3d.OpenGLDisplayUtils
import MedEye3d.DisplayWords.textLinesFromStrings
import MedEye3d.StructsManag.getThreeDims



listOfColorUsed= falses(18)

CTIm= loaded[1]
labell=loaded[2]

#manual Modification array
CTImm= TextureSpec{Float32}(
    name= "CTIm",
    numb= Int32(3),
    isMainImage = true,
    minAndMaxValue= Float32.([0,100]))  

algoVisualization = TextureSpec{Float32}(
    name = "algoOutput",
    # we point out that we will supply multiple colors
    isContinuusMask=true,
    colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed)]
    ,minAndMaxValue= Float32.([0,1])# values between 0 and 1 as this represent probabilities
   )

   manualModif=  TextureSpec{UInt8}(
    name = "manualModif",
    numb= Int32(2),
    color = RGB(0.0,1.0,0.0)
    ,minAndMaxValue= UInt8.([0,1])
    ,isEditable = true
   )

    addTextSpecs=Vector{TextureSpec}(undef,3)
    addTextSpecs[1]=manualModif
    addTextSpecs[2]=CTImm
    addTextSpecs[3]=algoVisualization




    datToScrollDimsB= MedEye3d.ForDisplayStructs.DataToScrollDims(imageSize=  size(CTIm) ,voxelSize=(1,1,1), dimensionToScroll = 3 );


    import MedEye3d.DisplayWords.textLinesFromStrings

    mainLines= textLinesFromStrings(["main Line1", "main Line 2"]);
    supplLines=map(x->  textLinesFromStrings(["sub  Line 1 in $(x)", "sub  Line 2 in $(x)"]), 1:size(CTIm)[3] );
    
    import MedEye3d.StructsManag.getThreeDims
    
    tupleVect = [("algoOutput",labell) ,("CTIm",CTIm),("manualModif",zeros(UInt8,size(CTIm)) ) ]
    slicesDat= getThreeDims(tupleVect )


    fractionOfMainIm= Float32(0.8);


    mainScrollDat = FullScrollableDat(dataToScrollDims =datToScrollDimsB
    ,dimensionToScroll=1 # what is the dimension of plane we will look into at the beginning for example transverse, coronal ...
    ,dataToScroll= slicesDat
    ,mainTextToDisp= mainLines
    ,sliceTextToDisp=supplLines );





    SegmentationDisplay.coordinateDisplay(addTextSpecs ,fractionOfMainIm ,datToScrollDimsB ,1000);


    Main.SegmentationDisplay.passDataForScrolling(mainScrollDat);



