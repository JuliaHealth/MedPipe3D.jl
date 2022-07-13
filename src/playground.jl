using PythonCall
using CondaPkg
CondaPkg.add("simpleitk")
CondaPkg.add("itk")

CondaPkg.add_pip("simpleitk", version="")

pyimport("simpleitk")
pyimport("seaborn")

pyimport("simpleitk")

1+1


#include("D:\\projects\\vsCode\\MedPipe\\MedPipe\\src\\includeAll.jl")

# using Main.LoadFromMonai, Main.HDF5saveUtils,Main.visualizationFromHdf5, Main.distinctColorsSaved
# using PythonCall
# using MedEye3d
# import MedEye3d.ForDisplayStructs.TextureSpec
# using Distributions
# using Clustering
# using IrrationalConstants
# using ParallelStencil
# using HDF5


# #only for finding paths in folder better to do it with pure Julia
# glob=pyimport("glob")
# os=pyimport("os")

# #representing number that is the patient id in this dataset
# patienGroupName="0"
# z=7# how big is the area from which we collect data to construct probability distributions
# klusterNumb = 5# number of clusters - number of probability distributions we will use


# #directory of folder with files in this directory all of the image files should be in subfolder volumes 0-49 and labels labels if one ill use lines below
# data_dir = "D:\\dataSets\\CTORGmini\\"
# #directory where we want to store our HDF5 that we will use
# pathToHDF5="D:\\dataSets\\forMainHDF5\\smallLiverDataSet.hdf5"

# #loading paths of images in our dataset
# train_images = sort(pyconvert(Array,glob.glob(os.path.join(data_dir, "volumes 0-49", "*.nii.gz"))))
# train_labels = sort(pyconvert(Array,glob.glob(os.path.join(data_dir, "labels", "*.nii.gz"))))


# #zipping so we will have tuples with image and label names
# zipped= collect(zip(train_images,train_labels))
# tupl=zipped[1]
# #proper loading
# loaded = LoadFromMonai.loadByMonaiFromImageAndLabelPaths(tupl[1],tupl[2])
# #now We open the hdf5 and save the array with some required metadata

# #!!!!!!!!!! important if you are just creating the hdf5 file  do it with "w" option otherwise do it with "r+"
# fid = h5open(pathToHDF5, "w")
# #fid = h5open(pathToHDF5, "r+") 
# gr= getGroupOrCreate(fid, patienGroupName)

# #for this particular example we are intrested only in liver so we will keep only this 
# labelArr=map(entry-> UInt8(entry==1),loaded[2])



# #we save loaded and trnsformed data into HDF5 to avoid doing preprocessing every time
# saveMaskBeforeVisualization(fid,patienGroupName,loaded[1],"image", "CT" )
# saveMaskBeforeVisualization(fid,patienGroupName,labelArr,"labelSet", "boolLabel" )

# # here we did default transformations so voxel dimension is set to 1,1,1 in any other case one need to set spacing attribute manually to proper value
# # spacing can be found in metadata dictionary that is third entry in loadByMonaiFromImageAndLabelPaths output
# # here metadata = loaded[3]
# writeGroupAttribute(fid,patienGroupName, "spacing", [1,1,1])

# #******************for display
# #just needed so we will not have 2 same colors for two diffrent informations
# listOfColorUsed= falses(18)

# ##below we define additional arrays that are not present in original data but will be needed for annotations and storing algorithm output 

# #manual Modification array
# manualModif = TextureSpec{UInt8}(# choosing number type manually to reduce memory usage
#     name = "manualModif",
#     color =getSomeColor(listOfColorUsed)# automatically choosing some contrasting color
#     ,minAndMaxValue= UInt8.([0,1]) #important to keep the same number type as chosen at the bagining
#     ,isEditable = true ) # we will be able to manually modify this array in a viewer

# algoVisualization = TextureSpec{Float32}(
#     name = "algoOutput",
#     # we point out that we will supply multiple colors
#     isContinuusMask=true,
#     colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed)]
#     ,minAndMaxValue= Float32.([0,1])# values between 0 and 1 as this represent probabilities
#    )

#     addTextSpecs=Vector{TextureSpec}(undef,2)
#     addTextSpecs[1]=manualModif
#     addTextSpecs[2]=algoVisualization


# #2) primary display of chosen image 
# mainScrollDat= loadFromHdf5Prim(fid,patienGroupName,addTextSpecs,listOfColorUsed)

# # now we can use manualModif array to  create annotations





# #****************** constructing probability distributions

# #manually modify in annotator

# ##coordinates of manually set points
# coordsss= GaussianPure.getCoordinatesOfMarkings(eltype(image),eltype(manualModif),  manualModif, image) |>
#     (seedsCoords) ->GaussianPure.getPatchAroundMarks(seedsCoords,z ) |>
#     (patchCoords) ->GaussianPure.allNeededCoord(patchCoords,z )

# #getting patch statistics - mean and covariance
# patchStats = GaussianPure.calculatePatchStatistics(eltype(image),Float64, coordsss, image)

# #separate distribution for each marked point
# distribs = map(patchStat-> fit(MvNormal, reduce(hcat,(patchStat)))  , patchStats  )

# #in order to reduce computational complexity  we will reduce the number of used distributions using kl divergence

# #we are comparing all distributions 
# klDivs =map(outerDist->    map(dist->kldivergence( outerDist  ,dist), distribs  ), distribs  )
# klDivsInMatrix = reduce(hcat,(klDivs))
# #clustering with kmeans
# R = kmeans(klDivsInMatrix, klusterNumb; maxiter=200, display=:iter)

# #now identify indexes for some example distributions from each cluster
# indicies = zeros(Int64,klusterNumb )
# a = assignments(R) # get the assignments of points to clusters
# for i in 1:klusterNumb
#     for j in 1:length(distribs)
#         if(a[j] == i)
#             indicies[i]=j
#         end
#     end    
# end
# indicies

# #ditributions from diffrent clusters
# chosenDistribs = map(ind->distribs[ind] ,indicies)




















# # imageDataset.attrs['dataType']=  "CT"
# # labelBoolDataset.attrs['dataType']=  "boolLabel"

# # using Revise,CUDA

# # using MedEye3d
# # import MedEye3d.ForDisplayStructs.TextureSpec
# # using Distributions
# # using Clustering
# # using ParallelStencil


# # import MedEye3d
# # import MedEye3d.ForDisplayStructs
# # import MedEye3d.ForDisplayStructs.TextureSpec
# # using ColorTypes
# # import MedEye3d.SegmentationDisplay

# # import MedEye3d.DataStructs.ThreeDimRawDat
# # import MedEye3d.DataStructs.DataToScrollDims
# # import MedEye3d.DataStructs.FullScrollableDat
# # import MedEye3d.ForDisplayStructs.KeyboardStruct
# # import MedEye3d.ForDisplayStructs.MouseStruct
# # import MedEye3d.ForDisplayStructs.ActorWithOpenGlObjects
# # import MedEye3d.OpenGLDisplayUtils
# # import MedEye3d.DisplayWords.textLinesFromStrings
# # import MedEye3d.StructsManag.getThreeDims



# # listOfColorUsed= falses(18)

# # CTIm= loaded[1]
# # labell=loaded[2]

# # #manual Modification array
# # CTImm= TextureSpec{Float32}(
# #     name= "CTIm",
# #     numb= Int32(3),
# #     isMainImage = true,
# #     minAndMaxValue= Float32.([0,100]))  

# # algoVisualization = TextureSpec{Float32}(
# #     name = "algoOutput",
# #     # we point out that we will supply multiple colors
# #     isContinuusMask=true,
# #     colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed)]
# #     ,minAndMaxValue= Float32.([0,1])# values between 0 and 1 as this represent probabilities
# #    )

# #    manualModif=  TextureSpec{UInt8}(
# #     name = "manualModif",
# #     numb= Int32(2),
# #     color = RGB(0.0,1.0,0.0)
# #     ,minAndMaxValue= UInt8.([0,1])
# #     ,isEditable = true
# #    )

# #     addTextSpecs=Vector{TextureSpec}(undef,3)
# #     addTextSpecs[1]=manualModif
# #     addTextSpecs[2]=CTImm
# #     addTextSpecs[3]=algoVisualization




# #     datToScrollDimsB= MedEye3d.ForDisplayStructs.DataToScrollDims(imageSize=  size(CTIm) ,voxelSize=(1,1,1), dimensionToScroll = 3 );


# #     import MedEye3d.DisplayWords.textLinesFromStrings

# #     mainLines= textLinesFromStrings(["main Line1", "main Line 2"]);
# #     supplLines=map(x->  textLinesFromStrings(["sub  Line 1 in $(x)", "sub  Line 2 in $(x)"]), 1:size(CTIm)[3] );
    
# #     import MedEye3d.StructsManag.getThreeDims
    
# #     tupleVect = [("algoOutput",labell) ,("CTIm",CTIm),("manualModif",zeros(UInt8,size(CTIm)) ) ]
# #     slicesDat= getThreeDims(tupleVect )


# #     fractionOfMainIm= Float32(0.8);


# #     mainScrollDat = FullScrollableDat(dataToScrollDims =datToScrollDimsB
# #     ,dimensionToScroll=1 # what is the dimension of plane we will look into at the beginning for example transverse, coronal ...
# #     ,dataToScroll= slicesDat
# #     ,mainTextToDisp= mainLines
# #     ,sliceTextToDisp=supplLines );





# #     SegmentationDisplay.coordinateDisplay(addTextSpecs ,fractionOfMainIm ,datToScrollDimsB ,1000);


# #     Main.SegmentationDisplay.passDataForScrolling(mainScrollDat);



