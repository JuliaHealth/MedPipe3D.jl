# pathToHDF5="D:\\dataSets\\forMainHDF5\\smallLiverDataSet.hdf5"


# using Revise,CUDA
# includet("D:\\projects\\vsCode\\JuliaMedPipeB\\tests\\includeAll.jl")
# using Main.GaussianPure, Main.HDF5saveUtils,Main.visualizationFromHdf5, Main.distinctColorsSaved
# using MedEye3d
# import MedEye3d.ForDisplayStructs.TextureSpec
# using Distributions
# using Clustering
# using IrrationalConstants
# using ParallelStencil
# # using PythonCall
# # using CondaPkg
# # CondaPkg.add.("monai");
# # CondaPkg.resolve()


# patienGroupName="0"
# z=7# how big is the area from which we collect data to construct probability distributions
# klusterNumb = 5# number of clusters - number of probability distributions we will use


# #******************for display

# #1) open HDF5 file and define additional arrays needed for our algorithm
# fid = openHDF5(pathToHDF5)
# listOfColorUsed= falses(18)

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


# #2) primary display of chosen image and annotating couple points with a liver
# mainScrollDat= loadFromHdf5Prim(fid,patienGroupName,addTextSpecs,listOfColorUsed)
# #3) save manual modifications to HDF5
# saveManualModif(fid,patienGroupName , mainScrollDat)
# #4) filtering out from the manually modified array all set pixels and get constants needed for later evaluation of gaussian PDF

# manualModif= getArrByName("manualModif" ,mainScrollDat)
# image=  getArrByName("image" ,mainScrollDat)
# algoOutput= getArrByName("algoOutput" ,mainScrollDat)

# mainArrSize= size(image)



# #****************** constructing probability distributions



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


# #************************ applying  probability distributions to image 


# ### getting constants from distributions

# """
# calculate log normalization constant from distribution
# """
# function mvnormal_c0(d::AbstractMvNormal)
#     ldcd = logdetcov(d)
#     return - (length(d) * oftype(ldcd, log2π) + ldcd) / 2
# end

# """
# get constants needed for applying probability distributions
#  return vector   1) logConst 2) mu1 3) mu2 4) invcov00 5)invcov01 6)invcov10 7)invcov11 
# """
# function getDistrConstants(exampleDistr)
#     c0= mvnormal_c0(exampleDistr)
#     invCov= inv(exampleDistr.Σ)
#     return [c0,exampleDistr.μ[1],exampleDistr.μ[2],invCov[1,1],invCov[1,2],invCov[2,1],invCov[2,2]  ]
# end#getDistrConstants


# # creating matrix from constants
# allConstants = map(distr-> getDistrConstants(distr)  , chosenDistribs) |>
#                (vectOfvects)-> reduce(hcat, vectOfvects)


# #### defining CUDA kernel

# """
# utility macro to iterate in given range around given voxel
# """
# macro iterAround(ex   )
#     return esc(quote
#         for xAdd in -r:r
#             x= (threadIdx().x+ ((blockIdx().x -1)*CUDA.blockDim_x()))+xAdd
#             if(x>0 && x<=mainArrSize[1])
#                 for yAdd in -r:r
#                     y= (threadIdx().y+ ((blockIdx().y -1)*CUDA.blockDim_y()))+yAdd
#                     if(y>0 && y<=mainArrSize[2])
#                         for zAdd in -r:r
#                             z= (threadIdx().z+ ((blockIdx().z -1)*CUDA.blockDim_z()))+zAdd
#                             if(z>0 && z<=mainArrSize[3])
#                                 if((abs(xAdd)+abs(yAdd)+abs(zAdd)) <=r)
#                                     $ex
#                                 end 
#                             end
#                         end
#                     end    
#                 end    
#             end
#         end    
#     end)
# end
      

# """
# con - matrix of precalculated constants
# image - main image here computer tomography image
# mainArrSize - dimensions of image
# output - where we want to save the calculations
# r - size of the evaluated patch
# klusterNumb- number of clusters - number of probability distributions we will use
# """
# function applyGaussKernel(con,image,mainArrSize,output, r::Int,klusterNumb::Int)
#     for probDist in 1:klusterNumb
#         summ=0.0
#         sumCentered=0.0
#         lenn= UInt8(0)
#         #get mean
#         @iterAround begin 
#             lenn=lenn+1
#             summ+=image[x,y,z]    
#         end
#         summ=summ/lenn
#         #get standard deviation
#         @iterAround sumCentered+= ((image[x,y,z]-summ )^2)

#         #here we have standard deviation
#         sumCentered= sqrt(sumCentered/(lenn-1))
#         #centering - subtracting means...
#         summ=summ-con[2,probDist]
#         sumCentered=sumCentered-con[3,probDist]
#         #saving output
#         x= (threadIdx().x+ ((blockIdx().x -1)*CUDA.blockDim_x()))
#         y= (threadIdx().y+ ((blockIdx().y -1)*CUDA.blockDim_y()))
#         z= (threadIdx().z+ ((blockIdx().z -1)*CUDA.blockDim_z()))
#         if(x>0 && x<=mainArrSize[1] && y>0 && y<=mainArrSize[2] &&z>0 && z<=mainArrSize[3] )
#             output[x,y,z]=  max(exp(con[1,probDist]-( ((summ*con[4,probDist]+sumCentered*con[6,probDist])*summ+(summ*con[5,probDist]+sumCentered*con[7,probDist])*sumCentered)/2 ) ),output[x,y,z]  )
#         end  
#     end#for
#     return
# end#main kernel

# # for simplicity not using the occupancy API - in production one rather should
# threads=(8,4,8)
# blocks = (cld(mainArrSize[1],threads[1]), cld(mainArrSize[2],threads[2])  , cld(mainArrSize[3],threads[3]))
# using CUDA
# algoOutputGPU=CuArray(algoOutput)
# imageGPU=CuArray(image)
# conGPU = CuArray(allConstants)
# @cuda threads=threads blocks=blocks applyGaussKernel(conGPU,imageGPU,mainArrSize,algoOutputGPU, 5,klusterNumb)
# #@cuda threads=threads blocks=blocks applyGaussKernel(conGPU,imageGPU,mainArrSize,algoOutputGPU, z,klusterNumb)
# copyto!(algoOutput,algoOutputGPU)
# sum(algoOutput)# just to check is anythink copied

# #copy and divide by max so will be easier to visualize
# algoOutputB= getArrByName("algoOutput" ,mainScrollDat)
# maxEl = maximum(algoOutputGPU)
# algoOutputB[:,:,:]=algoOutput./maxEl

# ### just to show how slow it would be to achieve the same on CPU

# # ## single thread
# # function getMaxProb(point)
# #     coords= getCartesianAroundPoint(point,z)
# #     xxx=getSampleMeanAndStd( Float64,Float64, coords , image  )
# #     return maximum(map(dist-> Distributions.pdf(dist, xxx),chosenDistribs))
# # end


# # output = map(getMaxProb, CartesianIndices(image))
# # maximum(output)
# # algoOutput[:,:,:]=output./maximum(output)


# # ## multithread
# # cartss = CartesianIndices(image)
# # Threads.@threads for i = 1:length(image)
# #     algoOutput[i] = getMaxProb(cartss[1])
# # end


# saveMaskbyName(fid,patienGroupName , mainScrollDat, "algoOutput")

# #******************************************************** relaxation labelling

# ########5) 
# """
# in case of the relaxation labelling for 2D case algorithm basically can be simplified to
# iteratively look into the surrounding elements and increase value of a voxel given voxels around
# had high enough value and decrese otherwise

# adapted from https://discourse.julialang.org/t/3d-medical-app-stencil/64019/4
# """

# const USE_GPU = true
# using ParallelStencil
# using ParallelStencil.FiniteDifferences3D

# @init_parallel_stencil(CUDA, Float64, 3);






# #cutoff set manually to rate
# @parallel_indices (ix,iy,iz) function relaxationLabellKern(In, rate)
#     # 7-point Neuman stencil
#     if (ix>1 && iy>1 && iz>1 &&      ix<(size(In,1))&& iy<(size(In,2)) && iz<(size(In,3)))
#         In[ix,iy,iz] = ( (In[ix-1,iy  ,iz  ] >rate)+
#                           (In[ix-1,iy  ,iz  ]>rate)+ (In[ix+1,iy  ,iz  ]>rate) +
#                           (In[ix  ,iy-1,iz  ]>rate) + (In[ix  ,iy+1,iz  ]>rate) +
#                           (In[ix  ,iy  ,iz-1]>rate) + (In[ix  ,iy  ,iz+1]>rate) )/7.0

     
#     end
#     return
# end

# @views function relaxLabels(In, iterNumb,rate)
#     # Calculation
#     for i in 1:iterNumb
#         innerRate=rate + ((i/iterNumb)/3)
#         @parallel relaxationLabellKern(In,innerRate)
#     end#for    
#     return
# end

# rate=0.15
# relaxLabels(algoOutputGPU,50,rate)

# copyto!(algoOutput,algoOutputGPU)
# Int(round(sum(algoOutput)))# just to check is anythink copied  #85162

# #copy and divide by max so will be easier to visualize
# algoOutputB= getArrByName("algoOutput" ,mainScrollDat)
# algoOutputB[:,:,:]=algoOutput



# #relaxLabels(algoOutputGPU,10)

# ###########7) displaying performance metrics

# # first we need to define the cutoff  over which we will decide that probability indicates that it is truly a liver 
# #####simple  tresholding
# function tresholdingKernel(mainArrSize,output)
  
#         x= (threadIdx().x+ ((blockIdx().x -1)*CUDA.blockDim_x()))
#         y= (threadIdx().y+ ((blockIdx().y -1)*CUDA.blockDim_y()))
#         z= (threadIdx().z+ ((blockIdx().z -1)*CUDA.blockDim_z()))
#         if(x>0 && x<=mainArrSize[1] && y>0 && y<=mainArrSize[2] &&z>0 && z<=mainArrSize[3] )
#             output[x,y,z]=  (output[x,y,z]>0.5)
#         end  
#     return
# end#main kernel

# # for simplicity not using the occupancy API - in production one rather should
# threads=(8,8,8)
# blocks = (cld(mainArrSize[1],threads[1]), cld(mainArrSize[2],threads[2])  , cld(mainArrSize[3],threads[3]))
# @cuda threads=threads blocks=blocks tresholdingKernel(mainArrSize,algoOutputGPU)



# using MedEval3D
# using MedEval3D.BasicStructs
# using MedEval3D.MainAbstractions
# conf= ConfigurtationStruct(md=true, dice=true)
# numberToLookFor = 1.0
# liverGold= getArrByName("liver" ,mainScrollDat)

# preparedDict=MedEval3D.MainAbstractions.prepareMetrics(conf)
# calculateAndDisplay(preparedDict,mainScrollDat, conf, numberToLookFor,CuArray(liverGold),algoOutputGPU )




# mainScrollDat.mainTextToDisp

# copyto!(algoOutput,algoOutputGPU)
# Int(round(sum(algoOutput)))# just to check is anythink copied  #85162
# #copy and divide by max so will be easier to visualize
# algoOutputB= getArrByName("algoOutput" ,mainScrollDat)
# algoOutputB[:,:,:]=algoOutput














# saveManualModif(fid,patienGroupName , mainScrollDat)

# close(fid)
