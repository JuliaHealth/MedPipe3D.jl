"""
file with structs
"""
module BasicStructs
using Parameters


"""
main struct with configuration and an algorithm function 
"""
@with_kw struct StepDefStruct
    mainFun # main function to Be invoked over data  stored in DataList

end #StepDefStruct



"""
data about single array and its meta data
"""
@with_kw struct DataItem
    patientId::String # used for identification of patient
    voxX ::Float64 # Voxelspacing x 
    voxY ::Float64 # Voxelspacing y 
    voxZ ::Float64 # Voxelspacing z 
    isCt ::Bool # if true we ware dealing with computer tomography image
    isPET::Bool # if true it is PET image
    isMRI::Bool # if true it is MRI image
    isMask::Bool # if true it is mask 
    isAlgoOutput::Bool # if true it is saved output of our algorithm 
    array # 3 dimensional array with our data
end #DataItem


"""
list of data items plus some meta data 
"""
@with_kw struct DataList
    patientId ::String
    listOfData ::Vector<DataItem>
    mvspx ::Float64 # Voxelspacing x 

end #DataList

"""
holding data needed for correct display
"""
@with_kw struct ForDiplayConf
    mvspx ::Float64 # Voxelspacing x 

end #DataItem



end#BasicStructs