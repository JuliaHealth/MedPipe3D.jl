
using HDF5
using Random

export process_hdf5, batch_indices, get_hdf5_groups_and_attributes, read_attributes, has_split_attributes, split_by_attributes, stratified_split, group_indices, split_by_config


"""
process_hdf5 Function:

Checks for split attributes and calls the appropriate split function.

get_hdf5_groups_and_attributes Function:

Retrieves groups and their attributes from the HDF5 file.

has_split_attributes Function:

Checks for the presence of "is_training", "is_validation", and "is_test" attributes.

split_by_attributes Function:

Splits groups based on the "is_training", "is_validation", and "is_test" attributes.

split_by_config Function:

Handles splitting based on configuration, including shuffling, test/validation set portions, and n-fold cross-validation.

"""



"""
    process_hdf5(db_path::String, config::Configuration, rng::AbstractRNG) -> Dict

Process the HDF5 file and split the groups based on the configuration or attributes.

# Arguments
- `db_path::String`: Path to the HDF5 database.
- `config::Configuration`: Configuration struct containing parameters for data splitting.
- `rng::AbstractRNG`: Random number generator for shuffling.

# Returns
- `Dict`: A dictionary with keys "training", "validation", and "test" containing batched indices of the respective sets.

# Description
This function processes the HDF5 file to extract groups and attributes. It then splits the groups based on the provided configuration or attributes. The resulting indices are grouped into sublists of length `config.batch_size`. If the indices cannot be divided equally and `config.drop_last` is true, the last too-small sublist is ignored.
"""
function process_hdf5(db_path::String, config::Configuration, rng::AbstractRNG) :: Dict
    groups, attributes = get_hdf5_groups_and_attributes(db_path)
    split_dict = has_split_attributes(attributes) ? split_by_attributes(groups, attributes,rng) : split_by_config(groups, attributes, config, rng)

    list_keys=collect(keys(split_dict))
    remove!("test,"list_keys) #remove test from keys
    split_dict["test"]=batch_indices(split_dict["test"], config.batch_size, config.drop_last)
    for key in keys(split_dict)
        split_dict["train"] = batch_indices(split_dict[key], config.batch_size, config.drop_last)
        split_dict["val"] = batch_indices(split_dict[key], config.batch_size, config.drop_last)
    end
    
    return split_dict
end



"""
    batch_indices(indices::Vector{Int}, batch_size::Int, drop_last::Bool) -> Vector{Vector{Int}}

Group indices into sublists of a specified batch size.

# Arguments
- `indices::Vector{Int}`: Vector of indices to be batched.
- `batch_size::Int`: Size of each batch.
- `drop_last::Bool`: If true, drop the last sublist if it is smaller than `batch_size`.

# Returns
- `Vector{Vector{Int}}`: A vector of sublists, each containing indices of length `batch_size`. If `drop_last` is true, the last sublist is dropped if it is smaller than `batch_size`.

# Description
This function takes a vector of indices and groups them into sublists of the specified batch size. If the indices cannot be divided equally and `drop_last` is true, the last too-small sublist is ignored.
"""
function batch_indices(indices::Vector{Int}, batch_size::Int, drop_last::Bool)::Vector{Vector{Int}}
    batched_indices = [indices[i:min(i+batch_size-1, end)] for i in 1:batch_size:length(indices)]
    if drop_last && length(batched_indices[end]) < batch_size
        pop!(batched_indices)
    end
    return batched_indices
end

"""
    get_hdf5_groups_and_attributes(db_path::String) -> Tuple{Vector{String}, Dict{String, Any}}

Get all groups from the HDF5 database and their attributes.

# Arguments
- `db_path::String`: Path to the HDF5 database.

# Returns
- `Tuple{Vector{String}, Dict{String, Any}}`: A tuple containing a vector of group names and a dictionary of attributes.
"""
function get_hdf5_groups_and_attributes(db_path::String) :: Tuple{Vector{String}, Dict{String, Any}}
    groups = String[]
    attributes = Dict{String, Any}()
    h5open(db_path, "r") do file
        for group in keys(file)
            push!(groups, group)
            attributes[group] = read_attributes(file[group])
        end
    end
    return groups, attributes
end

"""
    read_attributes(group) -> Dict{String, Any}

Read all attributes from the given HDF5 group and return them as a dictionary.

# Arguments
- `group`: The HDF5 group from which to read attributes.

# Returns
- `Dict{String, Any}`: A dictionary where keys are attribute names and values are attribute values.
"""
function read_attributes(group) :: Dict{String, Any}
    attrs = Dict{String, Any}()
    for attr in attributes(group)
        attrs[attr] = read(group, attr)
    end
    return attrs
end

"""
    has_split_attributes(attributes::Dict{String, Any}) -> Bool

Check if the attributes contain "is_training", "is_validation", and "is_test".

# Arguments
- `attributes::Dict{String, Any}`: Dictionary of attributes.

# Returns
- `Bool`: True if all split attributes are present, false otherwise.
"""
function has_split_attributes(attributes::Dict{String, Any}) :: Bool
    return all(attr in attributes for attr in ["is_training", "is_validation", "is_test"])
end

"""
    split_by_attributes(groups::Vector{String}, attributes::Dict{String, Any}) -> Dict

Split the groups based on the "is_training", "is_validation", and "is_test" attributes.

# Arguments
- `groups::Vector{String}`: Vector of group names.
- `attributes::Dict{String, Any}`: Dictionary of attributes.

# Returns
- `Dict`: A dictionary with keys "training", "validation", and "test" containing indices of the respective sets.
"""
function split_by_attributes(groups::Vector{String}, attributes::Dict{String, Any}) :: Dict
    split_dict = Dict("training" => String[], "validation" => String[], "test" => String[])
    for group in groups
        if attributes[group]["is_training"]
            push!(split_dict["training"], group)
        elseif attributes[group]["is_validation"]
            push!(split_dict["validation"], group)
        elseif attributes[group]["is_test"]
            push!(split_dict["test"], group)
        end
    end
    return Dict("0" => split_dict)
end

"""
    stratified_split(indices::Vector{Int}, classes::Vector{String}, proportion::Float64) -> Tuple{Vector{Int}, Vector{Int}}

Split indices into two groups with approximately similar class compositions based on the given proportion.

# Arguments
- `indices::Vector{String}`: Vector of indices to be split.
- `classes::Vector{String}`: Vector of class labels corresponding to the indices.
- `proportion::Float64`: Proportion of indices to be included in the first group.

# Returns
- `Tuple{Vector{Int}, Vector{Int}}`: Two vectors of indices representing the two groups.
"""
function stratified_split(indices::Vector{String}, classes::Vector{String}, proportion::Float64,rng)

    # Group indices by class
    class_indices = group_indices(classes)
    
    # Initialize the two groups
    group1 = Int[]
    group2 = Int[]
    
    # For each class, split the indices according to the proportion
    for (cls, idxs) in class_indices
        n = length(idxs)
        n_group1 = Int(round(proportion * n))
        
        # Shuffle indices to ensure randomness
        shuffled_idxs = shuffle(idxs,rng)
        
        # Split indices into two groups
        append!(group1, shuffled_idxs[1:n_group1])
        append!(group2, shuffled_idxs[n_group1+1:end])
    end
    
    return group1, group2
end

"""
    group_indices(classes::Vector{String}) -> Dict{String, Vector{Int}}

Group indices by their corresponding class labels.

# Arguments
- `classes::Vector{String}`: Vector of class labels.

# Returns
- `Dict{String, Vector{Int}}`: Dictionary where keys are class labels and values are vectors of indices.
"""
function group_indices(classes::Vector{String})
    dict = Dict{String, Vector{Int}}()
    for (idx, elem) in enumerate(classes)
        if !haskey(dict, elem)
            dict[elem] = [idx]
        else
            push!(dict[elem], idx)
        end
    end
    return dict
end



"""
    split_by_config(groups::Vector{String}, attributes::Dict{String, Any}, config::Configuration, rng::AbstractRNG) -> Dict

Split the groups based on the configuration.

# Arguments
- `groups::Vector{String}`: Vector of group names.
- `attributes::Dict{String, Any}`: Dictionary of attributes.
- `config::Configuration`: Configuration struct containing parameters for data splitting.
- `rng::AbstractRNG`: Random number generator for shuffling.

# Returns
- `Dict`: A dictionary with keys "training", "validation", and "test" containing indices of the respective sets.

# Description
This function splits the given groups into training, validation, and test sets based on the provided configuration. If shuffling is enabled in the configuration, the groups are shuffled using the provided random number generator. The function also handles n-fold cross-validation if specified in the configuration. If class attributes are present in the attributes dictionary, a stratified split is performed to ensure that each fold has a similar class distribution.
"""
function split_by_config(groups::Vector{String}, attributes::Dict{String, Any}, config::Configuration, rng::AbstractRNG) :: Dict
    # Initialize the split dictionary
    split_dict = Dict("training" => String[], "validation" => String[], "test" => String[])
    
    # Shuffle the groups if required
    if config.is_shuffle
        Random.shuffle!(groups, rng)
    end
    
    # Calculate the number of groups for each set
    n = length(groups)
    n_test = Int(floor(config.test_set_portion * n))
    n_val = Int(floor(config.val_set_portion * n))
    n_train = n - n_test - n_val
    
    # Split the groups
    # Tutaj powinno być sprawdzanie class 
    split_dict["test"] = groups[1:n_test]
    split_dict["validation"] = groups[n_test+1:n_test+n_val]
    split_dict["training"] = groups[n_test+n_val+1:end]
    
    # Handle n-fold cross-validation if specified
    if config.n_fold_cross_val > 1
        if "class" in keys(attributes[groups[1]])
            # Extract class labels for stratified split
            class_labels = [attributes[group]["class"] for group in groups]
            
            # Perform stratified n-fold cross-validation
            # Jeżeli nie ma class to tak jak u góry, do zmodyfikowania 42 min
            folds = stratified_split(groups, class_labels, config.n_fold_cross_val, rng)
            
            for i in 1:config.n_fold_cross_val
                fold_groups = folds[i]
                split_dict["fold_$i"] = Dict("training" => fold_groups, "validation" => String[], "test" => String[])
            end
        end
    end
    
    return split_dict
end

