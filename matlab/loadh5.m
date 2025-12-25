function [ outputStructure ] = loadh5( fileToLoad )
%LOADH5 Loads all datasets in an HDF5 file into a structure
%   Loads all datasets in an HDF5 file into a matlab structure. The
%   built in h5info and h5read matlab functions are used to load the data.
%   Complex matrices are converted from a structure containing separate
%   real and imag fields into a complex matrix.
%
%        datasetsInFile = loadh5('filename.h5');
%

    info = h5info(fileToLoad);
    
    outputStructure = LoadGroup(fileToLoad, info);
end

function [outputStructure] = LoadGroup(fileToLoad, groupToLoad)

    outputStructure = struct;

    % Load each group in this group
    for groupIndex=1:size(groupToLoad.Groups)        
        currentGroup = groupToLoad.Groups(groupIndex);
        
        splitString = strsplit(groupToLoad.Groups(groupIndex).Name, '/');
        currentGroupName = char(splitString(size(splitString,2)));
        
        if(FieldNameIsValid(currentGroupName))
            outputStructure.(currentGroupName) = struct;
            outputStructure.(currentGroupName) = LoadGroup(fileToLoad, currentGroup);
        end
    end

    % Load datasets in this group
    %k = findstr('Slice0001', groupToLoad.Name);
    %loadDatasetsForThisGroup = (k > 0);
    if(1)%loadDatasetsForThisGroup)
        for datasetIndex=1:size(groupToLoad.Datasets)
            fieldName = [groupToLoad.Name '/' groupToLoad.Datasets(datasetIndex).Name];

            % Discard fields that with invalid characters in the field names.
            % For processing archives, these are usually text or binary files 
            % contained within the processing archive
            if(FieldNameIsValid(fieldName))
                dataset = h5read(fileToLoad, fieldName);

                % If the dataset contains complex data, convert it from a structure
                % containing separate real and imaginary data matrices to a single
                % matrix containing complex data
                if(isstruct(dataset))
                    fields = fieldnames(dataset);
                    isComplex = length(fields) == 2 && strcmp(fields{1}, 'real') && strcmp(fields{2}, 'imag');
                    if(isComplex)
                       dataset =  double(dataset.real) + 1i .* double(dataset.imag);
                    end
                end

                outputStructure.(groupToLoad.Datasets(datasetIndex).Name) = dataset;        
            end        
        end
    end
    
end

function [isValid] = FieldNameIsValid(fieldName)
    fieldNameDoesNotContainDot = size(strfind(fieldName, '.'),1) == 0;
    fieldNameDoesNotContainDash = size(strfind(fieldName, '-'),1) == 0;
    fieldNameStartsWithNonNumeric = size(str2num(fieldName(1)),1) == 0;
    isValid = fieldNameDoesNotContainDot && fieldNameDoesNotContainDash && fieldNameStartsWithNonNumeric;
end