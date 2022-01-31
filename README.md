# mPCA_detect
binary cancer detection based on resnet50. 

input: openslide compatible WSI

output: qupath json, probability map



# usage: python detection_inference_cluster.py 

[required] users should pass one of the following arguments:

    --by_csv "/path/to/file.csv" containing full filepaths to WSI (see example)
    
    --by_folder "/path/to/folder" folder path to WSI files, does not search recursively
    
    --by_image "/path/to/image.tif" full filepath for openslide compatible WSI


[optional] 

    --save_location "/path/to/save/outputs" will default to "./output"
    


# creating environment
