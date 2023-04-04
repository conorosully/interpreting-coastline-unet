# interpreting-unet
Interpreting a U-Net used for coastal water body segmentation using permutation importance

This repository contains the code required to reproduce the results in the conference paper:

> To update

This code is only for academic and research purposes. Please cite the above paper if you intend to use whole/part of the code. 

## Data Files

We have used the following dataset in our analysis: 

1. Sentinel-2 Water Edges Dataset (SWED) from [UK Hydrographic Office](https://openmldata.ukho.gov.uk/#:~:text=The%20Sentinel%2D2%20Water%20Edges,required%20for%20the%20segmentation%20mask.).

 The data is available under the Geospatial Commission Data Exploration license.

## Code Files
You can find the following files in the src folder:

- `train_unet.py` Code used to train semantic segmentation model
- `analysis_of_indices.ipynb` The main analysis file used to calculate permutation importance and create all figures in the research paper.
- `utils.py` Helper file containing functions used to perform the analysis in the main analysis file. 

## Result Files
You can find the following files used in the analysis:

- `UNET-SCALE-13MAR23.pth` Final U-Net model whihc is interpreted in this analysis
- `Metrics_28MAR23.csv` Contains the permutation importance for different spectral bands and combinations of bands
