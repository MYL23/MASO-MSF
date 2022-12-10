# MASO-MSF
## Environments
PyTorch v1.8.0  
Python 3.7  
## Data Downloads
Download [GeoLife](https://www.microsoft.com/en-us/download/details.aspx?id=52367) datasets 

## MASO Structure
The folder contains the code related to constructing the Multi-Attribute-Scale-Object (MASO) structure based on the GeoLife dataset, which consists of four main processing files:  
(1) MatchTrajectoryLabel: match trajectories with the corresponding transportation modes.  
(2) SplitSegment: split the trajectory to segments.  
(3) CropProject: segment cropping and projection. 
(4) ComputeAttributes: local motion attributes computation. 

## MSF Model
The folder contains the code related to the Multi-Stage Fusion Model (MSF), which consists of three major files:
(1) DataLoader: samples input and data augmentation.  
(2) MainModel: the main structure of the MSF model.  
(3) Train: model training.    
(4) Test: model evaluation.  
(5) PrintMetricInformation: accuracy, precision, recall, confusion matrix.   
