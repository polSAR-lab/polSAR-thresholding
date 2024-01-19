# Thresholding of Polarimetric SAR Images of Coastal Zones Based on Three-Component Decomposition and Likelihood Ratio

Created by Chun Liu, Sutong Wang, and Jian Yang


This repository contains a new threshold segmentation method for polarimetric SAR images ：__Thresholding of Polarimetric SAR Images of Coastal Zones Based on Three-Component Decomposition and Likelihood Ratio__.

__The paper of the proposed method has been submitted.__

The experimental results in the paper were obtained from __Matlab code__. The Python version code is publicly available here, and the Matlab code will also be released after it is sorted out.

This paper proposes a method for automatic threshold segmentation of polarimetric SAR coastal zone images based on three-component decomposition. The main objective of the method is to address the issue of wrong segmentation in coastal zone images when the intensity distribution of land does not satisfy a bimodal distribution. The suggested technique utilizes the volume scattering power and double-bounce scattering power to estimate high and low thresholds for segmentation, taking advantage of the properties of low volume scattering power and high double-bounce scattering power. The method follows Freeman three-component decomposition to extract the volume scattering power and double-bounce scattering power components of the feature. These extracted components are the sample screening parameters, which then used to screen the sample window region, and the average power of the sample region is utilized to determine the threshold search interval. The threshold point with the largest likelihood ratio is then searched within this interval. In order to achieve two-region or three-region segmentation of the coastal zone image, the volume scattering power is split to identify the low scattering region, and the double-bounce scattering power is separated to identify the strong scattering region.The experimental findings of AirSAR San Francisco, RADARSAT-2 Singapore, Boao, and Dalian, as well as TerraSAR-X Singapore data, demonstrate that the proposed method outperforms the traditional threshold segmentation method. It effectively prevents the wrong segmentation of intricate coastal scenes, showcasing an average segmentation precision of 80% and an average intersection over union rate exceeding 0.6.



## News
- **2024-01-18** Our proposed threshold segmentation method outperforms traditional threshold segmentation methods (mPA>0.8 & mIoU>0.6) on multiple datasets.


## Usage

### Requirements

- h5py 
- matplotlib
- scipy 
- numpy 
- pandas

### Dataset

Experiments based on AirSAR San Francisco, RADARSAT-2 Singapore, Boao and Dalian, and TerraSAR-X Singapore data.
Here we provide the AirSAR San Francisco dataset and RADARSAT-2 San Francisco dataset for your convenience in running our code.

### Parameter Setting
You can set the parameters in config.py.

### Evaluation
Get the segmentation results of our proposed method, run:

```
python main.py
```
You can enter the number of categories you want to segment, here we suggest you enter 2 or 3. If you choose not to enter, then the default is to segment 3 categories.

Get the evaluation metrics for our proposed method and other methods, run: 
```
python evaluate.py
```



## Citation 
If you find our work useful in your research, please consider citing: 
```

```
