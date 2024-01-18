# Thresholding of Polarimetric SAR Images of Coastal Zones Based on Three-Component Decomposition and Likelihood Ratio

Created by Chun Liu, Sutong Wang, and Jian Yang


此资源库包含极化SAR图像的一种新的阈值分割方法（已被 XXX 接收）。
This repository contains a new threshold segmentation method for polarimetric SAR images ：__Thresholding of Polarimetric SAR Images of Coastal Zones Based on Three-Component Decomposition and Likelihood Ratio__ (accepted by XXX).


极化SAR海岸带图像在进行阈值分割时容易因为地物强度分布不满足双峰分布而出现错误分割问题。本文提出一种基于三成分分解的极化SAR海岸带图像自动阈值分割方法。利用低散射体散射功率低和强散射二次散射功率高的特性，提出方法分别使用体散射功率和二次散射功率确定高低阈值来实现分割。首先使用Freeman三成分分解提取地物的体散射功率和二次散射功率成分；然后利用提取的样本筛选参数进行典型样本窗口区域筛选；最后基于样本区域平均功率，确定阈值搜索区间，搜索似然比最大的阈值点。分割体散射功率确定低散射区域，分割二次散射功率确定强散射区域，以此实现海岸带图像的两区域或三区域分割。AirSAR旧金山，RADARSAT-2新加坡、博鳌和大连，以及TerraSAR-X新加坡数据实验结果表明，提出方法性能远远优于传统阈值分割方法，能避免复杂海岸场景的错误分割，平均分割精度高达80%，平均交叠率高于0.6。
This paper proposes a method for automatic threshold segmentation of polarimetric SAR coastal zone images based on three-component decomposition. The main objective of the method is to address the issue of wrong segmentation in coastal zone images when the intensity distribution of land does not satisfy a bimodal distribution. The suggested technique utilizes the volume scattering power and double-bounce scattering power to estimate high and low thresholds for segmentation, taking advantage of the properties of low volume scattering power and high double-bounce scattering power. The method follows Freeman three-component decomposition to extract the volume scattering power and double-bounce scattering power components of the feature. These extracted components are the sample screening parameters, which then used to screen the sample window region, and the average power of the sample region is utilized to determine the threshold search interval. The threshold point with the largest likelihood ratio is then searched within this interval. In order to achieve two-region or three-region segmentation of the coastal zone image, the volume scattering power is split to identify the low scattering region, and the double-bounce scattering power is separated to identify the strong scattering region.The experimental findings of AirSAR San Francisco, RADARSAT-2 Singapore, Boao, and Dalian, as well as TerraSAR-X Singapore data, demonstrate that the proposed method outperforms the traditional threshold segmentation method. It effectively prevents the wrong segmentation of intricate coastal scenes, showcasing an average segmentation precision of 80% and an average intersection over union rate exceeding 0.6.



## News
- **2023-12-21** 我们提出的阈值分割方法在多个数据集上的表现优于传统的阈值分割方法（mPA>0.8 & mIoU>0.6）。
- **2023-12-21** Our proposed threshold segmentation method outperforms traditional threshold segmentation methods (mPA>0.8 & mIoU>0.6) on multiple datasets.


## Usage

### Requirements

- h5py 
- matplotlib
- scipy 
- numpy 
- pandas

### Dataset

基于AirSAR旧金山，RADARSAT-2新加坡、博鳌和大连，以及TerraSAR-X新加坡数据进行实验。在这里我们提供AirSAR旧金山数据集。
Experiments based on AirSAR San Francisco, RADARSAT-2 Singapore, Boao and Dalian, and TerraSAR-X Singapore data.Here we provide the AirSAR San Francisco dataset.

### Parameter Setting
你可以在config.py里设置参数。
You can set the parameters in config.py.

### Evaluation
得到我们所提出的方法的分割结果，运行：
Get the segmentation results of our proposed method, run:

```
python main.py
```

得到我们所提出的方法和其他方法的评价指标，运行：
Get the evaluation metrics for our proposed method and other methods, run: 
```
python evaluate.py
```



## Citation
如果您发现我们的工作对您的研究有用，请考虑引用： 
If you find our work useful in your research, please consider citing: 
```

```
