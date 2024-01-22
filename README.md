# Thresholding of Polarimetric SAR Images of Coastal Zones Based on Three-Component Decomposition and Likelihood Ratio
**Python implementation of a new threshold segmentation method for polarimetric SAR images based on three-component decomposition and likelihood ratio.**<br>

Created by Chun Liu, Sutong Wang, and Jian Yang

The method utilizes volume scattering power and double-bounce scattering power to obtain high and low thresholds for segmentation by taking advantage of low volume scattering power and high double-bounce scattering power. A sample window region is filtered over the volume scattering and double-bounce scattering power and a threshold search interval is determined based on the average power of the sample region. Then, the optimal threshold is obtained based on the likelihood ratio.


The experimental results in the paper were obtained from __Matlab code__. The Python version code is publicly available here, and the Matlab code will also be released after it is sorted out.




## News
- **2024-01-18** Our proposed threshold segmentation method outperforms traditional threshold segmentation methods (mPA>0.8 & mIoU>0.6) on multiple datasets.
- **2024-01-18** The paper has been submitted.

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

### Segmentation result Getting
Get the segmentation results of our proposed method, run:

```
python main.py <category_count>
```
You can change <category_count> to the number of categories you want to segment, 2 or 3 is recommended. If you don't enter this parameter，the default is 3.
### Evaluation
Get the evaluation metrics for our proposed method and other methods, run: 
```
python evaluate.py
```



## Citation 
If you find our work useful in your research, please consider citing: 
```

```
