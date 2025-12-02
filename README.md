# Thresholding of Polarimetric SAR Images of Coastal Zones Based on Three-Component Decomposition and Likelihood Ratio
**Python implementation of a new threshold segmentation method for polarimetric SAR images based on three-component decomposition and likelihood ratio.**<br>

Created by Chun Liu, Sutong Wang, and Jian Yang

This method utilizes volume scattering power and double-bounce scattering power sampling to obtain the optimal threshold based on likelihood ratio.

The experimental results in the paper were obtained from __Matlab code__. The Python version code is publicly available here, and the Matlab code will also be released after it is sorted out.




## News
- **2024-01-18** Our proposed threshold segmentation method outperforms traditional threshold segmentation methods (mPA>0.8 & mIoU>0.6) on multiple datasets.
- **2024-01-18** The paper has been submitted.
- **2024-06-12** The paper has been published on IEEE Xplore.

## Usage

### Requirements

- h5py 
- matplotlib
- scipy 
- numpy 
- pandas
- sklearn

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
@ARTICLE{10543127,
  author={Liu, Chun and Wang, Sutong and Yang, Jian},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Thresholding of Polarimetric SAR Images of Coastal Zones Based on Three-Component Decomposition and Likelihood Ratio}, 
  year={2024},
  volume={21},
  number={},
  pages={1-5},
  keywords={Scattering;Image segmentation;Histograms;Sea measurements;Radar polarimetry;Land surface;Entropy;Coastal zone;likelihood ratio;polarimetric SAR;three-component decomposition;thresholding},
  doi={10.1109/LGRS.2024.3407848}
  }

```
