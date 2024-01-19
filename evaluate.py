import compare_thresh_seg_fun
import h5py
import numpy as np
import sys
filename='radar_sanfransico_part'
mat1 = h5py.File(f'input_data\\{filename}\\{filename}.mat', 'r')
mat2 = h5py.File(f'input_data\\{filename}\\wishart_seg.mat','r')
X1=compare_thresh_seg_fun.afterprocess(mat2)
data1 = mat1.get('input_data')
data1 = np.array(data1)
input_data = np.transpose(data1)
if len(input_data.shape)==2:
    img1, img2 = compare_thresh_seg_fun.preprocessSAR(input_data)
    # 新阈值分割
    seg_Pv, seg_Pd, X3, data, Para1, Para2, thres_pv, thres_pd = compare_thresh_seg_fun.newthresSAR(input_data, filename)
elif len(input_data.shape)==4:
    img1, img2 = compare_thresh_seg_fun.preprocess(input_data)
    input_data = input_data['real']
    # 新阈值分割
    seg_Pv, seg_Pd, X3, data, Para1, Para2, thres_pv, thres_pd = compare_thresh_seg_fun.newthres(input_data, filename)
elif len(input_data.shape)==3 and input_data.shape[2]==9:
    x=input_data.shape[0]
    y=input_data.shape[1]
    input_data = input_data.reshape(x, y, 3, 3)
    input_data = input_data['real']
    # 新阈值分割
    seg_Pv, seg_Pd, X3, data, Para1, Para2, thres_pv, thres_pd = compare_thresh_seg_fun.newthres(input_data, filename)
    img1, img2 = compare_thresh_seg_fun.preprocess(input_data)
else:
    print("ERROR!Wrong Input Data!")
    sys.exit()
index=[]
# OTSU
X0 = compare_thresh_seg_fun.otsu(img1, img2)
acc1, total_acc1, iou1, miou1 = compare_thresh_seg_fun.metrics_all(X0, X1)
index.append([acc1[0],acc1[1],acc1[2], total_acc1, iou1[0],iou1[1],iou1[2], miou1])
# 最大熵
X0=compare_thresh_seg_fun.entropy(img1,img2)
acc1, total_acc1, iou1, miou1 = compare_thresh_seg_fun.metrics_all(X0, X1)
index.append([acc1[0],acc1[1],acc1[2], total_acc1, iou1[0],iou1[1],iou1[2], miou1])
# 迭代法
X0=compare_thresh_seg_fun.iterative(img1,img2)
acc1, total_acc1, iou1, miou1 = compare_thresh_seg_fun.metrics_all(X0, X1)
index.append([acc1[0],acc1[1],acc1[2], total_acc1, iou1[0],iou1[1],iou1[2], miou1])
# 矩方法
X0=compare_thresh_seg_fun.moment(img1,img2)
acc1, total_acc1, iou1, miou1 = compare_thresh_seg_fun.metrics_all(X0, X1)
index.append([acc1[0],acc1[1],acc1[2], total_acc1, iou1[0],iou1[1],iou1[2], miou1])
# EM
X0=compare_thresh_seg_fun.EM(img1,img2)
acc1, total_acc1, iou1, miou1 = compare_thresh_seg_fun.metrics_all(X0, X1)
index.append([acc1[0],acc1[1],acc1[2], total_acc1, iou1[0],iou1[1],iou1[2], miou1])
#新阈值分割
acc1, total_acc1, iou1, miou1 = compare_thresh_seg_fun.metrics_all(X3, X1)
index.append([acc1[0],acc1[1],acc1[2], total_acc1, iou1[0],iou1[1],iou1[2], miou1])
print(index)


