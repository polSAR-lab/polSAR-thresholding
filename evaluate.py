import compare_thresh_seg_fun
import h5py

filename='Radsat_sanfransico_part.mat'
mat1 = h5py.File(f'D:\\code_seg\\input_data\\radar_sanfransico_part\\{filename}', 'r')
mat2 = h5py.File(f'D:\\code_seg\\input_data\\radar_sanfransico_part\\wishart_seg.mat','r')
X1=compare_thresh_seg_fun.afterprocess(mat2)
img1, img2 = compare_thresh_seg_fun.preprocess(mat1)
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
X0,data,Para1,Para2,thres_pv,thres_pd=compare_thresh_seg_fun.newthres(mat1,filename)
acc1, total_acc1, iou1, miou1 = compare_thresh_seg_fun.metrics_all(X0, X1)
index.append([acc1[0],acc1[1],acc1[2], total_acc1, iou1[0],iou1[1],iou1[2], miou1])
print(index)


