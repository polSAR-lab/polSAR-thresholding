import compare_thresh_seg_fun
import h5py
import matplotlib.pyplot as plt
import numpy as np
filename='Radsat_sanfransico_part.mat'
k=int(input("请输入要分割的类别: "))
mat1 = h5py.File(f'D:\\code_seg\\input_data\\radar_sanfransico_part\\{filename}', 'r')
if k == 3:
    X0,data,Para1,Para2,thres_pv,thres_pd=compare_thresh_seg_fun.newthres(mat1,filename)
if k == 2:
    X0, data, Para1, Para2=compare_thresh_seg_fun.newthres2(mat1,filename)
plt.imshow(((1-X0) * 80).astype(np.uint8), cmap='gray')
plt.axis('off')
plt.show()
bin_width = 0.01
plt.hist(data.ravel(), bins=np.arange(0, 1 + bin_width, bin_width), density=True, color='r', alpha=0.7, label='Span')
plt.hist(Para2.ravel(), bins=np.arange(0, 1 + bin_width, bin_width), density=True, color='b', alpha=0.7, label='Pd')
plt.hist(Para1.ravel(), bins=np.arange(0, 1 + bin_width, bin_width), density=True, color='g', alpha=0.7, label='Pv')
plt.legend()
plt.axis([0, 1, 0, 10])
plt.xlabel('intensity', fontsize=15)
plt.ylabel('density', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


