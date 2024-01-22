import compare_thresh_seg_fun
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
filename='radar_sanfransico_part'
if len(sys.argv) != 2:
    k = 3
    print("No valid category input, default 3")
else:
    k = int(sys.argv[1])
mat1 = h5py.File(f'input_data\\{filename}\\{filename}.mat', 'r')
data1 = mat1.get('input_data')
data1 = np.array(data1)
input_data = np.transpose(data1)
if len(input_data.shape)==2:
    seg_Pv, seg_Pd, X0, data, Para1, Para2, thres_pv, thres_pd = compare_thresh_seg_fun.newthresSAR(input_data,filename)
    if k ==2:
        plt.imshow(((1 - seg_Pv) * 80).astype(np.uint8), cmap='gray')
        plt.axis('off')
        plt.show()
        plt.imshow(((1 - seg_Pd) * 80).astype(np.uint8), cmap='gray')
        plt.axis('off')
        plt.show()
    else:
        plt.imshow(((X0) * 80).astype(np.uint8), cmap='gray')
        plt.axis('off')
        plt.show()
elif len(input_data.shape)==4:
    input_data = input_data['real']
    seg_Pv, seg_Pd, X0, data, Para1, Para2, thres_pv, thres_pd = compare_thresh_seg_fun.newthres(input_data, filename)
    if k == 2:
        plt.imshow(((1 - seg_Pv) * 80).astype(np.uint8), cmap='gray')
        plt.axis('off')
        plt.show()
        plt.imshow((seg_Pd * 80).astype(np.uint8), cmap='gray')
        plt.axis('off')
        plt.show()
    else:
        plt.imshow(((X0) * 80).astype(np.uint8), cmap='gray')
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
elif len(input_data.shape)==3 and input_data.shape[2]==9:
    input_data = input_data['real']
    x=input_data.shape[0]
    y=input_data.shape[1]
    input_data = input_data.reshape(x, y, 3, 3)
    seg_Pv, seg_Pd, X0, data, Para1, Para2, thres_pv, thres_pd = compare_thresh_seg_fun.newthres(input_data, filename)
    if k == 2:
        plt.imshow(((1 - seg_Pv) * 80).astype(np.uint8), cmap='gray')
        plt.axis('off')
        plt.show()
        plt.imshow((seg_Pd * 80).astype(np.uint8), cmap='gray')
        plt.axis('off')
        plt.show()
    else:
        plt.imshow(((X0) * 80).astype(np.uint8), cmap='gray')
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
else:
    print("ERROR!Wrong Input Data!")
    sys.exit()

