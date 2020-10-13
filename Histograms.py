import numpy as np
import matplotlib.pyplot as plt

cup_depth = [
    # fasciculatum (black)
    [6.12, 5.73, 4, 5, 5, 4.5, 5, 5, 4, 5.75, 4.1, 5.18, 4.08, 4.7, 5.7, 6.13, 5.99, 6.85, 4.57, 4.75, 4.87,
     3.87, 6.42, 4.58, 6.07, 4.27, 4.81, 4.06, 6, 4.08, 4.95, 4.39, 5.44, 4.34, 6, 5, 6, 2.3, 4.8, 5.24, 4,
     5.97, 4.97, 5.96, 4.38, 4.51, 3.86, 4.73, 4.74, 4.62, 4.14, 3.49, 5.07, 4.16, 4.42, 4.5, 5.2, 4.25,
     4.03, 3.38, 3.8],
    # franciscanum (grey)
    [3.41, 3.72, 1.32, 2.76, 4.77, 3.89, 4.72, 3.17, 2.95, 2.22, 3.86, 3.57, 4.69, 2.02, 4.85, 5, 3.25, 3,
     3.5, 3, 4, 4.16, 3.89, 3.88, 4.05, 3.51, 3.81, 3.94, 4.29, 4.63, 2.91, 4.45, 2.5, 3.35, 3.76, 3.88, 4.19, 3.47]
]
cup_lobe_ratio = [
    # fasciculatum (black)
    [1.176923077, 1.503937008, 1.6, 1.428571429, 2.5, 1.8, 1.428571429, 1.25, 1.142857143, 1.642857143, 1.076115486,
     1.392473118, 1.240121581, 0.909090909, 1.170431211, 1.422273782, 1.21010101, 1.683046683, 1.187012987, 0.901328273,
     1.337912088, 0.969924812, 1.569682152, 1.339181287, 1.576623377, 1.636015326, 2.176470588, 1.022670025,
     1.550387597, 1.545454545, 1.518404908, 1.821576763, 1.187772926, 1.98173516, 1.090909091, 1, 1.2, 0.625,
     0.661157025, 1.706840391, 1, 1.658333333, 1.496987952, 1.122410546, 1.331306991, 1.334319527, 1.335640138,
     1.150851582, 1.640138408, 1.235294118, 1.604651163, 1.057575758, 2.06097561, 1.02970297, 1.572953737, 1.535836177,
     1.163310962, 1.261127596, 1.439285714, 0.988304094, 1.144578313],
    # franciscanum (grey)
    [1.1, 1.488, 0.647058824, 0.865203762, 1.204545455, 0.864444444, 1.322128852, 0.854447439, 0.888554217, 0.860465116,
     0.781376518, 1.136942675, 1.116666667, 0.721428571, 1.023206751, 1.428571429, 0.8125, 0.75, 1.166666667, 1.2,
     1.333333333, 0.827037773, 1.262987013, 1.137829912, 1.033163265, 1.121405751, 0.886046512, 0.987468672,
     1.175342466, 1.268493151, 1.168674699, 1.009070295, 0.632911392, 0.817073171, 1.175, 1.021052632, 1.396666667,
     0.788636364]
]
plant_length = [
    # fasciculatum (black)
    [12.89, 14.01, 3.9, 5.3, 3, 3.5, 3.25, 10, 5.25, 3.5, 15.77, 14.39, 15.29, 13.84, 9.99, 21, 12.53,
     14.72, 14.97, 13.63, 14.23, 10.18, 16.22, 12.82, 10.3, 8.45, 17.76, 15.45, 12.71, 9.8, 9.65, 7.9,
     13.97, 8.8, 9.6, 9.9, 7.5, 10.31, 10.48, 12.08, 12.27, 12.67, 12.84, 20.01, 13.84, 14.34, 11.79,
     9.62, 17.45, 11.22, 13.75, 7.53, 12.66, 19.5, 18.18, 15.3, 19, 8.97, 15.17, 12.14, 13.36,
     ],
    # franciscanum (grey)
    [10.49, 15.75, 15.58, 14.34, 13.3, 18.5, 15.12, 10.93, 10.16, 8.77, 15.36, 10.75, 16.86, 8.44, 16.69, 9, 5.2, 9.5,
     6.2, 5.2, 5.5, 15.24, 10.27, 20.07, 12.09, 10.51, 12.37, 16.86, 11.26, 16.72, 7.29, 12.44, 12.26, 11.49, 18.54,
     12.85, 11.49, 15.63]
]

angle = [
    # fasciculatum (Black)
    [102.4, 119.71, 128.66, 137, 145.18, 95.16, 127.7, 111.33, 110, 138, 128.86, 127.53, 125.73, 116.57, 136, 124.32,
     137.64, 102.93, 126, 130.7, 112.7, 156.1, 120, 128, 123, 133.5, 141.8, 120.1, 128.86, 119.15, 124.379, 81, 112,
     131, 125.9, 130.25, 134.8, 110.7, 157.8, 101, 139, 131, 126.16, 138.93, 141.79, 146.65, 112.34, 125.83, 136.2,
     134.32, 124.19, 120, 134.71, 119, 138, 136, 128, 138, 135, 128.4, 138.16],
    # franciscanum (grey)
    [135, 140, 138.3, 138, 143, 125.8, 153, 134.8, 155.97, 140, 140.59, 131, 133.7, 143.61, 177.9, 138, 152, 144, 144,
     155, 137, 131, 137, 155, 143, 175.64, 133, 133, 157.7, 124.5, 146.09, 154, 142, 154, 153, 163.5, 170.7, 147.92]
]

fig, axes = plt.subplots(nrows=3, ncols=1)
csfont = {'fontname':'Times New Roman'}
ax0, ax1, ax2 = axes.flatten()


colors = ['black', '0.6']

# Cup Depth (mm)
x0 = np.arange(1.3, 6.8, 0.7)
y0 = np.arange(0, 22, 2)
ax0.hist(cup_depth, bins=x0, density=False, histtype='bar', color=colors, label=colors)
ax0.set_xlabel('Calyx Cup Depth (mm)', csfont, fontsize=7)
ax0.set_ylabel('Frequency', csfont, fontsize=7)
ax0.tick_params(axis='both', labelsize=3, width=1)
ax0.set_xticks(x0)
ax0.set_yticks(y0)


# Cup Depth:Lobe Length Ratio
x1 = np.arange(0.625, 2.5, 0.25)
y1 = np.arange(0, 22, 2)
ax1.hist(cup_lobe_ratio, bins=x1, density=False, histtype='bar', color=colors, label=colors)
ax1.set_xlabel('Ratio Calyx Cup:Lobe', csfont, fontsize=7)
ax1.set_ylabel('Frequency', csfont, fontsize=7)
ax1.tick_params(axis='both', labelsize=3, width=1)
ax1.set_xticks(x1)
ax1.set_yticks(y1)


# angle
x2 = np.arange(81, 178, 12)
y2 = np.arange(0, 26, 4)
ax2.hist(angle, bins=8, density=False, histtype='bar', color=colors, label=colors)
ax2.set_xlabel('Corolla Tube Angle (Degrees)', csfont, fontsize=7)
ax2.set_ylabel('Frequency', csfont, fontsize=7)
ax2.tick_params(axis='both', labelsize=3, width=1)
ax2.set_xticks(x2)
ax2.set_yticks(y2)

fig.set_size_inches(2.25, 4.25)

#ax1.get_figure().savefig('/Users/benbenton/Desktop/Aphyllon 6-28-2020/Hist_New.svg')
fig.tight_layout()
plt.show()
