import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
from matplotlib import rcParams
# If use jupyter

TimesSong = FontProperties(fname='/media/happy507/DataSpace1/liuyuxin/times-new-roman.ttf')
font_manager.fontManager.addfont('/media/happy507/DataSpace1/liuyuxin/times-new-roman.ttf')

import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')



# 数据
algorithms = [
    "GCA-Net", "GDN-Net", "FFA-Net", "MSBDN",
    "4KDehazing", "Dehazeformer-T", "MIT-Net", "Ours"
]
entropy = [6.397, 7.045, 7.040, 7.055, 6.980, 6.981, 7.058, 7.112]
niqe = [4.793, 4.576, 3.873, 4.065, 3.972, 3.667, 3.586, 3.483]
brisque = [38.276, 33.433, 31.354, 32.609, 36.870, 31.377, 30.734, 30.684]

x = np.arange(len(algorithms))  # x轴位置

# 创建图形
fig, ax1 = plt.subplots(figsize=(10, 6))

# Entropy和NIQE在左轴
ax1.bar(x - 0.2, entropy, width=0.4, label='Entropy (↑)', color='skyblue', edgecolor='black')
ax1.bar(x + 0.2, niqe, width=0.4, label='NIQE (↓)', color='lightpink', edgecolor='black')
ax1.set_ylabel('Entropy / NIQE', font={'size':12})
ax1.set_xlabel('Algorithms', font={'size':14})
ax1.set_xticks(x)
ax1.set_xticklabels(algorithms, rotation=15, font={'size':12})
ax1.legend(loc='upper left', fontsize=10)

# BRISQUE在右轴
ax2 = ax1.twinx()
ax2.plot(x, brisque, marker='o', color='tomato', label='BRISQUE (↓)', linewidth=2)
ax2.set_ylabel('BRISQUE', font={'size':12})
ax2.legend(loc='upper right', fontsize=10)

# 调整图形
plt.tight_layout()
plt.show()
