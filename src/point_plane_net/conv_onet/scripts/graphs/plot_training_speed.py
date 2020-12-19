import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['text.usetex'] = True
import numpy as np
import pdb

keyword = 'Validation metric (iou)'

def get_iou_list(fname):
    out = []
    for line in open(fname):
        if keyword in line:
          out.append(float(line.split(': ')[-1]))

    return out



iter = list(range(1, 35, 1))
n = len(iter)
iou_onet = get_iou_list('con_graph/chair_onet.log')
iou_pointconv = get_iou_list('con_graph/chair_pointconv.log')
iou_our_grid = get_iou_list('con_graph/chair_grid32.log')
iou_our_1plane = get_iou_list('con_graph/chair_1plane.log')
iou_our_3plane = get_iou_list('con_graph/chair_3plane.log')

fig, ax = plt.subplots()
plt.plot(iter, iou_onet[:n], label='ONet', linewidth=4)
plt.plot(iter, iou_pointconv[:n], label='PointConv', linewidth=4)
plt.plot(iter, iou_our_1plane[:n], label='Ours-2D ($64^2$)', linewidth=4)
plt.plot(iter, iou_our_3plane[:n], label='Ours-2D '+r'($3 \times 64^2$)', linewidth=4)
plt.plot(iter, iou_our_grid[:n], label='Ours-3D ($32^3$)', linewidth=4)
plt.xlabel('Training Iterations '+r'$(\times 10K)$', fontsize=24)
plt.ylabel('Validation IoU', fontsize=24)
legend = ax.legend(fontsize=18)
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.set_xticks(iter)
# ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

# axis spacing
ax.set_xticks(ax.get_xticks()[::5])

ax.tick_params(axis='both', which='major', labelsize=24)
ax.tick_params(axis='both', which='minor', labelsize=24)

# set graph size
fig.set_size_inches(10, 6)

# add grid
plt.grid()
fig.savefig('tmp.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()
