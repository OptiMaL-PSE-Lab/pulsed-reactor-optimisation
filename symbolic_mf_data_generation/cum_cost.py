import sys 
import os 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import * 
import matplotlib.pyplot as plt

def read_cum_cost(path):
	data = read_json(path)['data']
	d_list = []
	for d in data:
		if d['cost'] != 'running':
			d_list.append(d)
	cost = [d['cost'] for d in d_list]
	cum_cost = np.cumsum(cost)
	return cum_cost

path_names = {'MF Exp Design':'symbolic_mf_data_generation/exp_design/data.json','Low Fidelity':'symbolic_mf_data_generation/low/data.json','High Fidelity':'symbolic_mf_data_generation/high_full/data.json'}
l_ls = ['solid','dashed','dotted']
fig,ax=plt.subplots(1,1,figsize=(6,4))
for i in range(len(path_names.keys())):
	key = list(path_names.keys())[i]
	cum_cost = read_cum_cost(path_names[key])/3600
	ax.plot(np.arange(len(cum_cost)),cum_cost,label=key,lw=2,ls=l_ls[i],c='k')
ax.set_ylabel('Cumulative Time (hr)')
ax.set_xlabel('Data Collected')
ax.grid(alpha=0.5)
ax.legend(frameon=False)
# #turn off right and top axis 
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
fig.savefig('symbolic_mf_data_generation/cum_cost.png',dpi=300,bbox_inches='tight')

