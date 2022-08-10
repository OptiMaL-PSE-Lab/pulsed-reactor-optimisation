import numpy as np 
import matplotlib.pyplot as plt 
import pickle
plt.rcParams["font.family"] = "cursive"

def plot_fidelities(M):
	f1 = [0,0.25,0.5,0.75,1]
	f2 = [0,0.25,0.5,0.75,1]
	names = ['Max Aspect Ratio','Max Mesh Non-Orthogonality','Average Mesh Non-Orthogonality','Max Skewness','Time to create mesh (s)']
	fig,axs = plt.subplots(1,1,figsize=(4,3))
	i = 3 # TIME 
	plt.tight_layout()
	plt.subplots_adjust(top=0.95,bottom=0.15,right=0.85,left=0.0)
	Mp = M[i]
	print(Mp)
	im = axs.imshow(Mp)
	plt.colorbar(im,ax=axs,fraction=0.046, pad=0.04,label=names[i])
	axs.set_ylabel('Axial Fidelity')
	axs.set_xlabel('Radial Fidelity')
	axs.set_xticks(np.arange(len(f1)),(f1))
	axs.set_yticks(np.arange(len(f2)),np.flip(f2))
	fig.savefig('multi_fidelity/qualities.pdf')
	return 

# with open('multi_fidelity/fidelity_qualities.pickle', 'rb') as handle:
#         M = pickle.load(handle)
# plot_fidelities(M)