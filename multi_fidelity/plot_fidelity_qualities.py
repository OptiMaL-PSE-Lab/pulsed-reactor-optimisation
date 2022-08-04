import numpy as np 
import matplotlib.pyplot as plt 
import pickle
plt.rcParams["font.family"] = "cursive"

def plot_fidelities(M):
	f1 = np.linspace(0,1,10)
	f2 = [0,0.25,0.5,0.75,1]
	M = np.flip(M,axis=0)
	names = ['Max Aspect Ratio','Max Mesh Non-Orthogonality','Average Mesh Non-Orthogonality','Max Skewness']
	fig,axs = plt.subplots(2,2,figsize=(8,4))
	plt.subplots_adjust(wspace=0.15,left=0.08,right=0.99,bottom=0.12,top=0.925,hspace=0.6)
	for i, ax in zip(range(len(M)), axs.ravel()):
		im = ax.imshow(M[i].T)
		ax.set_title(names[i])
		plt.colorbar(im,ax=ax)
		ax.set_ylabel('Axial Fidelity')
		ax.set_xlabel('Radial Fidelity')
		ax.set_xticks(np.arange(10),list(np.arange(11)/10)[1:])
		ax.set_yticks(np.arange(len(f2)),np.flip(f2))
	fig.savefig('multi_fidelity/qualities.pdf')
	return 

