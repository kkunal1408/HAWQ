import numpy as np
import matplotlib.pyplot as plt

dist0=np.array(open("results/distribution_0.txt").read().replace('[','').replace(']','').split(', '),dtype=np.float32)
dist1=np.array(open("results/distribution_1.txt").read().replace('[','').replace(']','').split(', '),dtype=np.float32)
dist2=np.array(open("results/distribution_2.txt").read().replace('[','').replace(']','').split(', '),dtype=np.float32)
dist4=np.array(open("results/distribution_4.txt").read().replace('[','').replace(']','').split(', '),dtype=np.float32)
#print(dist0)
#norm = np.linalg.norm(sum_distribution)
#sum_distribution = sum_distribution/np.mean(sum_distribution)
def plot_histogram(dist):
    x=np.arange(dist.size)
    plt.figure()
    #plt.plot(x, dist )
    plt.hist(dist, bins=10)
    #ax.set(xlim=(1, 9))
    #ax.set_yscale('log')
    #ax.grid(linestyle='--', linewidth=0.5)
    #ax.set_xlabel('Operand bitwidth (B)')
    #ax.set_ylabel('Energy/MAC(fJ)')
    #ax.legend()
    plt.title('output_distribution')
    #plt.savefig('results/dist1.png')
    #plt.show()
    print(f"mean: {np.mean(dist)} variance: {np.var(dist)}")
    scaled_dist = dist/np.mean(dist)
    print(f"mean: {np.mean(scaled_dist)} variance: {np.var(scaled_dist)}")
for dist in [dist4, dist0/np.mean(dist0)-dist2/np.mean(dist2)]:
	plot_histogram(dist)
plt.show()

#plt.show()
