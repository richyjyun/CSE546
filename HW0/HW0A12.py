import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

n=40000 # From problem A.6

#(a)
Z=np.random.randn(n)
plt.step(sorted(Z), np.arange(1,n+1)/float(n))

#(b)
k = [1,8,64,512]
for x in k:
	Y = np.sum(np.sign(np.random.randn(n, x))*np.sqrt(1./x), axis=1)
	plt.step(sorted(Y), np.arange(1,n+1)/float(n))

plt.xlim(-3,3)
plt.ylim(0,1)
plt.xlabel("Observations")
plt.ylabel("Probability")
plt.legend(['Gaussian','1','8','64','512'])	
plt.show()