import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Define function
def fx(x):
    return 4*np.sin(np.pi*x)*np.cos(6*np.pi*x**2)

# Define average bias-squared
def bias(n, m, x):
	b = 0
	for j in range(1, int(n/m)+1): 			#+1 so it includes int(n/m)
	
		fj = 0
		for k in range((j-1)*m+1, j*m+1): 	#+1 for inclusion
			fj = fj + fx(x[k-1]) 			#-1 for 0 based indexing
		fj = fj/m
		
		for i in range((j-1)*m+1, j*m+1): 	#+1 for inclusion
			b = b + (fj-fx(x[i-1]))**2 		#-1 for 0 based indexing
			
	return b/n

# Initialize variables
n = 256
M = [1, 2, 4, 8, 16, 32]
x = np.linspace(0, 1, n)
f = fx(x)
y = fx(x) + np.random.normal(0, 1, n)

# Calculate 
EmpErr = [None]*len(M)
Bias = [None]*len(M)
Var = [None]*len(M)
TotErr = [None]*len(M)
for i in range(0, len(M)):
	# Calculate fm 
	m = M[i]
	bins = np.linspace(0, n, int(n/m)+1)
	inds = np.linspace(0, n-1, n)
	dig = np.digitize(inds, bins)
	fm = [y[dig == j].mean() for j in range(1, len(bins))]
	inds = np.array(inds, dtype='int')
	FM = [None]*len(inds)
	for j in range(0, len(inds)):
		FM[j] = fm[dig[j]-1]
		
	# Set variables for this m
	err = FM-f;
	sqerr = [e ** 2 for e in err]
	EmpErr[i] = sum(sqerr)/n
	Bias[i] = bias(n, m, x)
	Var[i] = 1/m
	TotErr[i] = Bias[i]+Var[i]

# Plot
plt.plot(M, EmpErr)
plt.plot(M, Bias)
plt.plot(M, Var)
plt.plot(M, TotErr)

plt.xlabel("m")
plt.ylabel('y')
plt.legend(['Emperical Error', 'Average Bias-squared', 'Average Variance', 'Average Error'])
plt.title('Error as a function of m')
plt.show()



