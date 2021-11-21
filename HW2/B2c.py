import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
x = np.linspace(0, 4, 100)
y = (2 - np.sqrt(x))**2

x = np.sort(np.append(x, -x[1:len(x)]))
y = np.append(np.flip(y[1:len(y)]), y)

plt.fill_between(x, y, -y)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()