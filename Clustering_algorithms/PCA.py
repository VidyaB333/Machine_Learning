from matplotlib import pyplot as plt
import sklearn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

x = load_digits().data
print(x.shape)
y = load_digits().target
print(y.shape)

print(type(x))
print(x[0, :])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
#print(x_scaled[0,:])

pca = PCA(n_components=3)

x_reduced = pca.fit_transform(x_scaled)
print(x_reduced[0,:])
print(pca.explained_variance_)
print(pca.explained_variance_ratio_.sum())

variance = []
n_comp = []
pc = 30
for i in range(pc):
    pca = PCA(n_components=i+1)
    x_reduced = pca.fit_transform(x_scaled)
    #print(pca.explained_variance_ratio_)
    n_comp.append(i+1)
    variance.append(pca.explained_variance_ratio_.sum())
    print(i+1 ,pca.explained_variance_ratio_.sum())
plt.plot(n_comp, variance, 'r')
plt.plot(n_comp, variance, 'bs')
plt.grid()
plt.show()
