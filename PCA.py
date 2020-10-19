import pandas as pd
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math

########################################################################################
# Adapted from https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
########################################################################################

url = "https://github.com/Benbenton01/Aphyllon/blob/main/PCA_raw_data.csv"
# load dataset into Pandas DataFrame

df = pd.read_csv(url, names=['Angle', 'Cup Depth', 'Lobe Length', 'Cup:Lobe', 'Flower Number', 'Flower Length',
                             'Plant Length', 'Pedicel Length', 'target', 'name'])

features = ['Angle', 'Cup Depth', 'Lobe Length', 'Cup:Lobe', 'Flower Number', 'Flower Length', 'Plant Length',
            'Pedicel Length']

# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:, ['target']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
txt = df.loc[:, ['name']].values

pca = PCA(n_components=3)

principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component1', 'principal component2',
                                                              'principal component3'])

finalDf = pd.concat([principalDf, df['target']], axis=1)
names = []
finalDf['name'] = txt
# print (finalDf)

import matplotlib.pyplot as plt

print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Components: \n{pca.components_}")
print(f"Explained Variance: {pca.explained_variance_}")

# Figure Parameters
fig = plt.figure(figsize=(7, 4.5))
plt.rcParams["font.family"] = "Times New Roman"
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel(f'PC1 ({round((pca.explained_variance_ratio_[0])*100, 1)}%)', fontsize=7)
ax.set_ylabel(f'PC2 ({round((pca.explained_variance_ratio_[1])*100, 1)}%)', fontsize=7)
targets = (['A. fasciculatum ', 'A. franciscanum'])
colors = ['black', '0.6']
markers = ['s', 'o']
size = 40

for target, color, marker in zip(targets, colors, markers):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component1'],
               finalDf.loc[indicesToKeep, 'principal component2'],
               c=color,
               s=5,
               marker=marker)

for i, point in finalDf.iterrows():
    if point['name'] == 'RSA0006071':
        ax.text(point['principal component1'], point['principal component2'], point['name'])

fig.set_size_inches(4.625, 3)
ax.legend(targets, loc=0, prop={'size': 9})
ax.tick_params(axis='both', labelsize=7)
plt.show()
