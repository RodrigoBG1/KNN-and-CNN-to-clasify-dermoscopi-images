import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

#Load the Excel data
df = pd.read_csv(r'C:\Users\Rodrigo\OneDrive\Escritorio\UP\IntelligentsAgents\archive\HAM10000_metadata.csv')

# Load the image data
histograms_array = np.load('histogram_data.npy')

"""--------------------PCA 2D-----------------------"""

# Apply PCA to the histograms array
pca = PCA(n_components=2)  
pca_histograms = pca.fit_transform(histograms_array)
print("Original shape:", histograms_array.shape)
print("PCA shape:", pca_histograms.shape)

# Get the unique diagnosis labels and their corresponding colors
dx_labels = df['dx'].unique()
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']

# Create a dictionary to map diagnosis labels to colors
dx_colors = dict(zip(dx_labels, colors))

# dx_color[bkl]

# Plot the samples in 2D using PC
plt.figure(figsize=(8, 8))

# Save the information in a numpy file
np.save('pca_2d.npy', pca_histograms)

for i, label in enumerate(dx_labels):
    idx = df['dx'] == label
    pca_samples = pca_histograms[idx]
    plt.scatter(pca_samples[:, 0], pca_samples[:, 1], c=[dx_colors[label]] * len(pca_samples), label=label)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('HAM10000 Dataset (2D PCA)')
plt.legend()
plt.show()


"""--------------------PCA 3D-----------------------"""
# Apply PCA to the histograms array
pca = PCA(n_components=3)  # retain 3 components for 3D plot
pca_histograms = pca.fit_transform(histograms_array)
print("Original shape:", histograms_array.shape)
print("PCA shape:", pca_histograms.shape)

# Get the unique diagnosis labels and their corresponding colors
dx_labels = df['dx'].unique()
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']

# Create a dictionary to map diagnosis labels to colors
dx_colors = dict(zip(dx_labels, colors))

# Plot the samples in 3D using PCA
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

for i, label in enumerate(dx_labels):
    idx = df['dx'] == label
    pca_samples = pca_histograms[idx]
    ax.scatter(pca_samples[:, 0], pca_samples[:, 1], pca_samples[:, 2], c=[dx_colors[label]] * len(pca_samples), label=label)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('HAM10000 Dataset (3D PCA)')
ax.legend()
plt.show()
plt.show()

x = histograms_array
skf = StratifiedKFold(n_splits=3)
skf.get_n_splits(x)

# Save the 3D PCA data to a numpy file
np.save('pca_3d.npy', pca_histograms)