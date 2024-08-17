import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load your dataset (replace with your own data)
df = pd.read_csv(r'C:\Users\Rodrigo\OneDrive\Escritorio\UP\IntelligentsAgents\archive\HAM10000_metadata.csv')

# Define your features (X) and target variable (y)
X = df['image_id']  # features
y = df['dx']  # target variable

# Define the outer 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store the results
outer_test_scores = []
outer_test_reports = []
outer_test_matrices = []

# Perform the outer 5-fold cross-validation
train_index, test_index = next(iter(skf.split(X, y)))
X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]

print("Outer split:")
print("Train indices:", train_index)
print("Test indices:", test_index)
print(len(test_index))

# Define the inner 5-fold cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store the inner results
inner_val_scores = []

train_images_id = []
train_dx = []
validation_images_id = []
validation_dx = []
test_images_id = []
test_dx = []

# Perform the inner 5-fold cross-validation
for inner_train_index, inner_val_index in kf.split(X_train, y_train):
    X_inner_train, X_inner_val = X_train.iloc[inner_train_index], X_train.iloc[inner_val_index]
    y_inner_train, y_inner_val = y_train.iloc[inner_train_index], y_train.iloc[inner_val_index]
    
    train_images_id.append(X_inner_train.tolist())
    train_dx.append(y_inner_train.tolist())
    validation_images_id.append(X_inner_val.tolist())
    validation_dx.append(y_inner_val.tolist())
    test_images_id.append(X_test.tolist())
    test_dx.append(y_test.tolist())
   
    print("Inner split:")
    print("Train indices:", inner_train_index)
    print("Validation indices:", inner_val_index)
    print(len(inner_train_index) + len(inner_val_index))

print("train_images_id: ", train_images_id)
print("train_dx:", train_dx)
print("validation_images_id: ", validation_images_id)
print("validation_dx: ", validation_dx)
print("test_images_id: ", test_images_id)
print("test_dx: ", test_dx)

# Save the data
np.save('train_images_id.npy', np.array(train_images_id, dtype=object))
np.save('train_dx.npy', np.array(train_dx, dtype=object))
np.save('validation_images_id.npy', np.array(validation_images_id, dtype=object))
np.save('validation_dx.npy', np.array(validation_dx, dtype=object))
np.save('test_images_id.npy', np.array(test_images_id, dtype=object))
np.save('test_dx.npy', np.array(test_dx, dtype=object))