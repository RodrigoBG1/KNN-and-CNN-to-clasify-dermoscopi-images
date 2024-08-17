from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv(r'C:\Users\Rodrigo\OneDrive\Escritorio\UP\IntelligentsAgents\archive\HAM10000_metadata.csv')
image_id = df.iloc[:, 1].tolist()  # Convert to list
pca = np.load('pca_2d.npy')
# Create a dictionary with image_id as key and PCA values as value
dic = {id: pca[i] for i, id in enumerate(image_id)}

# Load the cross-validation data
train_images_id = np.load('train_images_id.npy', allow_pickle=True)
train_dx = np.load('train_dx.npy', allow_pickle=True)
validation_images_id = np.load('validation_images_id.npy', allow_pickle=True)
validation_dx = np.load('validation_dx.npy', allow_pickle=True)
test_images_id = np.load('test_images_id.npy', allow_pickle=True)
test_dx = np.load('test_dx.npy', allow_pickle=True)

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_percentages = np.round(cm_percentages/100, 2)  # Convert to percentage and round to 2 decimal places
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentages, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

best_accuracy = 0
best_f1 = 0
best_y_test = None
best_y_pred_test = None
best_labels = None

# Iterate over the 5 splits
for i in range(5):
    # Prepare training data
    X_train = np.array([dic[img] for img in train_images_id[i] if img != -1])
    y_train = np.array([dx for dx in train_dx[i] if dx != -1])
    
    # Prepare validation data
    X_val = np.array([dic[img] for img in validation_images_id[i] if img != -1])
    y_val = np.array([dx for dx in validation_dx[i] if dx != -1])
    
    # Create and train the model
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)
    
    # Make predictions on validation set
    y_pred_val = knn.predict(X_val)
    
    # Evaluate on validation set
    print(f"Validation (Split {i+1}):")
    accuracy_val = accuracy_score(y_val, y_pred_val)
    f1_val = f1_score(y_val, y_pred_val, average='weighted')
    print(f'Accuracy: {accuracy_val}')
    print(f'F1 Score: {f1_val}')
    
    # Prepare test data
    X_test = np.array([dic[img] for img in test_images_id[i]])
    y_test = np.array(test_dx[i])
    
    # Make predictions on test set
    y_pred_test = knn.predict(X_test)
    
    # Evaluate on test set
    print(f"Test (Split {i+1}):")
    accuracy_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    print(f'Accuracy: {accuracy_test}')
    print(f'F1 Score: {f1_test}')
    
    # Get unique labels
    labels = np.unique(np.concatenate((y_test, y_pred_test)))
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred_test, labels=labels)
    #print(f"Confusion Matrix (Split {i+1}):\n{cm}\n")

    # Update best accuracy and confusion matrix
    if accuracy_test > best_accuracy or (accuracy_test == best_accuracy and f1_test > best_f1):
        best_accuracy = accuracy_test
        best_f1 = f1_test
        best_y_test = y_test
        best_y_pred_test = y_pred_test
        best_labels = labels

print(f"Best Accuracy: {best_accuracy}")
print(f"Best F1 Score: {best_f1}")
plot_confusion_matrix(best_y_test, best_y_pred_test, best_labels)