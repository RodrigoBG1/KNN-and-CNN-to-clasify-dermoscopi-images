import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

train_images_id = np.load('train_images_id.npy', allow_pickle=True)
train_dx = np.load('train_dx.npy', allow_pickle=True)
validation_images_id = np.load('validation_images_id.npy', allow_pickle=True)
validation_dx = np.load('validation_dx.npy', allow_pickle=True)
test_images_id = np.load('test_images_id.npy', allow_pickle=True)
test_dx = np.load('test_dx.npy', allow_pickle=True)
"""print(train_images_id)
print(train_dx)"""

max_accuracy = 0

# Función para cargar y preparar las imágenes
def train_images_and_labels(images_id, images_dx):
    images = []
    labels = []
    for j, img_id in enumerate(images_id):
        if img_id != -1:
            # Cargar la imagen
            img_path = "OneDrive/Documentos/Verano2024/IntelligentAgents/archive/all_images/{}.jpg".format(img_id)
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            label = images_dx[j]
            images.append(image)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Codificar imagenes y etiquetas de validacion y prueba
def val_and_test_img_and_lab(images_id, images_dx):
    images = []
    labels = []
    for j, img_id in enumerate(images_id):
        if img_id != -1:
            # Cargar la imagen
            img_path = "OneDrive/Documentos/Verano2024/IntelligentAgents/archive/all_images/{}.jpg".format(img_id)
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            label = images_dx[j]
            images.append(image)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)

    # Codificar las etiquetas
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    labels = to_categorical(labels, num_classes=num_classes)

    return images, labels

for i in range(len(train_images_id)):
    # Cargar imágenes y etiquetas
    train_images, train_labels = train_images_and_labels(train_images_id[i], train_dx[i])

    """print(images)
    print(labels)"""

    # Codificar las etiquetas
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    num_classes = len(label_encoder.classes_)
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    lab_classes = label_encoder.classes_

    # Construir el modelo ResNet50
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in resnet.layers:
        layer.trainable = False

    model = Sequential()
    model.add(resnet)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compilar el modelo
    model.compile(optimizer=Adam(),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    validation_images, validation_labels = val_and_test_img_and_lab(validation_images_id[i], validation_dx[i])

    # Entrenar el modelo
    model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(validation_images, validation_labels))

    test_images, test_labels = val_and_test_img_and_lab(test_images_id, test_dx)

    """# Evaluar el modelo con datos de prueba
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f'Loss: {loss}, Accuracy: {accuracy}')"""

    # Predicción de nuevas imágenes
    predictions = model.predict(test_images)
    predicted_classes = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    """print("Predicciones:")
    for image, predicted_class in zip(test_images, predicted_classes):
        print(f"Imagen: {image}, Predicción: {predicted_class}")"""

    accuracy = accuracy_score(test_dx, predicted_classes)
    print(f'Accuracy: {accuracy}')
    f1 = f1_score(test_dx, predicted_classes, average='weighted')
    print(f'F1 score: {f1}')

    if accuracy > max_accuracy:
        max_accuracy = accuracy
        best_pred = predicted_classes

cm = confusion_matrix(test_dx, best_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=lab_classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Normalized Confusion Matrix")
plt.show()