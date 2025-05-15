import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from PIL import ImageFile
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import EfficientNetB7


# Uszkodzone pliki
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Ścieżki
train_dir = "E:/sieci_neuronowe/Images_dataset/train"
test_dir = "E:/sieci_neuronowe/Images_dataset/test"

# Augmentacja danych do treningu
train_datagen = ImageDataGenerator(
    # skalowanie pikseli
    rescale=1./255,
    # losowy obrót
    rotation_range=20,
    # przesunięcie
    width_shift_range=0.2,
    height_shift_range=0.2,
    # ścięcie obrazu
    shear_range=0.2,
    # powiększenie
    zoom_range=0.2,
    # obrót
    horizontal_flip=True,
    # podział na walidacje i trening
    validation_split=0.2
)

# skalowanie bez augmentacji danych testowych
test_datagen = ImageDataGenerator(rescale=1./255)

# Wczytanie obrazów z folderu
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    # tryb klasyfikacji
    class_mode='binary',
    # użycie danych treningowych
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Ładowanie danych testowych
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Model z wytrenowanymi wagami
base_model = EfficientNetB7(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    # dodanie warstw
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True,
    verbose=1
)

# Trening modelu
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Dokładność: {test_accuracy:.2f}")

y_true = test_generator.classes
y_pred = (model.predict(test_generator) > 0.5).astype('int32').flatten()
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))
conf_matrix = confusion_matrix(y_true, y_pred)
print("Macierz błędów:")
print(conf_matrix)

# Save model
model_save_path = 'model_efficientnetb7.h5'
model.save(model_save_path)
print(f"Model zapisano w {model_save_path}")

with open("classification_report_efficientnetb7.txt", "w") as f:
    f.write("Raport klasyfikacji:\n") 
    f.write(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))
    f.write("\nMacierz błędów:\n")
    f.write(np.array2string(conf_matrix))

plt.plot(history.history['accuracy'], label='Dokładność treningu')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacji')
plt.title('Dokładność treningu i walidacji')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Strata treningu')
plt.plot(history.history['val_loss'], label='Strata walidacji')
plt.title('Strata treningu i walidacji')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()
plt.show()
