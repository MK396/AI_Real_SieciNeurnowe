import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

image_path = "C:/Users/mikol/OneDrive/Semestr6/Sieci_Neuronowe/test_images/0010.jpg"
# Wczytanie obrazu i zmiana rozmiaru
img = image.load_img(image_path, target_size=(224, 224))
# Konwersja obrazu na tablicę NumPy
img_array = image.img_to_array(img)
# Normalizacja wartości pikseli (tak jak w generatorze: /255)
img_array = img_array / 255.0
# Dodanie wymiaru batcha (model oczekuje kształtu [batch_size, height, width, channels])
img_array = np.expand_dims(img_array, axis=0)
# Wczytanie modelu
model = tf.keras.models.load_model('model_efficentnetb0.h5')
# Predykcja
prediction = model.predict(img_array)

probability = prediction[0][0]
if probability > 0.5:
    print(f"Obraz sklasyfikowany jako REAl z prawdopodobieństwem {probability*100:.2f}%.")
else:
    print(f"Obraz sklasyfikowany jako AI z prawdopodobieństwem {(1 - probability)*100:.2f}%.")

plt.imshow(image.load_img(image_path))
if probability > 0.5:
    plt.title(f"Predykcja:'REAL'\nPrawdopodobieństwo: {probability*100:.2f}%")
else:
    plt.title(f"Predykcja:'AI'\nPrawdopodobieństwo: {(1 - probability)*100:.2f}%")
plt.axis('off')  # Ukrycie osi
plt.show()