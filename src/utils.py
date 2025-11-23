import matplotlib.pyplot as plt
import tensorflow as tf

def display_sample(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def create_mask(pred_mask):
    """Procesa la predicción para visualizarla (argmax)."""
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(model, dataset=None, num=1):
    """
    Muestra predicciones usando el modelo y un dataset.
    
    Args:
        model: El modelo entrenado (U-Net).
        dataset: El dataset de tf.data (batch) para predecir.
        num: Número de ejemplos a mostrar.
    """
    if dataset:
        # Iteramos sobre el dataset 'num' veces
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            # Usamos las funciones que definimos arriba
            display_sample([image[0], mask[0], create_mask(pred_mask)])
    else:
        # Si no hay dataset, no podemos predecir nada en este contexto genérico
        print("Error: Por favor proporciona un dataset para realizar predicciones.")