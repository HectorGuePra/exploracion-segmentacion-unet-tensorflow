import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

def get_encoder(input_shape=[128, 128, 3]):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False) 
    # Cargamos el modelo preentrenado MobileNetV2 y le quitamos la capa superior que es el clasificador para usarlo como extractor de características. El 3 son los canales de color RGB.
    #Codificador (downsampler)
    # Use the activations of these layers
    layer_names = [
        # Seleccionamos las capas cuyas salidas usaremos como características extraídas pero en realidada puedo escoger otras capas.
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    # Devuelve una lista con 5 tensores de características extraídas en las capas especificadas.

    down_stack.trainable = False
    return down_stack


#Decodificador (upsampler)
def get_decoder():
    up_stack = [
    #       Conv2DTranspose => Batchnorm => Dropout => Relu

    #   Args:
    #     filters: number of filters
    #     size: filter size
    #     norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    #     apply_dropout: If True, adds the dropout layer

    #   Returns:
    #     Upsample Sequential Model
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]
    return up_stack

def build_unet_model(output_channels:int, input_shape=[128, 128, 3]): # output_channels = número de clases en la segmentación 
    inputs = tf.keras.layers.Input(input_shape)

    # Downsampling through the model
    down_stack = get_encoder(input_shape)
    skips = down_stack(inputs) # Lista de tensores de características extraídas
    x = skips[-1] # Último tensor de características extraídas (4x4)
    skips = reversed(skips[:-1]) # Excluimos el último tensor(pq lo tenemos en x) y los invertimos para usarlos en el upsampling como chuleta
    up_stack = get_decoder()
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x) # Upsampling
        concat = tf.keras.layers.Concatenate() # Concatenamos creando un tensor con las características del upsampling y las la chuleta
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128
    # kernel_size: Tamaño del filtro convolucional
    # padding: 'same' para que la salida tenga el mismo tamaño que la entrada cuando el stride es 1
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

