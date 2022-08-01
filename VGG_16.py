import tensorflow as tf


def VGG_16(input_shape):

    """Implements AlexNet CNN architecture to your neural network.

    :param input_shape: shape of images
    :type input_shape: (height, width, num of channels)

    :return: model, full model of your network
    """

    input = tf.keras.Input(shape=input_shape)
    Z1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(input)
    Z2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(Z1)
    P1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(Z2)

    Z3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(P1)
    Z4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(Z3)
    P2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(Z4)

    Z5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(P2)
    Z6 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(Z5)
    Z7 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(Z6)
    P3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(Z7)

    Z8 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(P3)
    Z9 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(Z8)
    Z10 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(Z9)
    P4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(Z10)

    Z11 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(P4)
    Z12 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(Z11)
    Z13 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(Z12)
    P5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(Z13)

    FC1 = tf.keras.layers.Flatten()(P5)
    FC2 = tf.keras.layers.Dense(4096, activation="relu")(FC1)
    FC3 = tf.keras.layers.Dense(4096, activation="relu")(FC2)
    output = tf.keras.layers.Dense(1000, activation="softmax")(FC3)

    model = tf.keras.Model(inputs=input, outputs=output)

    return model
