import tensorflow as tf


def LeNet_5(input_shape):
    """Implements LeNet-5 CNN architecture to your neural network.

    :param input_shape: shape of images
    :type input_shape: (height, width, num of channels)

    :return: model, full model of your network
    """

    input = tf.keras.Input(shape=input_shape)

    Z1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=2)(input)
    A1 = tf.keras.layers.Activation("sigmoid")(Z1)
    P1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(A1)
    Z2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=1)(P1)
    A2 = tf.keras.layers.Activation("sigmoid")(Z2)
    P2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(A2)
    FC1 = tf.keras.layers.Flatten()(P2)
    FC2 = tf.keras.layers.Dense(120, activation="sigmoid")(FC1)
    FC3 = tf.keras.layers.Dense(84, activation="sigmoid")(FC2)
    output = tf.keras.layers.Dense(10)(FC3)

    model = tf.keras.Model(inputs=input, outputs=output)

    return model
