import tensorflow as tf

def AlexNet(input_shape):
    """Implements AlexNet CNN architecture to your neural network.

    :param input_shape: shape of images
    :type input_shape: (height, width, num of channels)

    :return: model, full model of your network
        """

    input = tf.keras.Input(shape=input_shape)
    Z1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4)(input)
    A1 = tf.keras.layers.ReLU()(Z1)
    P1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(A1)
    Z2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding="same")(P1)
    A2 = tf.keras.layers.ReLU()(Z2)
    P2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(A2)
    Z3 = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding="same")(P2)
    A3 = tf.keras.layers.ReLU()(Z3)
    Z4 = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding="same")(A3)
    A4 = tf.keras.layers.ReLU()(Z4)
    Z5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same")(A4)
    P3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)
    FC1 = tf.keras.layers.Flatten()(P3)
    FC2 = tf.keras.layers.Dense(4096, activation="relu")(FC1)
    FC3 = tf.keras.layers.Dense(4096, activation="relu")(FC2)
    output = tf.keras.layers.Dense(1000, activation="softmax")(FC3)

    model = tf.keras.Model(inputs=input, outputs=output)

    return model
