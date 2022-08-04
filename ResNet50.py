def identity_block(X, f, num_of_filters, training=True, initializer=tf.keras.initializers.random_uniform):
    """Implement first block of ResNet, there is no dimensional disagreement
    
    :param X: Input matrix
    :type X: np.array(m, n_height_prev, n_width_prev, C_prev)
    :param f: shape of second kernel
    :type f: integer
    :param num_of_filters: num of filters in main paths(3 nums)
    :type num_of_filters: list
    :param training: True if implementation is for training purposes
    :type training: Boolean
    :param initializer: the function to be used for initialization of the parameters
    :type initializer: function

    :return: X, output of identity block
    :rtype: np.array(m, height, width, C)
    """

    F1, F2, F3 = num_of_filters
    X_shortpath = X

    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer())(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X, training=training)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same', kernel_initializer=initializer())(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X, training=training)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer())(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X, training=training)

    X = tf.keras.layers.Add()([X_shortpath, X])
    X = tf.keras.layers.Activation('relu')(X)

    
    return X


def convolutional_block(X, f, num_of_filters, stride=2, training=True, initializer=tf.keras.initializers.Glorot_uniform):
    """Implement first block of ResNet, there is dimensional disagreement
    
    :param X: Input matrix
    :type X: np.array(m, n_height_prev, n_width_prev, C_prev)
    :param f: shape of second kernel
    :type f: integer
    :param num_of_filters: num of filters in main paths(3 nums)
    :type num_of_filters: list
    :param stride: amount of stride
    :type stride: integer
    :param training: True if implementation is for training purposes
    :type training: Boolean
    :param initializer: the function to be used for initialization of the parameters
    :type initializer: function

    :return: X, output of identity block
    :rtype: np.array(m, height, width, C)
    """

    F1, F2, F3 = num_of_filters
    X_shortpath = X

    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=1, strides=(stride, stride), padding='valid', kernel_initializer=initializer())(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X, training=training)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same', kernel_initializer=initializer())(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X, training=training)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer())(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X, training=training)

    X_shortpath = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=(stride, stride), padding='valid', kernel_initalizer=initializer())(X_shortpath)
    X_shortpath = tf.keras.layers.BatchNormalization(axis=3)(X_shortpath, training=training)

    X = tf.keras.layers.Add()([X, X_shortpath])
    X = tf.keras.layers.Activation('relu')(X)

    
    return X


def ResNet50(input_shape = (64, 64, 3), classes=10):
    """Implementation of ResNet50 neural network architecture
    
    :param input_shape: shape of input
    :type input_shape: np.array
    :param classes: num of classification classes
    :type classes: integer

    :return: model
    :rtype: tf.keras.models.Model()
    """

    input_X = tf.keras.layers.Input(input_shape)
    
    X = tf.keras.layers.ZeroPadding2D((3, 3), input_X)

    X = tf.keras.layers.Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = 'glorot_uniform')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
    X = identity_block(X, 3, [128, 128, 512]) 
    X = identity_block(X, 3, [128, 128, 512]) 
    X = identity_block(X, 3, [128, 128, 512]) 
    
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
    X = identity_block(X, 3, [256, 256, 1024]) 
    X = identity_block(X, 3, [256, 256, 1024]) 
    X = identity_block(X, 3, [256, 256, 1024]) 
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
    X = identity_block(X, 3, [512, 512, 2048]) 
    X = identity_block(X, 3, [512, 512, 2048]) 

    X = tf.keras.layers.AveragePooling2D((2, 2))(X)

    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(classes, activation='softmax', kernel_initializer = 'glorot_uniform')(X)
    
    model = tf.keras.models.Model(inputs = input_X, outputs = X)

    
    return model
