from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D
from keras.layers import Concatenate, Activation
from keras.models import Model
from keras import backend as K
import tensorflow as tf

def up_sampled(x, factor,k_size):
    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(k_size, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)
    return x

def side_branch(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)

    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)

    return x


def Ugg():
    # Input
    #img_input = Input(shape=(480,480,3), name='input')
    img_input = Input(shape=(320, 320, 3), name='input')
    #img_input = Input(shape=(256, 256, 3), name='input')

    # Block 1
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(c1)
    #b1= side_branch(x, 1)
    mp1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(c1) # 160 160 64
    #print(K.int_shape(mp1))
    # Block 2
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(mp1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(c2)
    #b2= side_branch(x, 2) # 480 480 1
    mp2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(c2) # 80 80 128
    #print(K.int_shape(mp2))
    # Block 3
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(mp2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(c3)
    #b3= side_branch(x, 4) # 480 480 1
    mp3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(c3) # 40 40 256
    #print(K.int_shape(mp3))
    # Block 4
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(mp3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(c4)
    #print(K.shape(c4))
    #b4= side_branch(x, 8) # 480 480 1
    mp4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(c4) # 20 20 512
    #print(K.int_shape(mp4))
    # Block 5 MID
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(mp4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(c5) # 20 20 512
    #print(K.int_shape(c5))

    #b5= side_branch(x, 16) # 480 480 1

    # Block 6
    up6 = up_sampled(c5, 2, 512)  # 40 40 512
    fuse1 = Concatenate(axis=-1)([up6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(fuse1)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2')(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv3')(c6)

    # Block 7
    up7 = up_sampled(c6, 2, 256)  # 80 80 256
    fuse2 = Concatenate(axis=-1)([up7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv1')(fuse2)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv2')(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv3')(c7)

    # Block 8
    up8 = up_sampled(c7, 2, 128)  # 160 160 128
    fuse3 = Concatenate(axis=-1)([up8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block8_conv1')(fuse3)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block8_conv2')(c8)

    # Block 9
    up9 = up_sampled(c8, 2, 64)  # 320 320 64
    fuse4 = Concatenate(axis=-1)([up9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block9_conv1')(fuse4)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block9_conv2')(c9)

    outone = Conv2D(1, (1, 1), activation='sigmoid', name='outone')(c9)

    # model
    model = Model(inputs=[img_input], outputs=outone)
    #load_weights_from_hdf5_group_by_name(model, filepath)

    model.compile(loss=cross_entropy_balanced,
                  metrics={'outone': std_pixel_error},
                  optimizer='adam')

    return model


def cross_entropy_balanced(y_true, y_pred):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    #y_0 = tf.zeros_like(y_true)
    #count_neg = tf.reduce_mean(tf.cast(tf.equal(y_true, y_0), tf.float32))
    #count_pos = tf.reduce_sum(y_true)

    #print(count_neg.eval())
    #print(count_pos.eval())
    #quit()

    # Equation [2]
    #factor = 1.1
    beta = 1.0 * count_neg / (count_neg + count_pos)
    #pos_weight = factor * count_pos / (count_neg + count_pos)
    #pos_weight = factor * beta / (1 - beta)
    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def std_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def load_weights_from_hdf5_group_by_name(model, filepath):
    ''' Name-based weight loading '''

    import h5py

    f = h5py.File(filepath, mode='r')

    flattened_layers = model.layers
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in flattened_layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # we batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            if len(weight_values) != len(symbolic_weights):
                raise Exception('Layer #' + str(k) +
                                ' (named "' + layer.name +
                                '") expects ' +
                                str(len(symbolic_weights)) +
                                ' weight(s), but the saved weights' +
                                ' have ' + str(len(weight_values)) +
                                ' element(s).')
            # set values
            for i in range(len(weight_values)):
                weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
                K.batch_set_value(weight_value_tuples)
