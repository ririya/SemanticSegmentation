from keras.layers import Conv2D,Conv2DTranspose, BatchNormalization,Activation,MaxPooling2D,Dropout, concatenate, Input,Reshape
from keras.models import Model
from keras.utils import plot_model
import vgg16

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def unetVGG16(params, batchnorm = True):

    input_shape = (params.input_height, params.input_width, 3)
    input_img = Input(shape=input_shape)

    (initial_model, c1, c2, c3, c4, c5, _) = vgg16.VGG16(weights=params.preTrainedWeights, input_tensor=input_img,
                                                         input_shape=input_shape, include_top=False)

    # Expansive Path

    n_filters = 512

    u6 = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(params.dropout)(u6)
    c6 = conv2d_block(u6, n_filters, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(int(n_filters/2), (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(params.dropout)(u7)
    c7 = conv2d_block(u7, int(n_filters/2), kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(int(n_filters/4), (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(params.dropout)(u8)
    c8 = conv2d_block(u8, int(n_filters / 4), kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(int(n_filters/8), (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(params.dropout)(u9)
    c9 = conv2d_block(u9, int(n_filters / 8), kernel_size=3, batchnorm=batchnorm)

    c10 = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')(c9)

    if params.loss == 'binary_crossentropy':
        o = Conv2D(1, (1, 1), activation='sigmoid')(c10)
    else:
        o = Conv2D(params.numClasses, (3, 3), padding='same')(c10)
        o = Activation('softmax')(o)

    model = Model(inputs=input_img, outputs=o)

    plot_model(model, show_shapes=True, to_file='UNET.png')

    return model


def unet(params,n_filters=16, batchnorm=True):

    if params.RGB:

        input_shape = (params.input_height, params.input_width, 3)
    else:
        input_shape = (params.input_height, params.input_width, 1)

    input_img = Input(shape=input_shape)

    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(params.dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(params.dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(params.dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(params.dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(params.dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(params.dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(params.dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(params.dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    if params.loss == 'binary_crossentropy':
        o = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    else:
        o = Conv2D(params.numClasses, (3, 3), padding='same')(c9)
        o = Activation('softmax')(o)

    model = Model(inputs=input_img, outputs=o)

    plot_model(model, show_shapes=True, to_file='UNET.png')

    return model

