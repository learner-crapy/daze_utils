import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Add
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPool1D, ZeroPadding1D, LSTM, Bidirectional
from keras.models import Sequential, Model
from sklearn.metrics import confusion_matrix
from keras.layers.merge import concatenate
from scipy import optimize

'''
function: define vgg16 model
input: 3 dimentions with the shape like (4583, 12, 5000). type: tuple
output: vgg16 model
'''
def vgg16(Shape):
    vgg_16_model=Sequential()
    vgg_16_model.add(Conv1D(filters=64, kernel_size=3, padding='same',  input_shape=Shape))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Activation('relu'))
    vgg_16_model.add(Conv1D(filters=64, kernel_size=3, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Activation('relu'))
    vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    vgg_16_model.add(Conv1D(filters=128, kernel_size=3, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Activation('relu'))
    vgg_16_model.add(Conv1D(filters=128, kernel_size=3, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Activation('relu'))
    vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    vgg_16_model.add(Conv1D(filters=256, kernel_size=3, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Activation('relu'))
    vgg_16_model.add(Conv1D(filters=256, kernel_size=3, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Activation('relu'))
    vgg_16_model.add(Conv1D(filters=256, kernel_size=3, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Activation('relu'))
    vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    vgg_16_model.add(Conv1D(filters=512, kernel_size=3, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Activation('relu'))
    vgg_16_model.add(Conv1D(filters=512, kernel_size=3, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Activation('relu'))
    vgg_16_model.add(Conv1D(filters=512, kernel_size=3, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Activation('relu'))
    vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    vgg_16_model.add(Conv1D(filters=512, kernel_size=3, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Activation('relu'))
    vgg_16_model.add(Conv1D(filters=512, kernel_size=1, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Activation('relu'))
    vgg_16_model.add(Conv1D(filters=512, kernel_size=1, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Activation('relu'))
    vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    vgg_16_model.add(GlobalAveragePooling1D())
    vgg_16_model.add(Dense(256, activation='relu'))
    vgg_16_model.add(Dropout(0.4))
    vgg_16_model.add(Dense(128, activation='relu'))
    vgg_16_model.add(Dropout(0.4))
    vgg_16_model.add(Dense(9, activation='sigmoid'))
    vgg_16_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.BinaryAccuracy(
            name='accuracy', dtype=None, threshold=0.5),tf.keras.metrics.Recall(name='Recall'),tf.keras.metrics.Precision(name='Precision'), 
                        tf.keras.metrics.AUC(
            num_thresholds=200,
            curve="ROC",
            summation_method="interpolation",
            name="AUC",
            dtype=None,
            thresholds=None,
            multi_label=True,
            label_weights=None,
        )])
        
    # vgg_16_model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    vgg_16_model.summary()
    return vgg_16_model

def Lenet5(Shape):
    lenet_5_model=Sequential()
    lenet_5_model.add(Conv1D(filters=6, kernel_size=3, padding='same', input_shape=Shape))
    lenet_5_model.add(BatchNormalization())
    lenet_5_model.add(Activation('relu'))
    lenet_5_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    lenet_5_model.add(Conv1D(filters=16, strides=1, kernel_size=5))
    lenet_5_model.add(BatchNormalization())
    lenet_5_model.add(Activation('relu'))
    lenet_5_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    lenet_5_model.add(GlobalAveragePooling1D())
    lenet_5_model.add(Dense(64, activation='relu'))
    lenet_5_model.add(Dense(32, activation='relu'))
    lenet_5_model.add(Dense(27, activation = 'sigmoid'))
    lenet_5_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.BinaryAccuracy(
            name='accuracy', dtype=None, threshold=0.5),tf.keras.metrics.Recall(name='Recall'),tf.keras.metrics.Precision(name='Precision'), 
                        tf.keras.metrics.AUC(
            num_thresholds=200,
            curve="ROC",
            summation_method="interpolation",
            name="AUC",
            dtype=None,
            thresholds=None,
            multi_label=True,
            label_weights=None,
        )])

    lenet_5_model.summary()
    return lenet_5_model

def AlexNet(Shape):
    alexNet_model=Sequential()
    alexNet_model.add(Conv1D(filters=96, kernel_size=11, strides=4, input_shape=Shape))
    alexNet_model.add(BatchNormalization())
    alexNet_model.add(Activation('relu'))
    alexNet_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    alexNet_model.add(Conv1D(filters=256, kernel_size=5, padding='same'))
    alexNet_model.add(BatchNormalization())
    alexNet_model.add(Activation('relu'))
    alexNet_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    alexNet_model.add(Conv1D(filters=384, padding='same', kernel_size=3))
    alexNet_model.add(BatchNormalization())
    alexNet_model.add(Activation('relu'))
    alexNet_model.add(Conv1D(filters=384, kernel_size=3))
    alexNet_model.add(BatchNormalization())
    alexNet_model.add(Activation('relu'))
    alexNet_model.add(Conv1D(filters=256, kernel_size=3))
    alexNet_model.add(BatchNormalization())
    alexNet_model.add(Activation('relu'))
    alexNet_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    alexNet_model.add(GlobalAveragePooling1D())
    alexNet_model.add(Dense(128, activation='relu'))
    alexNet_model.add(Dropout(0.4))
    alexNet_model.add(Dense(128, activation='relu'))
    alexNet_model.add(Dropout(0.4))
    alexNet_model.add(Dense(27, activation='sigmoid'))
    alexNet_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.BinaryAccuracy(
            name='accuracy', dtype=None, threshold=0.5),tf.keras.metrics.Recall(name='Recall'),tf.keras.metrics.Precision(name='Precision'), 
                        tf.keras.metrics.AUC(
            num_thresholds=200,
            curve="ROC",
            summation_method="interpolation",
            name="AUC",
            dtype=None,
            thresholds=None,
            multi_label=True,
            label_weights=None,
        )])
    alexNet_model.summary()
    return alexNet_model

# RESNET50
def identity_block(X, f, filters):
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv1D(filters = F1, kernel_size = 1, activation='relu', strides = 1, padding = 'valid')(X)
    X = BatchNormalization()(X)
    X = Conv1D(filters = F2, kernel_size = f, activation='relu', strides = 1, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Conv1D(filters = F3, kernel_size = 1, activation='relu', strides = 1, padding = 'valid')(X)
    X = BatchNormalization()(X)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    return X

def convolutional_block(X, f, filters, s = 2):
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv1D(F1, 1, activation='relu', strides = s)(X)
    X = BatchNormalization()(X)
    X = Conv1D(F2, f, activation='relu', strides = 1,padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Conv1D(F3, 1, strides = 1)(X)
    X = BatchNormalization()(X)
    X_shortcut = Conv1D(F3, 1, strides = s)(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    return X

def ResNet50(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding1D(3)(X_input)
    X = Conv1D(64, 7, strides = 2)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPool1D(pool_size=2, strides=2, padding='same')(X)
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])
    X = convolutional_block(X, f = 3, filters = [128,128,512], s = 2)
    X = identity_block(X, 3, [128,128,512])
    X = identity_block(X, 3, [128,128,512])
    X = identity_block(X, 3, [128,128,512])
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])
    X = MaxPool1D(pool_size=2, strides=2, padding='same')(X)
    X = GlobalAveragePooling1D()(X)
    X = Dense(27,activation='sigmoid')(X)
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    return model

def resNet50Model(Shape):
    resNet50_model = ResNet50(input_shape = Shape)
    resNet50_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.BinaryAccuracy(
            name='accuracy', dtype=None, threshold=0.5),tf.keras.metrics.Recall(name='Recall'),tf.keras.metrics.Precision(name='Precision'), 
                        tf.keras.metrics.AUC(
            num_thresholds=200,
            curve="ROC",
            summation_method="interpolation",
            name="AUC",
            dtype=None,
            thresholds=None,
            multi_label=True,
            label_weights=None,
        )])
    resNet50_model.summary()
    return resNet50_model


# inception
def inception_block(prev_layer):
    conv1=Conv1D(filters = 64, kernel_size = 1, padding = 'same')(prev_layer)
    conv1=BatchNormalization()(conv1)
    conv1=Activation('relu')(conv1)
    conv3=Conv1D(filters = 64, kernel_size = 1, padding = 'same')(prev_layer)
    conv3=BatchNormalization()(conv3)
    conv3=Activation('relu')(conv3)
    conv3=Conv1D(filters = 64, kernel_size = 3, padding = 'same')(conv3)
    conv3=BatchNormalization()(conv3)
    conv3=Activation('relu')(conv3)
    conv5=Conv1D(filters = 64, kernel_size = 1, padding = 'same')(prev_layer)
    conv5=BatchNormalization()(conv5)
    conv5=Activation('relu')(conv5)
    conv5=Conv1D(filters = 64, kernel_size = 5, padding = 'same')(conv5)
    conv5=BatchNormalization()(conv5)
    conv5=Activation('relu')(conv5)
    pool= MaxPool1D(pool_size=3, strides=1, padding='same')(prev_layer)
    convmax=Conv1D(filters = 64, kernel_size = 1, padding = 'same')(pool)
    convmax=BatchNormalization()(convmax)
    convmax=Activation('relu')(convmax)
    layer_out = concatenate([conv1, conv3, conv5, convmax], axis=1)
    return layer_out

def inception_model(input_shape):
    X_input=Input(input_shape)
    X = ZeroPadding1D(3)(X_input)
    X = Conv1D(filters = 64, kernel_size = 7, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPool1D(pool_size=3, strides=2, padding='same')(X)
    X = Conv1D(filters = 64, kernel_size = 1, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = inception_block(X)
    X = inception_block(X)
    X = MaxPool1D(pool_size=7, strides=2, padding='same')(X)
    X = GlobalAveragePooling1D()(X)
    X = Dense(27,activation='sigmoid')(X)
    model = Model(inputs = X_input, outputs = X, name='Inception')
    return model

def InceptionModel(Shape):
    inception_model = inception_model(input_shape = Shape)
    inception_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.BinaryAccuracy(
            name='accuracy', dtype=None, threshold=0.5),tf.keras.metrics.Recall(name='Recall'),tf.keras.metrics.Precision(name='Precision'), 
                        tf.keras.metrics.AUC(
            num_thresholds=200,
            curve="ROC",
            summation_method="interpolation",
            name="AUC",
            dtype=None,
            thresholds=None,
            multi_label=True,
            label_weights=None,
        )])
    inception_model.summary()
    return inception_model

def LstmModel(Shape):
    lstm_model = Sequential()
    lstm_model.add(LSTM(64, input_shape=Shape, return_sequences=True))
    lstm_model.add(LSTM(64))
    lstm_model.add(Dense(32, activation = 'relu'))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(Dense(27, activation = 'sigmoid'))
    lstm_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.BinaryAccuracy(
            name='accuracy', dtype=None, threshold=0.5),tf.keras.metrics.Recall(name='Recall'),tf.keras.metrics.Precision(name='Precision'), 
                        tf.keras.metrics.AUC(
            num_thresholds=200,
            curve="ROC",
            summation_method="interpolation",
            name="AUC",
            dtype=None,
            thresholds=None,
            multi_label=True,
            label_weights=None,
        )])
    lstm_model.summary()
    return lstm_model