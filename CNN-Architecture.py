inputs = tf.keras.layers.Input(shape = (64, 64,3))

layer1 = tf.keras.layers.Conv2D(42,(3,3),activation = 'relu',use_bias = 1,kernel_regularizer = 'l2',padding = 'same')(inputs)
MaxPool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1,1),padding = 'same')(layer1)


layer2 = tf.keras.layers.Conv2D(42,(3,3),activation = 'relu',use_bias = 1,kernel_regularizer= 'l2',padding = 'same')(MaxPool1)
MaxPool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1,1),padding = 'same')(layer2)


#Inception Layer 1
Inception_layer1_con_1 = tf.keras.layers.Conv2D(42,(3,3),activation = 'relu',use_bias = 1,kernel_regularizer = 'l2',strides = (1,1),padding ='same')(MaxPool2)
Inception_layer1_con_2 = tf.keras.layers.Conv2D(42,(5,5),activation = 'relu',use_bias = 1,kernel_regularizer = 'l2',strides = (1,1),padding ='same')(MaxPool2)
MaxPool3 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1,1),padding = 'same')(MaxPool2)
Concatenate1 = tf.keras.layers.Concatenate(axis=-1)([Inception_layer1_con_1,Inception_layer1_con_2,MaxPool3])

#Inception Layer 2 
Inception_layer2 = tf.keras.layers.Conv2D(64,(1,1),activation = 'relu',use_bias = 1,kernel_regularizer = 'l2',strides = (1,1),padding ='same')(Concatenate1)

Inception_layer2_con_1 = tf.keras.layers.Conv2D(42,(3,3),activation = 'relu',use_bias = 1,kernel_regularizer = 'l2',strides = (1,1),padding ='same')(Inception_layer2)
Inception_layer2_con_2 = tf.keras.layers.Conv2D(42,(5,5),activation = 'relu',use_bias = 1,kernel_regularizer = 'l2',strides = (1,1),padding ='same')(Inception_layer2)
MaxPool4 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1,1),padding = 'same')(Inception_layer2)
Concatenate2 = tf.keras.layers.Concatenate(axis=-1)([Inception_layer2_con_1,Inception_layer2_con_2,MaxPool4])

Concatenate3 =  tf.keras.layers.Concatenate(axis=-1)([Concatenate1, Concatenate2])
layer3 = tf.keras.layers.Conv2D(64,(1,1),activation = 'relu',use_bias = 1,kernel_regularizer = 'l2',strides = (1,1),padding ='same')(Concatenate3)

#Fully Connected Layers
Flatten =  tf.keras.layers.Flatten()(layer3)
hidden_1 = tf.keras.layers.Dense(20, activation = 'relu')(Flatten)
dropout = tf.keras.layers.Dropout(rate = 0.1)(hidden_1)
outputs = tf.keras.layers.Dense(200, activation = tf.keras.activations.softmax)(dropout)


model = tf.keras.Model(inputs = inputs, outputs = outputs)
model.summary()

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1./255,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    dtype=None,
)

checkpoint_filepath = './checkpoint.ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    verbose = 1)
    
model.load_weights('./checkpoint')
model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.02), loss = tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'])
print('done')
his = model.fit(datagen.flow(X_train, Y_train, batch_size=500),steps_per_epoch=len(X_train) /500, epochs=800,validation_data=(X_val,Y_val),shuffle = True,callbacks=[model_checkpoint_callback])
