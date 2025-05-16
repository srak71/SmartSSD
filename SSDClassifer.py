import tensorflow as tf 
import os

# user dir
current_dir= os.getcwd()

# known filepaths
train_dir= os.path.join(current_dir, "NEU-CLS", "train", "images")
val_dir= os.path.join(current_dir, "NEU-CLS", "validation", "images")

# unified variables
image_size= (128, 128)
batch_size= 32 # number of images to process before updating mode weights 
epochs= 10 # number of times entire dataset trains on model
num_classes= 6

# Augmenting training images by rotating flipping and altering brightness
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                rotation_range=20, # random rotations
                                                                horizontal_flip=True, # random horizontal flips
                                                                brightness_range=(0.8, 1.2)) # random brightness

# just normalizing in the case of val images
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) 

# actually generating the image defined from my ImageDataGenerator() function
train_gen = train_datagen.flow_from_directory(train_dir,
                                              target_size=image_size,
                                              color_mode='grayscale',
                                              batch_size=batch_size,
                                              class_mode='categorical')

val_gen = val_datagen.flow_from_directory(val_dir,
                                          target_size=image_size,
                                          color_mode='grayscale',
                                          batch_size=batch_size,
                                          class_mode='categorical')

model = tf.keras.models.Sequential([
    # Block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation= 'relu', input_shape= (128, 128, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    # Block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation= 'relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.25),
    
    # Block 3
    tf.keras.layers.Conv2D(128, (3, 3), activation= 'relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    # Classification head
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation= 'relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation= 'softmax')])


model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()


earlyStop = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss',
                                             patience= 3, # test for improvement in next 3, if none stop and load old
                                             restore_best_weights= True,
                                             verbose= 1)

# after stopping loading weights from old epoch (the one with the lowest val loss)
modelCheckpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras',
                                                     monitor= 'val_loss',
                                                     save_best_only= True,
                                                     verbose= 1)

callbacks = [earlyStop, modelCheckpoint]



history = model.fit(train_gen,
                    epochs= epochs,
                    validation_data= val_gen,
                    callbacks= callbacks,
                    verbose= 1)

