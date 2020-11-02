from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.optimizers import adam
import os
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras import regularizers
# dimensions of our images.
img_width, img_height = 224,224
train_data_dir = r'C:\Users\randy\Downloads\the_way_to_train\recheck'
validation_data_dir = r'C:\Users\randy\PycharmProjects\PJ1\classifiaction\test5'
test_dir= r'C:\Users\randy\PycharmProjects\PJ1\classifiaction\test'

train_count =sum([len(files) for r, d, files in os.walk(train_data_dir)])
val_count =sum([len(files) for r, d, files in os.walk(validation_data_dir)])
test_count =sum([len(files) for r, d, files in os.walk(test_dir)])
validation_split=0.25
print(train_count)

nb_train_samples = train_count#*(1-validation_split)
nb_validation_samples =val_count#*validation_split #0.514 0.485
print(nb_train_samples)
print(nb_validation_samples)


nb_test_samples=test_count

epochs = 10
batch_size =16
steps_epoch = nb_train_samples/batch_size
validation_steps= nb_validation_samples/batch_size

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


#compile the model

checkpoint = ModelCheckpoint('voVGG16_2c_2.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
checkpoint2 = ModelCheckpoint('voVGG16_2c_3.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1. / 255)#,validation_split=validation_split)

# this is the augmentation configuration we will use for testing:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes=['3view','others'],
    shuffle=True)#,subset='training')
validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes=['3view','others'],
    shuffle=True)#,subset='validation')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes=['3view','others'],
    shuffle=False)

head_model = load_model('VGG16_2c_3.h5')
'''
base_model = VGG16(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(30, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
head_model = Model(inputs=base_model.input, outputs=predictions)
'''
#compile the model
Adam=adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False,clipnorm=1.)
head_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

head_model.fit_generator(
    train_generator,
    steps_per_epoch=steps_epoch ,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps
    ,callbacks=[checkpoint])



end_model=load_model('voVGG16_2c_2.h5')
for i, layer in enumerate(end_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in end_model.layers[:19]:
   layer.trainable = False
for layer in end_model.layers[19:]:
   layer.trainable = True
from keras.optimizers import SGD
end_model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])





end_model.fit_generator(
    train_generator,
    steps_per_epoch=steps_epoch ,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps
    ,callbacks=[checkpoint2])


Y_pred = end_model.predict_generator(test_generator, nb_test_samples/batch_size)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['3view', 'others']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))
end_model.save('VO_2_classification_model.h5')
