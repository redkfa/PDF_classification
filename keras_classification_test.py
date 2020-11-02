from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from keras.models import Sequential
import tensorflow as tf
from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import os
from sklearn.metrics import auc

#validation_data_dir = r'C:\Users\randy\PycharmProjects\PJ1\classifiaction\test'
validation_data_dir = r'C:\Users\randy\PycharmProjects\PJ1\classifiaction\test5'
#C:\Users\randy\Downloads\betterdataset\test   494#
#C:\Users\randy\PycharmProjects\PJ1\classifiaction\test2   #16
test_count =sum([len(files) for r, d, files in os.walk(validation_data_dir)])
nb_validation_samples =test_count
batch_size =8

validation_steps= nb_validation_samples/batch_size
print(test_count)
print(validation_steps)

img_width, img_height = 224,224
my_model = load_model('VO_2_classification_model.h5')
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    shuffle=False,
    batch_size=batch_size)



Y_pred = my_model.predict_generator(validation_generator,len(validation_generator),verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
y_true = validation_generator.classes

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['3view', 'others']
print(classification_report(y_true, y_pred, target_names=target_names))


'''
loss, acc = my_model.evaluate_generator(validation_generator, steps=len(validation_generator), verbose=1)
print('test acc = %.3f'%(acc))
print('test loss = %.3f'%(loss))
'''

'''
y_pred_keras = Y_pred.ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(validation_generator.classes, y_pred_keras)
auc_keras = auc(fpr_keras,tpr_keras)
print(auc_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='ROC (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
print(auc_keras)
'''





'''
#fpr, tpr, thresholds = metrics.roc_curve(y_true,Y_pred, pos_label=2)
plt.plot(fpr_keras,tpr_keras,marker = 'o')
plt.show()
#AUC = auc(fpr, tpr)
'''