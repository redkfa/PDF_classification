import csv
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
import pandas as pd
from keras.models import Sequential
import tensorflow as tf
from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import os
from sklearn.metrics import auc
from  sklearn.metrics import precision_recall_fscore_support
from  sklearn.metrics import cohen_kappa_score


def pandas_classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred
            )
    avg = list(precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            average='weighted'))
    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)
    support = class_report_df.loc['support']
    total = support.sum()
    avg[-1] = total
    class_report_df['avg / total'] = avg
    return class_report_df.T

def idnw_input(predict_data_dir,model_load):
    predict_data_dir = predict_data_dir
    test_count =sum([len(files) for r, d, files in os.walk(predict_data_dir)])
    nb_validation_samples =test_count
    batch_size =8
    validation_steps= nb_validation_samples/batch_size
    print(test_count)
    print(validation_steps)
    img_width, img_height = 224,224
    model = load_model(model_load)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        predict_data_dir,
        target_size=(img_width, img_height),
        color_mode='grayscale',
        shuffle=False,
        batch_size=batch_size)
    Y_pred = model.predict_generator(validation_generator,len(validation_generator),verbose=1)#
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = validation_generator.classes


    print('Confusion Matrix')
    cmt=confusion_matrix(validation_generator.classes, y_pred)
    print(cmt)

    groups = ["3view","others"]
    _3view=[cmt[0][0],cmt[1][0]]
    others=[cmt[0][1],cmt[1][1]]
    dict ={"Confusion Matrix":groups,
           "3view":_3view,
           "others":others}
    select_df = pd.DataFrame(dict)
    select_df.set_index("Confusion Matrix", inplace=True)
    print(select_df)


    # dcm=pd.DataFrame(cmt).transpose()
   # ironman1 = pd.Series(cmt, index=['Confusion Matrix'])
   # print(ironman1)

    print('Classification Report')
    target_names = ['3view', 'others']
    crt=classification_report(y_true, y_pred, target_names=target_names,output_dict=True)
    df=pd.DataFrame(crt).transpose()
    print(df)
    df_class_report = pandas_classification_report(y_true=y_true, y_pred=y_pred)
    print(df_class_report)
    df_class_report.to_csv('classification_csv_file.csv',  sep=',')
    kappa= cohen_kappa_score(y_true, y_pred)
    kappa2=[kappa]
    kappalist=['kappa']
    groupkappa=['val']
    dict2 = {'  ':kappalist,'val':kappa2
             }
    kdf=pd.DataFrame(dict2)
    kdf.set_index('  ', inplace=True)
    print(kdf)
    print('-------------------------------')
    #print("cohen_kappa_score: %s" %kappa)
    select_df.append(df)
    print(select_df)
    res = pd.concat([select_df,df],sort=False)
   # print(res)
    res2 = pd.concat([res,kdf],sort=False)

    print(res2)
    res2.to_csv('classification_csv_file.csv',sep=',')

    '''
    with open('classification_csv_file.csv', 'a') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile,delimiter=',')

        cmt_table=['Confusion Matrix',cmt]
        kappa_table = ['Cohen kappa score', kappa]
        writer.writerow(cmt_table)
        writer.writerow(kappa_table)
        writer.writerow('model')
     '''
    with open('classification_csv_file.csv', 'a') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

if __name__ == "__main__":
    predict_data_dir = r'C:\Users\randy\PycharmProjects\PJ1\classifiaction\test5'
    my_model = 'Sequential_gray.h5'
    idnw_input(predict_data_dir, my_model)




'''

y_pred_keras = y_pred.ravel()
print(y_pred_keras)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(validation_generator.classes, y_pred_keras)
auc_keras = auc(fpr_keras,tpr_keras)

figname = 'test_plt.jpg'
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='ROC (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig(figname)
print('auc=%s'% auc_keras)

from contextlib import redirect_stdout

plt.show()
    writer.writerow(['aou',figname])
    writer.writerow(['auc', auc_keras])


    def myprint(s):
        with open('my_csv_file.csv', 'a') as f:
            writer2 = csv.writer(f)
            writer2.writerow([s])
    my_model.summary(print_fn=myprint)
'''



