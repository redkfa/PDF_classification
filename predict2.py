# load_model_sample.py
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


save_dir = r'C:\Users\randy\PycharmProjects\PJ1\tsetinput'

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]


    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


if __name__ == "__main__":

    # load model
    model = load_model("Sequential_2c.h5")
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # image path
    img_path = r'C:\Users\randy\PycharmProjects\PJ1\classifiaction\test2\3view\305.jpg'
    #img_path = r'C:\Users\randy\PycharmProjects\PJ1\classifiaction\test2\others\o07.jpg'

    # load a single image
    new_image = load_image(img_path)

    # check prediction
    pred = model.predict(new_image)

    if pred[0][0] >= 0.5:
        prediction = '3view'
    else :
        prediction = 'others'
    print(pred)
    print(pred[0])
    print(pred[0][0])
    print(prediction)


