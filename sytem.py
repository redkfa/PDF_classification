from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from pdf2image import convert_from_path
import PIL
#4896*6336
pdf_dir = r"C:\Users\randy\Downloads\Fiber Optic Connectors\ST"
PDFsave_dir = r'C:\Users\randy\PycharmProjects\PJ1\PDF'
load_path = r'C:\Users\randy\Downloads\the_way_to_train\others\3view'
save_dir = r'C:\Users\randy\Downloads\the_way_to_train\recheck\3view'
other_dir= r'C:\Users\randy\Downloads\the_way_to_train\recheck\others'
test_count =sum([len(files) for r, d, files in os.walk(load_path)])
print(test_count)

def pdf2img(pdf_dir):
    os.chdir(pdf_dir)
    for pdf_file in os.listdir(pdf_dir):

        if pdf_file.endswith(".pdf"):

            pages = convert_from_path(pdf_file, dpi=50)
            pdf_file = pdf_file[:-4]

            for page in pages:
                page.save("%s/%s-page%d.png" % (PDFsave_dir,pdf_file,pages.index(page)), "png")
    return 0


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
    #turn pdf to img
    #pdf2img(pdf_dir)
    # load model

    model = load_model(r"C:\Users\randy\PycharmProjects\PJ1\VGG16_2c_2.h5")
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # image path
    #img_path = r'C:\Users\randy\PycharmProjects\PJ1\classifiaction\test2\3view\305.jpg'
    #img_path = r'C:\Users\randy\PycharmProjects\PJ1\classifiaction\test2\others\o07.jpg'


    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(load_path):
        for file in f:
            if '.png' in file:
                files.append(os.path.join(r, file))

    for f in files:
        print(f)
        # load a single image
        new_image = load_image(f)
        img = cv2.imread(f)
        base = os.path.basename(f)
        # check prediction
        pred = model.predict(new_image)

        if pred[0][0] >= 0.5:
            prediction = '3view'
        #elif 0.25<=pred[0][0] < 0.75:
        #    prediction = 'check'
        else :
            prediction = 'others'
        print(pred)
        print(pred[0])
        print(pred[0][0])
        print(prediction)
        if prediction == '3view':
            #cv2.imwrite("%s/%stest.png" % (save_dir,base) , img)
            print('1')
       # elif prediction == 'check':
        #    cv2.imwrite(r"C:\Users\randy\Downloads\the_way_to_train\new_output\check\%s" % (base), img)
        else:
            cv2.imwrite("%s/%stest.png" % (other_dir,base) , img)
'''
    all_count = sum([len(files) for r, d, files in os.walk(load_path)])
    print(all_count)
    view_count = sum([len(files) for r, d, files in os.walk(save_dir)])
    print(view_count)
    other_count = sum([len(files) for r, d, files in os.walk(save_dir)])
    print(other_count)
'''