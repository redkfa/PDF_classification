'''
from functools import reduce
from PIL import Image



# 計算圖片的局部哈希值--pHash
def phash(img):
    """
    :param img: 圖片
    :return: 返回圖片的局部hash值
    """
    img = img.resize((8, 8), Image.ANTIALIAS).convert('L')
    avg = reduce(lambda x, y: x + y, img.getdata()) / 64.
    hash_value=reduce(lambda x, y: x | (y[1] << y[0]), enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())), 0)
    print(hash_value)
    return hash_value

# 計算漢明距離:
def hamming_distance(a, b):
    """
    :param a: 圖片1的hash值
    :param b: 圖片2的hash值
    :return: 返回兩個圖片hash值的漢明距離
    """
    hm_distance=bin(a ^ b).count('1')
    print(hm_distance)
    return hm_distance


# 計算兩個圖片是否相似:
def is_imgs_similar(img1,img2):
    """
    :param img1: 圖片1
    :param img2: 圖片2
    :return:  True 圖片相似  False 圖片不相似
    """
    return True if hamming_distance(phash(img1),phash(img2)) <= 5 else False




if __name__ == '__main__':

    # 讀取圖片
    sensitive_pic = Image.open("classifiaction/test/3view/8183S-page17.jpg")
    target_pic = Image.open("classifiaction/test/3view/8183S-page16.jpg")

    # 比較圖片相似度
    result=is_imgs_similar(target_pic, sensitive_pic)

    print(result)
'''
import cv2
import time
import numpy as np
import os
from shutil import copyfile
load_path = r"C:\Users\randy\Downloads\the_way_to_train\no_repeat\others"
save_path = r"C:\Users\randy\Downloads\the_way_to_train\no_repeat\others"
files = []
# 均值哈希算法
def aHash(img):
    img = cv2.resize(img, (8, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    np_mean = np.mean(gray)  # 求numpy.ndarray平均值
    ahash_01 = (gray > np_mean) + 0  # 大于平均值=1，否则=0
    ahash_list = ahash_01.reshape(1, -1)[0].tolist()  # 展平->转成列表
    ahash_str = ''.join([str(x) for x in ahash_list])
    return ahash_str


def pHash(img):
    img = cv2.resize(img, (32, 32))  # 默认interpolation=cv2.INTER_CUBIC
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_roi = dct[0:8, 0:8]  # opencv实现的掩码操作

    avreage = np.mean(dct_roi)
    phash_01 = (dct_roi > avreage) + 0
    phash_list = phash_01.reshape(1, -1)[0].tolist()
    phash_str = ''.join([str(x) for x in phash_list])
    return phash_str


def dHash(img):
    img = cv2.resize(img, (9, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    hash_str0 = []
    for i in range(8):
        hash_str0.append(gray[:, i] > gray[:, i + 1])
    hash_str1 = np.array(hash_str0) + 0
    hash_str2 = hash_str1.T
    hash_str3 = hash_str2.reshape(1, -1)[0].tolist()
    dhash_str = ''.join([str(x) for x in hash_str3])
    return dhash_str


def hammingDist(s1, s2):
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])


# 通过得到RGB每个通道的直方图来计算相似度
def classify_hist_with_split(image1, image2, size=(256, 256)):
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data


# 计算单通道的直方图的相似值
def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


if __name__ == '__main__':
    count = 0
    remove_list=[]
    index_count=23#23
    for r, d, f in os.walk(load_path):
        for file in f:

            if '.png' in file:
                files.append(os.path.join(r, file))
    start = time.time()

    for index in range(index_count,len(files)):

        index2 = index + 1
        for index2  in range((index+1),(len(files))):
            if files[index] not in os.listdir(os.getcwd()):
                index+1
            if files[index2] not in os.listdir(os.getcwd()):
                index+1
            img1 = files[index]
            raw_img1 = img1
            img1 = cv2.imread(raw_img1)
            #ahash_str1 = aHash(img1)
            #phash_str1 = pHash(img1)
           # dhash_str1 = dHash(img1)
 #           if index+1 <= len(files)-1:
            img2 = files[index2]
            raw_img2 = img2
            if raw_img1 != raw_img2 :
                img2 = cv2.imread(raw_img2)

                #ahash_str2 = aHash(img2)
              #  phash_str2 = pHash(img2)
             #   dhash_str2 = dHash(img2)
             #   a_score = 1 - hammingDist(ahash_str1, ahash_str2) * 1. / (32 * 32 / 4)
           #     p_score = 1 - hammingDist(phash_str1, phash_str2) * 1. / (32 * 32 / 4)
           #     d_score = 1 - hammingDist(dhash_str1, dhash_str2) * 1. / (32 * 32 / 4)
                #n = classify_hist_with_split(img1, img2)
                print(raw_img1)
                print(raw_img2)
                image1 = cv2.resize(img1, (224,224))
                image2 = cv2.resize(img2, (224,224))
                sub_image1 = cv2.split(image1)
                sub_image2 = cv2.split(image2)
                sub_data = 0
                for im1, im2 in zip(sub_image1, sub_image2):
                    sub_data += calculate(im1, im2)
                sub_data = sub_data / 3
                n= sub_data

                print('比較%s與%s之間' % (os.path.basename(raw_img1), os.path.basename(raw_img2)))
                print('三直方圖算法相似度：', n)

             #   print('a_score:{},p_score:{},d_score{}'.format(a_score, p_score, d_score))
                if n > 0.7:
                    remove_list.append(raw_img2)
                    #os.remove( raw_img2)
                    #print('removed %s'% raw_img2)
                    #cv2.imwrite("%s/%stest.png" % (save_path, os.path.basename(raw_img2)), img2)
                    count = count + 1
                    index_count=index

                #else:
                   # count = count+1
    for f in remove_list:
        os.remove(f)
    end = time.time()
    print("Total Spend time：", str((end - start) / 60)[0:6] + "分鐘")
    print(count)
