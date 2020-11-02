import time
import cv2
from matplotlib import pyplot as plt
'''

def match_ORB():
    img1 = cv2.imread('classifiaction/test/3view/8183S-page17.jpg', 0)
    img2 = cv2.imread('classifiaction/test/3view/8183S-page16.jpg', 0)

    # 使用SURF_create特征检测器 和 BFMatcher描述符
    orb = cv2.xfeatures2d.SURF_create(float(3000))
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # matches是DMatch对象，DMatch是以列表的形式表示，每个元素代表两图能匹配得上的点。
    bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # ===========================   输出匹配的坐标  ===================================
    # kp1的索引由DMatch对象属性为queryIdx决定，kp2的索引由DMatch对象属性为trainIdx决定
    # 获取001.png的关键点位置。可以遍历matches[:20]前20个最佳的匹配点
    x, y = kp1[matches[0].queryIdx].pt
    print(x, y)
    cv2.rectangle(img1, (int(x), int(y)), (int(x) + 2, int(y) + 2), (0, 0, 255), 2)
    cv2.imshow('001', img1)
    cv2.waitKey(0)

    # 获取002.png的关键点位置
    x2, y2 = kp2[matches[0].trainIdx].pt
    print(x2, y2)
    cv2.rectangle(img2, (int(x2), int(y2)), (int(x2) + 2, int(y2) + 2), (0, 0, 255), 2)
    cv2.imshow('002', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # ==============================================================================

    # 使用plt将两个图像的第一个匹配结果显示出来
    img3 = cv2.drawMatches(img1=img1, keypoints1=kp1,
                           img2=img2, keypoints2=kp2,
                           matches1to2=matches[:20], outImg=img2,
                           flags=2)
    return img3


if __name__ == '__main__':
    start_time = time.time()
    img3 = match_ORB()
    plt.imshow(img3)
    plt.show()
    end_time = time.time()
    print("Total Spend time：", str((end_time - start_time) / 60)[0:6] + "分钟")
'''
'''
#利用直方圖計算兩者間差異


import matplotlib.pyplot as plt
import cv2
import numpy as py

img = cv2.imread('classifiaction/test/3view/8197S-page15.jpg')
img1 = cv2.imread('classifiaction/test/3view/8197S-page16.jpg')

H1 = cv2.calcHist([img], [1], None, [256], [0, 256])
H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)  

H2 = cv2.calcHist([img1], [1], None, [256], [0, 256])
H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)

# 利用compareHist（）進行相似度比較
similarity = cv2.compareHist(H1, H2, 0)
print(similarity)

# img和img1直方圖展示
plt.subplot(2, 1, 1)
plt.plot(H1)
plt.subplot(2, 1, 2)
plt.plot(H2)
plt.show()
'''

import cv2
import time
import numpy as np
import os
from shutil import copyfile
load_path = r"C:\Users\randy\Downloads\the_way_to_train\no_repeat\3view"
save_path = r"C:\Users\randy\Downloads\the_way_to_train\no_repeat\3view"
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
    start = time.time()
    for r, d, f in os.walk(load_path):
        for file in f:
            if '.png' in file:
                files.append(os.path.join(r, file))

    index_count = 11  # 11
    for index in range(len(files)):
        index2=index+1

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

                #else:
                   # count = count+1

    end = time.time()

    for f in remove_list:
        os.remove(f)
    print("Total Spend time：", str((end - start) / 60)[0:6] + "分鐘")
    print(count)


