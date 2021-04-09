import math
import license_plate_recognition
import numpy as np
import cv2


def get_license_plate(filename):
    # 读文件
    # 灰度化
    origin_image = cv2.imread(filename)
    # 高斯滤波
    gray_image = cv2.imread(filename)[:, :, 0]
    img = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # cv2.imwrite('step1_gaussian.bmp', img)

    # 垂直sobel(y方向)
    sobel_car = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
    # cv2.imwrite('step2_sobel.bmp', sobel_car)

    # 自适应二值化
    ret, binary_car = cv2.threshold(sobel_car, 0, 255, cv2.THRESH_OTSU)
    # cv2.imwrite('step3_binary.bmp', binary_car)

    # 闭合
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 5))
    close_car = cv2.morphologyEx(binary_car, cv2.MORPH_CLOSE, kernelX, iterations=1)
    # cv2.imwrite('step4_close.bmp', close_car)

    # 中值滤波
    image = cv2.medianBlur(close_car, 15)
    # cv2.imwrite('step5_dilate_erode.bmp', image)

    # 轮廓检测
    # cv2.RETR_EXTERNAL表示只检测外轮廓
    # cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    image1 = origin_image.copy()
    cv2.drawContours(image1, contours, -1, (0, 0, 0), 2)
    # cv2.imwrite('step6_contours.bmp', image1)

    # 筛选
    cnt = 0
    for i, item in enumerate(contours):
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中
        # cv2.boundingRect用一个最小的矩形，把找到的形状包起来
        rect = cv2.boundingRect(item)
        # 左上角坐标
        x = rect[0]
        y = rect[1]
        # 宽和高
        weight = rect[2]
        height = rect[3]
        if (weight > (height * 2)) and (weight < (height * 5)) and height > 10:
            cnt = cnt + 1
            index = i
            image2 = origin_image.copy()
            cv2.drawContours(image2, contours, index, (255, 0, 0), 2)
            # cv2.imwrite('step7_' + str(cnt) + '_license_plate.bmp', image2)

    cnt = contours[index]
    image3 = origin_image.copy()
    h, w = image3.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    k = vy / vx
    b = y - k * x
    lefty = int(b)
    righty = int(k * w + b)
    img = cv2.line(image3, (w, righty), (0, lefty), (255, 0, 0), 2)
    cv2.imwrite('step0_line.bmp', img)

    a = math.atan(k)
    a = math.degrees(a)
    image4 = origin_image.copy()
    # 图像旋转
    h, w = image4.shape[:2]
    print(h, w)
    # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1)
    # 第三个参数：变换后的图像大小
    dst = cv2.warpAffine(image4, M, (int(w * 1), int(h * 1)))
    cv2.imwrite('step0_car.bmp', dst)


filename = '5.bmp'
get_license_plate(filename)
license_plate_recognition.get_license_plate('step0_car.bmp')
