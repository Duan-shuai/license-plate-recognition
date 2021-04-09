import numpy as np
import cv2


def get_license_plate(filename):
    # 读文件
    # 灰度化
    origin_image = cv2.imread(filename)
    # 高斯滤波
    gray_image = cv2.imread(filename)[:, :, 0]
    img = cv2.GaussianBlur(gray_image, (5, 5), 0)
    cv2.imwrite('step1_gaussian.bmp', img)

    # 垂直sobel(y方向)
    sobel_car = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
    cv2.imwrite('step2_sobel.bmp', sobel_car)

    # 边缘提取
    # edges = cv2.Canny(sobel_car, 10, 50)
    # cv2.imwrite('step2.bmp', edges)

    # 自适应二值化
    ret, binary_car = cv2.threshold(sobel_car, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite('step3_binary.bmp', binary_car)

    # 闭合
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 5))
    close_car = cv2.morphologyEx(binary_car, cv2.MORPH_CLOSE, kernelX, iterations=1)
    cv2.imwrite('step4_close.bmp', close_car)

    # 中值滤波
    image = cv2.medianBlur(close_car, 15)
    cv2.imwrite('step5_dilate_erode.bmp', image)

    # 轮廓检测
    # cv2.RETR_EXTERNAL表示只检测外轮廓
    # cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    image1 = origin_image.copy()
    cv2.drawContours(image1, contours, -1, (0, 0, 0), 2)
    cv2.imwrite('step6_contours.bmp', image1)

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
            cv2.imwrite('step7_' + str(cnt) + 'multi_license_plate.bmp', image2)


filename = '4.bmp'
get_license_plate(filename)