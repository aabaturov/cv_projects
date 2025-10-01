import cv2
import numpy as np
import pickle
import os
import json


def find_triangles(image_path, number):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_EXTERNAL | cv2.RETR_LIST
    #with open(filepath, 'wb') as f:
    #    pickle.dump([], f)
    json_data = {}
    label_num = 1
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 3:
            area = cv2.contourArea(approx)
            if area < 100 or area > 300:
                continue  # Пропускаем неподходящие по размеру треугольники
            
             # Проверка на равносторонность
            pts = approx.reshape(3, 2)
            a = np.linalg.norm(pts[0] - pts[1])
            b = np.linalg.norm(pts[1] - pts[2])
            c = np.linalg.norm(pts[2] - pts[0])
            max_side = max(a, b, c)
            min_side = min(a, b, c)
            if max_side - min_side > 5:  # Порог допустимой разницы в длинах сторон
                continue  # Не равносторонний — пропускаем
            
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)
            
            pts = approx.reshape(3, 2)  # получаем массив с тремя точками
            triangle_dict = {}

            for i, (x, y) in enumerate(pts):
                triangle_dict[f"x{i}"] = int(x)
                triangle_dict[f"y{i}"] = int(y)

            json_data[f"triangle_{label_num}"] = triangle_dict
            label_num += 1
                        
    # Показываем изображение (по желанию)
    cv2.imshow('Detected Triangles' + str(number), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

find_triangles("pr5_result/pr9/data/image/image_11.png", 11)
#print("Обнаруженные треугольники сохранены в изображении.")