import cv2
import numpy as np
import os

# Пути
image_path = 'pr5_result/pr9/data/image/image_11.png'
template_dir = 'pr5_result/pr9/data/template'
threshold = 0.6  # Порог совпадения

# Загрузка изображения
image = cv2.imread(image_path)
original = image.copy()
gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Создание пирамиды изображения
image_pyramid = [gray_orig]
for i in range(3):  # глубина пирамиды
    gray = cv2.pyrDown(image_pyramid[-1])
    image_pyramid.append(gray)

# Поиск шаблонов
for template_name in os.listdir(template_dir):
    template_path = os.path.join(template_dir, template_name)
    template = cv2.imread(template_path)
    if template is None:
        continue
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    for level in reversed(range(len(image_pyramid))):
        resized = image_pyramid[level]
        scale = image.shape[1] / float(resized.shape[1])

        result = cv2.matchTemplate(resized, template_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):
            top_left = (int(pt[0] * scale), int(pt[1] * scale))
            bottom_right = (int((pt[0] + w) * scale), int((pt[1] + h) * scale))
            cv2.rectangle(original, top_left, bottom_right, (0, 255, 0), 2)

# Отображение результата
cv2.imshow("Matches", original)
cv2.waitKey(0)
cv2.destroyAllWindows()