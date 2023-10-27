import cv2
color_matrix = cv2.imread("images/small_test.png")
print(color_matrix)
cv2.imshow('color', color_matrix)