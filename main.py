
import cv2


def main():
    cam = cv2.VideoCapture(0)
    result, image = cam.read()

    if result:
        cv2.imshow('Camera', image)
        cv2.waitKey(0)
    else:
        print('Result is False')


if __name__ == '__main__':
    main()
