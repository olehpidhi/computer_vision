import glob
from matplotlib import pyplot as plt
import os
import cv2 as cv
import numpy as np
import sys


def imcrop(img, rect):
   x, y, width, height = rect
   return img[y:y+height, x:x+width]

def ssd(A,B):
    squares = (A[:,] - B[:,]) ** 2
    return np.sum(squares)

def sad(A,B):
    diff = (A[:,] - B[:,])
    return np.sum(diff)

def ncc(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

def templateMatch(image, template, method = 'ssd'):
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    grayTemplate = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    if method == 'ncc':
        minError = 0
    else:
        minError = sys.maxsize

    position = (0, 0)
    for y in range(grayImage.shape[0] - grayTemplate.shape[0]):
        for x in range(grayImage.shape[1] - grayTemplate.shape[1]):
            cropped = imcrop(grayImage, (x, y, grayTemplate.shape[1], grayTemplate.shape[0]))

            if method == 'ssd':
                error = ssd(cropped, grayTemplate)
            elif method == 'sad':
                error = sad(cropped, grayTemplate)
            elif method == 'ncc':
                error = ncc(cropped, grayTemplate)

            if method == 'ncc':
                if error > minError:
                    if error == 1:
                        return (x, y)
                    minError = error
                    position = (x, y)
            else:
                if error < minError:
                    if error == 0:
                        return (x, y)
                    minError = error
                    position = (x, y)
    return position

if __name__ == '__main__':
    image = cv.imread(os.path.join("img", "0001.jpg"))
    rect = cv.selectROI(image)
    cv.destroyAllWindows()

    croppedImage = imcrop(image, rect)

    _, width, height = croppedImage.shape[::-1]
    x, y = templateMatch(image, croppedImage, "sad")

    topLeft = (x, y)
    bottomRight = (x + width, y + height)

    imageToShow = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    cv.rectangle(imageToShow, topLeft, bottomRight, 255, 2)

    plt.subplot(121), plt.imshow(imageToShow)
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv.cvtColor(croppedImage, cv.COLOR_BGR2RGB))
    plt.title('Template Image'), plt.xticks([]), plt.yticks([])

    plt.show()
    cv.waitKey()