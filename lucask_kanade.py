from template_matching import *
import argparse
from copy import deepcopy


def imshow(image):
    plt.imshow(image, cmap='gray')
    plt.show()

def translatedImageInv(image, p):
    # M = np.float32([[1, 0, p[0]],
    #                 [0, 1, p[1]]])
    M = np.float32([[1+p[0], p[2], p[4]],
                    [p[1], 1 + p[3],  p[5]]])

    # Affine transformation for 6 parameters, with only x, y translation
    # M = np.float32([[1, 0, p[4]],
    #                 [0, 1, p[5]]])

    M = cv.invertAffineTransform(M)
    if len(image.shape) == 2:
        rows, cols = image.shape
    else:
        rows, cols, _ = image.shape
    translated = cv.warpAffine(image, M, (cols, rows))
    return translated

def makeJacobian(array):
    lambd = lambda x: np.array([[x[0], 0, x[1], 0, 1, 0],
                            [0, x[0], 0, x[1], 0, 1]])

    return np.apply_along_axis(lambd, 1, array)

def lucasKanade(image, template, rect):
    params = np.zeros(6)

    imageCurrent = image.copy()
    x, y, width, height = rect
    rect = (x - 1, y - 1, width, height)

    for i in range(100):
        image = translatedImageInv(imageCurrent, params)
        gy, gx = np.gradient(image)

        gx_w = imcrop(gx, rect)
        gy_w = imcrop(gy, rect)

        candidate = imcrop(image, rect)
        errorImage = (template - candidate)
        errorImageRemapped = np.tile(errorImage.flatten('F'), (len(params), 1)).T

        X, Y = np.meshgrid(range(candidate.shape[0]), range(candidate.shape[1]))
        coords2d = np.array([X.flatten('F')+1, Y.flatten('F')+1]).T
        jacobian = makeJacobian(coords2d)

        steepest = np.asarray([np.asarray(grad).dot(jacobian[i]) for i, grad in enumerate(zip(gx_w.flatten('F'), gy_w.flatten('F')))])
        hessian = steepest.T.dot(steepest)

        costFunction = np.sum(np.multiply(steepest, errorImageRemapped), axis = 0)

        dp = np.linalg.inv(hessian).dot(costFunction.T)
        params = params + dp.T

        if (np.linalg.norm([dp[4], dp[5]])) < 0.1:
            print("success")
            break

        # print("DP - ", np.linalg.norm([dp[4], dp[5]]))

    return params

def getRect(target, window_size):
    return np.array([
        int(np.ceil(target[0])) - window_size[0] // 2,
        int(np.ceil(target[1])) - window_size[1] // 2,
        window_size[0],
        window_size[1]
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', required=True)
    parser.add_argument('-target', nargs='+', type=int, required=True)
    parser.add_argument('-window', nargs='+', type=int, required=True)
    parsed = parser.parse_args()

    (path, target, windowSize) = (parsed.path, parsed.target, parsed.window)
    imagesList = sorted(glob.glob(os.path.join(path, '*.jpg')))
    currentImage = cv.imread(imagesList.pop(0), 0)
    currentImage = np.array(currentImage, dtype='float64')

    rect = getRect(target, windowSize)

    template = imcrop(currentImage, rect=rect)

    image = cv.imread(imagesList[0], 0)
    image = np.array(image, dtype='float64')

    params = lucasKanade(image, template, rect=rect)
    print(params)

    for idx, img in enumerate(imagesList):
        nextImage = cv.imread(img, 0)
        nextImageCopy = cv.imread(img, 0)
        nextImage = np.array(nextImage, dtype='float64')

        params = lucasKanade(nextImage, template, rect=rect)

        x, y, width, height = rect

        affine = np.array([[1 + params[0], params[2], params[4]], [params[1], 1 + params[3], params[5]]])
        target = affine.dot(np.append(np.array([x, y]), 1))

        rect = (int(round(target[0])), int(round(target[1])), width, height)

        template = imcrop(nextImage, rect)

        topLeft = (x, y)
        bottomRight = (x + width, y + height)
        cv.rectangle(nextImageCopy, topLeft, bottomRight, 255, 2)
        cv.imshow("", nextImageCopy)
        cv.waitKey(10)



if __name__== "__main__":
    main()