import os
import cv2 as cv
import numpy as np


def cut_roi(image, roi):
    return image[
           roi[1]:roi[1] + roi[3],
           roi[0]:roi[0] + roi[2]]

def get_window(center, window_size):
    return (int(np.ceil(center[0])) - window_size[0] // 2,
            int(np.ceil(center[1])) - window_size[1] // 2,
            window_size[0],
            window_size[1])

def mean_shift(src, window):
    num_of_iterations = 70
    min_distance = 1
    roi = cut_roi(src, window)
    centroid = np.zeros(2)
    last_iter = 0
    for iter_cnt in range(num_of_iterations):
        new_centroid = np.zeros(2)

        roi = cut_roi(src, get_window(centroid + window[:2], window[2:4]))
        cnt = 0
        h, w = roi.shape

        for i in range(h):
            for j in range(w):
                if roi[i, j]:
                    new_centroid[1] += i
                    new_centroid[0] += j
                    cnt += 1
        new_centroid /= cnt
        if np.linalg.norm(centroid - new_centroid) < min_distance:
            print(iter_cnt, np.linalg.norm(centroid - new_centroid))
            break
        else:
            centroid = new_centroid.copy()
            last_iter = iter_cnt

    print(last_iter)
    return get_window(centroid + window[:2], window[2:4])


def main():
    image_folder = 'img'
    images = [cv.imread(os.path.join(image_folder, img_name))
              for img_name in sorted(os.listdir(image_folder))]
    target_point = np.array([320, 200])
    prev_img = images.pop()
    window_size = [100, 100]
    track_window = (int(np.ceil(target_point[0])) - window_size[0] // 2,
                    int(np.ceil(target_point[1])) - window_size[1] // 2,
                    window_size[0],
                    window_size[1])
    track_window2 = (int(np.ceil(target_point[0])) - window_size[0] // 2,
                    int(np.ceil(target_point[1])) - window_size[1] // 2,
                    window_size[0],
                    window_size[1])
    roi = cut_roi(prev_img, track_window)
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    for img_idx, curr_img in enumerate(images):
        frame = curr_img
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # apply meanshift to get the new location
        ret, track_window2 = cv.meanShift(dst, track_window2, term_crit)
        track_window = mean_shift(dst, track_window)
        # Draw it on image

        x, y, w, h = track_window
        img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

        x, y, w, h = track_window2
        img2 = cv.rectangle(frame, (x, y), (x + w, y + h), [0, 0, 255], 2)
        cv.imshow('img2', img2)
        k = cv.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv.imwrite(chr(k) + ".jpg", img2)
    cv.destroyAllWindows()


if __name__== "__main__":
    main()