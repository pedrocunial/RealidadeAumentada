import cv2
import numpy as np


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

_, frame = cap.read()
orig_shape = (frame.shape[:2])[::-1]

aruco_img = cv2.imread('marcadores/board_aruco.png')
aruco_img = cv2.cvtColor(aruco_img, cv2.COLOR_BGR2RGB)
height, width, _ = aruco_img.shape
orig_corners, orig_ids, _ = cv2.aruco.detectMarkers(aruco_img, dictionary)

waifu = cv2.imread('marcadores/mako.png')
waifu = cv2.resize(waifu.transpose(1, 0, 2), (width, height))

white = np.zeros((height, width, 3), dtype=np.uint8)
white[::] = 255


def match_corners(ids, orig_corners=orig_corners, orig_ids=orig_ids):
    result = []
    for i in ids:
        idx, _ = np.where(orig_ids == i[0])
        result.append(orig_corners[idx[0]])
    return np.array(result)


def merge(orig, final):
    height, width = orig.shape
    res = orig.copy()
    for i in range(height):
        for j in range(width):
            if not final[i, j].all(0):
                res[i, j] = final[i, j]
    return res


def good_merge(orig, final):
    return np.clip(final * 255 + orig, 0, 255).astype(np.uint8)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    corners, ids, rejected = cv2.aruco.detectMarkers(frame, dictionary)

    if ids is not None:
        orig = match_corners(ids)
        dest = np.array(corners)
        orig = orig.reshape(-1, orig.shape[-1])
        dest = dest.reshape(-1, dest.shape[-1])
        homography, _ = cv2.findHomography(orig, dest)
        final = cv2.warpPerspective(white, homography, orig_shape)
        curr_waifu = cv2.warpPerspective(waifu, homography, orig_shape)
        mask = cv2.bitwise_not(final)
        # final = good_merge(img, final)
        res = (mask & frame) | curr_waifu
        cv2.imshow('frame', res)
    else:
        # Display the resulting frame
        img = cv2.aruco.drawDetectedMarkers(frame, corners)
        cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
