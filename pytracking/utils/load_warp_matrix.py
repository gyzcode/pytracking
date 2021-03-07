import cv2 as cv

def load_warp_matrix(name):
    fn = '/home/gyz/dataset/otb100/{}_warp.yml'.format(name)
    fs = cv.FileStorage(fn, cv.FileStorage_READ)
    warp_matrix = []
    i = 2
    while True:
        w = fs.getNode('warp{}'.format(i)).mat()
        if w is None:
            break
        else:
            warp_matrix.append(w)
            i = i + 1
    fs.release()

    return warp_matrix