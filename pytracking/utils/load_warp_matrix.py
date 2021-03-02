import cv2 as cv

def load_warp_matrix(name, num):
    fn = '/home/gyz/dataset/otb100/{}_warp.yml'.format(name)
    fs = cv.FileStorage(fn, cv.FileStorage_READ)
    warp_matrix = []
    for i in range(2, num):
        w = fs.getNode('warp{}'.format(i)).mat()
        warp_matrix.append(w)
    fs.release()

    return warp_matrix