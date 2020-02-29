import random
import itertools
import numpy as np
import cv2

def readImage(path, params):

    if params.RGB:
        img = cv2.imread(path, 1)
    else:
        img = cv2.imread(path, 0)

    img = cv2.resize(img, (params.input_width, params.input_height))
    img = img.astype(np.float32)
    img = normalizeImg(img, params)

    if not params.RGB:
        img = np.expand_dims(img, axis=-1)

    return img

def readMask(path, params):

    scale = 255/(params.numClasses-1)

    img = cv2.imread(path, 0)
    img = cv2.resize(img, (params.input_width, params.input_height))
    img = img.astype(np.float32)

    img /= scale
    img = np.round(img)

    if params.loss == 'binary_crossentropy':
        return np.expand_dims(img, axis=-1)
    else:
        mask = np.zeros((img.shape[0], img.shape[1], params.numClasses))

        for c in range(params.numClasses):
            currMask = np.zeros((img.shape[0], img.shape[1]))
            ind = np.where(img == c)
            currMask[ind] = 1
            mask[:,:,c] = currMask

        return mask


def normalizeImg(img, params):

    if params.imgNorm.find('sub_mean') > -1:
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68

    elif params.imgNorm == 'sub_global_mean':
        img -= params.globalMean

    elif params.imgNorm == 'whiten_global':
        img -= params.globalMean
        img /= params.globalStd

    if params.imgNorm.find('divide_constant') > -1:

        img = img / params.divideConstant

    return img

def randomizeList(sampleList, Train = True):

    if Train:
        random.shuffle(sampleList)
    cycleList = itertools.cycle(sampleList)

    return cycleList

def generator(sampleList, params, Train = True):

    epoch = 1

    listLen = len(sampleList)

    cycleList = randomizeList(sampleList, Train)

    count = 0

    while True:

        X_batch = []
        Y_batch = []

        nSamplesBatch = 0

        while nSamplesBatch < params.batch_size:

            samplePath = next(cycleList)

            targetPath = samplePath.replace('images', 'masks')

            X_batch.append(readImage(samplePath,params))
            Y_batch.append(readMask(targetPath, params))

            count += 1

            if count >= listLen:

                epoch = epoch + 1

                cycleList = randomizeList(sampleList, Train)

                count = 0

            nSamplesBatch += 1

        X = np.array(X_batch)


        Y = np.array(Y_batch)


        yield X, Y