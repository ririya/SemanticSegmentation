import cv2
import numpy as np

class Params:
    def __init__(self, input_width, input_height, numClasses, batch_size,
             epochs, optimizer, imgNorm, dropout, preTrainedWeights, divideConstant,RGB,loss,modelFN):

        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.numClasses = numClasses
        self.batch_size = int(batch_size)
        self.epochs = epochs
        self.optimizer = optimizer
        self.imgNorm = imgNorm
        self.preTrainedWeights = preTrainedWeights
        self.dropout = dropout
        self.divideConstant = divideConstant
        self.RGB = RGB
        self.loss = loss
        self.modelFN = modelFN


def getClassWeights(sampleList, params):

    classWeights = np.zeros((params.numClasses,1))

    for samplePath in sampleList:
        targetPath = samplePath.replace('images', 'masks')

        scale = 255 / (params.numClasses - 1)

        img = cv2.imread(targetPath, 0)
        img = cv2.resize(img, (params.input_width, params.input_height))
        img = img.astype(np.float32)

        img /= scale
        img = np.round(img)

        for c in range(params.numClasses):
            ind = np.where(img == c)
            classWeights[c] += len(ind[0])

    maxWeight = np.max(classWeights)
    classWeights = maxWeight/classWeights
    classWeights /= np.sum(classWeights)

    return classWeights

