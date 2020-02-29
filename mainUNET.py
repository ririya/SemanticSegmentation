import glob
import random
import util
from TrainAndTest import train, test
from Model import unetVGG16, unet
import keras.optimizers as optmizers

random.seed(a=30, version=2)

numClasses = 2
batchSize = 32
input_width = 128
input_height = 128
epochs = 50
optimizer = optmizers.Adam(lr=0.001)
imgNorm = 'divide_constant'
# imgNorm = ''
channel_ordering = 'channels_last'
dropout = 0.3
preTrainedWeights = 'imagenet'
divideConstant = 255
RGB  = True
loss = 'weighted_crossentropy'
modelFN = unetVGG16

params = util.Params(input_width, input_height, numClasses, batchSize, epochs, optimizer, imgNorm,
                     dropout, preTrainedWeights, divideConstant,RGB,loss,modelFN)

percTrain = 0.8
percVal = 0.1
percTest = 0.1

trainFileDir =  '/home/bb-spr/PycharmProjects/InterviewPractice/images'
save_weights_dir = 'weights'

trainFiles = glob.glob(trainFileDir + "/*.png")
nFiles = len(trainFiles)
random.shuffle(trainFiles)

classWeights = util.getClassWeights(trainFiles, params)

valFiles = trainFiles[int(percTrain*nFiles):int((percTrain+percVal)*nFiles)]
testFiles = trainFiles[int((percTrain+percVal)*nFiles):]
trainFiles = trainFiles[:int(percTrain*nFiles)]


results,m = train(params, trainFiles,valFiles, save_weights_dir,classWeights)

test(results, m, params, save_weights_dir,testFiles)


