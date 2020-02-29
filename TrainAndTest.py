from lossesAndMetrics import *
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import Model
import imageLoading
import matplotlib.pyplot as plt

def train(params, trainFiles,valFiles, save_weights_dir,classWeights):
    metrics = ['accuracy']
    for classId in range(params.numClasses):
        metrics.append(single_class_accuracy(classId))
        # metrics.append(classAcc(classId))
        # metrics.append(accuracyMask(classId))
        # metrics.append(single_class_accuracy_Alt(classId))

    metrics.append(conf_matrix_diag_acc(params.numClasses, params.batch_size))

    curr_save_weights_path = save_weights_dir + '/weights-{epoch:02d}.hdf5'

    callbacks =   [EarlyStopping(patience=10, verbose=1),
                   ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
        ModelCheckpoint(curr_save_weights_path, monitor='val_loss', verbose=1, save_best_only=False, mode='max')]

    m = params.modelFN(params)

    if params.loss == 'weighted_crossentropy':
        m.compile(loss=weighted_categorical_crossentropy(classWeights, params.numClasses), optimizer=params.optimizer, metrics=metrics)
    else:
        m.compile(loss=params.loss, optimizer=params.optimizer, metrics=metrics)

    G = imageLoading.generator(trainFiles, params, Train=True)
    G2 = imageLoading.generator(valFiles,params,  Train=False)

    numberBatches = len(trainFiles) / params.batch_size

    validation_steps = len(valFiles) / params.batch_size

    results = m.fit_generator(G, numberBatches, verbose=2, validation_data=G2, validation_steps=validation_steps,
                           epochs=params.epochs, callbacks=callbacks)

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r",
             label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.savefig('LearningCurve.png')

    return (results,m)

def test(results, m, params, save_weights_dir,testFiles):
    bestEpoch = np.argmin(results.history["val_loss"])
    curr_save_weights_path = save_weights_dir + '/weights-' + "{:02d}".format(bestEpoch + 1) + '.hdf5'
    m.load_weights(curr_save_weights_path)

    G = imageLoading.generator(testFiles, params, Train=False)

    scores = m.evaluate_generator(G, len(testFiles)/ params.batch_size)

    print(scores)