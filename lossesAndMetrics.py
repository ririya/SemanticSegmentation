import keras.backend as K
import numpy as np


def weighted_categorical_crossentropy(weights, numClasses):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred)

        totalLoss = 0

        for c in range(numClasses):
            totalLoss += loss[:,:,:,c] * weights[c]
        loss = -totalLoss
        return loss

    return loss

def single_class_accuracy_Alt(class_id):
    def fnAlt(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        accuracy_mask = K.cast(K.equal(class_id_preds, class_id), K.floatx())
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), K.floatx()) * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return fnAlt



def classAcc(class_id):
    def class_acc(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        accuracy_mask = K.cast(K.equal(class_id_true, class_id), K.floatx())
        class_acc_tensor1 = K.cast(K.equal(class_id_true, class_id_preds), K.floatx()) * accuracy_mask
        class_acc = K.sum(class_acc_tensor1)
        return class_acc
    return class_acc


def accuracyMask(class_id):
    def accuracy_mask(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        accuracy_mask1 = K.cast(K.equal(class_id_true, class_id), K.floatx())
        acc_sum = K.maximum(K.sum(accuracy_mask1), 1)
        return acc_sum
    return accuracy_mask


def single_class_accuracy(class_id):
    def fn(y_true, y_pred):
        # class_id_true = K.argmax(y_true, axis=-1)
        # class_id_preds = K.argmax(y_pred, axis=-1)
        # accuracy_mask = K.cast(K.equal(class_id_true, class_id), K.floatx())
        # class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), K.floatx()) * accuracy_mask
        # class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        # return class_acc
        classAccFn = classAcc(class_id)
        class_acc = classAccFn(y_true, y_pred)

        accuracyMaskFn = accuracyMask(class_id)
        accuracy_mask = accuracyMaskFn(y_true, y_pred)

        return class_acc / accuracy_mask

    return fn



def conf_matrix_diag_acc(numClasses, batch_size):
    def fn_conf_matrix_diag_acc(y_true, y_pred):

        val = np.zeros((batch_size,numClasses))
        class_acc_total = K.variable(value=val, dtype='float32')

        for c in range(numClasses):

            curr_fn = single_class_accuracy(c)
            class_acc = curr_fn(y_true,y_pred)
            class_acc_total = class_acc_total + class_acc

        return class_acc_total/numClasses
    return fn_conf_matrix_diag_acc





