from neural_network import minibatch_gd, test_nn, four_nn
import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def init_weights(d, dp):
    return 0.01 * np.random.uniform(0.0, 1.0, (d, dp)), np.zeros(dp)

def main():

    x_train = np.load("data/x_train.npy")
    x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    y_train = np.load("data/y_train.npy")

    x_test = np.load("data/x_test.npy")
    x_test = (x_test - np.mean(x_test, axis=0))/np.std(x_test, axis=0)
    y_test = np.load("data/y_test.npy")

    np.random.seed(0)
    w1, b1 = init_weights(784, 256)
    w2, b2 = init_weights(256, 256)
    w3, b3 = init_weights(256, 256)
    w4, b4 = init_weights(256, 10)
    print("Initialized new weights.")
    epoch_num = 10
    start_t = time.time()
    w1, w2, w3, w4, b1, b2, b3, b4, losses = minibatch_gd(epoch_num, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, 10)
    total_time = time.time() - start_t

    classification = four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, test=True)

    class_names = np.array(["T-shirt/top","Trouser","Pullover","Dress",
            "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])
    cm = confusion_matrix(y_test, classification)
    classAccuracy = np.zeros((len(np.unique(y_test)),))
    for i in range(len(classAccuracy)):
        classAccuracy[i] = cm[i][i]/np.sum(cm[i,:])
    def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        classAccuracy = np.zeros((len(np.unique(y_true)),))
        for i in range(len(classAccuracy)):
            classAccuracy[i] = cm[i][i]/np.sum(cm[i,:])
        print("The accuracy of each class is {}".format(classAccuracy))
        # f = open("ClassficationRate.csv", "w")
        # for i in classAccuracy:
        #     f.write("{}\n".format(i))
        # f.write("{}\n".format(np.average(classAccuracy)))
        # f.close()
        # Only use the labels that appear in the data
        temp = unique_labels(y_true, y_pred)
        classes = classes[temp]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    plot_confusion_matrix(y_test, classification, classes=class_names, normalize=True, title="Confusion Matrix - {} Epochs".format(epoch_num), cmap=plt.cm.Blues)
    plt.savefig("{}_epoch_confusion_matrix.png".format(epoch_num))

    avg_class_rate = np.mean(classAccuracy)
    print("Average clssification rate: {:.2f}%".format(avg_class_rate * 100))
    print("Training time {:.2f} s, {:.2f} s/epoch".format(total_time, total_time/epoch_num))

if __name__ != '__main__':
    start_t = time.time()
    x_train = np.load("data/x_train.npy")
    x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    y_train = np.load("data/y_train.npy")

    x_test = np.load("data/x_test.npy")
    x_test = (x_test - np.mean(x_test, axis=0))/np.std(x_test, axis=0)
    y_test = np.load("data/y_test.npy")

    np.random.seed(0)
    w1, b1 = init_weights(784, 256)
    w2, b2 = init_weights(256, 256)
    w3, b3 = init_weights(256, 256)
    w4, b4 = init_weights(256, 10)
    print("Initialized new weights.")

    epoch_num = 10
    w1, w2, w3, w4, b1, b2, b3, b4, losses = minibatch_gd(50, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, 10)

    np.save('losses.npy', losses)

    avg_class_rate, class_rate_per_class = test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, 10)
    total_time = time.time() - start_t
    print("Average clssification rate: {:.2f}%".format(avg_class_rate * 100))
    print(class_rate_per_class)
    print("Total time is {:.2f} s, Average time is {:.2f} s".format(total_time, total_time/epoch_num))

    # np.save('weights/w1_50.npy', w1)
    # np.save('weights/w2_50.npy', w2)
    # np.save('weights/w3_50.npy', w3)
    # np.save('weights/w4_50.npy', w4)

    # np.save('weights/b1_50.npy', b1)
    # np.save('weights/b2_50.npy', b2)
    # np.save('weights/b3_50.npy', b3)
    # np.save('weights/b4_50.npy', b4)


main()

# losses = np.load('losses.npy')
# plt.plot(np.arange(len(losses)), losses, 'k.')
# plt.xlabel("Epoch Number")
# plt.ylabel("Loss")
# plt.title("Loss of the Model Over 50 Epochs")
# plt.savefig("Losses_vs_epoch.png")
