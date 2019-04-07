from neural_network import minibatch_gd, test_nn
import numpy as np

def init_weights(d, dp):
    return 0.01 * np.random.uniform(0.0, 1.0, (d, dp)), np.zeros(dp)

if __name__ == '__main__':
    x_train = np.load("data/x_train.npy")
    x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    y_train = np.load("data/y_train.npy")

    x_test = np.load("data/x_test.npy")
    x_test = (x_test - np.mean(x_test, axis=0))/np.std(x_test, axis=0)
    y_test = np.load("data/y_test.npy")

    load_weights = True #set to True if you want to use saved weights

    if load_weights:
        w1 = np.load('weights/w1.npy')
        w2 = np.load('weights/w2.npy')
        w3 = np.load('weights/w3.npy')
        w4 = np.load('weights/w4.npy')

        b1 = np.load('weights/b1.npy')
        b2 = np.load('weights/b2.npy')
        b3 = np.load('weights/b3.npy')
        b4 = np.load('weights/b4.npy')
        print("Loaded saved weights.")

    else:
        w1, b1 = init_weights(784, 256)
        w2, b2 = init_weights(256, 256)
        w3, b3 = init_weights(256, 256)
        w4, b4 = init_weights(256, 10)
        print("Initialized new weights.")

    # w1, w2, w3, w4, b1, b2, b3, b4, losses = minibatch_gd(30, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, 10)
    np.save('weights/w1.npy', w1)
    np.save('weights/w2.npy', w2)
    np.save('weights/w3.npy', w3)
    np.save('weights/w4.npy', w4)

    np.save('weights/b1.npy', b1)
    np.save('weights/b2.npy', b2)
    np.save('weights/b3.npy', b3)
    np.save('weights/b4.npy', b4)

    avg_class_rate, class_rate_per_class = test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, 10)

    print(avg_class_rate, class_rate_per_class)
