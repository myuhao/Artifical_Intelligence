import numpy as np

"""
    Minigratch Gradient Descent Function to train model
    1. Format the data
    2. call four_nn function to obtain losses
    3. Return all the weights/biases and a list of losses at each epoch
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n,d) numpy array where d=number of features
        y_train (np array) - (n,) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
    Hints:
        Should work for any number of features and classes
        Good idea to print the epoch number at each iteration for sanity checks!
        (Stdout print will not affect autograder as long as runtime is within limits)
"""
def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=True):
    # Number of samples in the training set.
    num_sample = len(y_train)
    # Batch size.
    batch_size = 200

    losses = []
    for e in range(epoch):
        print("epoch {} starts.".format(e))
        # Shuffle data
        if shuffle == True:
            # Creating random index to shuffle the training set.
            p = np.random.permutation(num_sample)
            x_train = x_train[p,:]
            y_train = y_train[p]
        loss = 0
        for i in range(int(num_sample/batch_size)):
            x = x_train[i*batch_size:(i+1)*batch_size,:]
            y = y_train[i*batch_size:(i+1)*batch_size]
            loss += four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x, y, test=False)
        losses.append(loss)
    return w1, w2, w3, w4, b1, b2, b3, b4, losses

"""
    Use the trained weights & biases to see how well the nn performs
        on the test data
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
    Hints:
        Good place to show your confusion matrix as well.
        The confusion matrix won't be autograded but necessary in report.
"""
def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):
    y_test = y_test
    classification = four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, test=True)
    # Length = number of sample, 1 if correctly classified, 0 otherwise.
    avg_class_rate = np.zeros(classification.shape)
    avg_class_rate[classification==y_test] = 1
    avg_class_rate = np.sum(avg_class_rate)/len(y_test)
    # Length = num_classes. ith_element++ if correctly labeld as class i.
    sample_per_class = np.zeros((num_classes,))
    class_rate_per_class = np.zeros((num_classes,))
    for i in y_test:
        sample_per_class[i] += 1
    for i in classification[classification==y_test]:
        class_rate_per_class[i] += 1
    class_rate_per_class /= sample_per_class
    return avg_class_rate, class_rate_per_class

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x, y, test):
    Z1, acache1 = affine_forward(x, w1, b1)
    A1, rcache1 = relu_forward(Z1)
    Z2, acache2 = affine_forward(A1, w2, b2)
    A2, rcache2 = relu_forward(Z2)
    Z3, acache3 = affine_forward(A2, w3, b3)
    A3, rcache3 = relu_forward(Z3)
    F, acache4 = affine_forward(A3, w4, b4)

    if test == True:
        return np.argmax(F, axis=1)

    loss, dF = cross_entropy(F, y)
    dA3, dw4, db4 = affine_backward(dF, acache4)
    dZ3 = relu_backward(dA3, rcache3)
    dA2, dw3, db3 = affine_backward(dZ3, acache3)
    dZ2 = relu_backward(dA2, rcache2)
    dA1, dw2, db2 = affine_backward(dZ2, acache2)
    dZ1 = relu_backward(dA1, rcache1)
    dX, dw1, db1 = affine_backward(dZ1, acache1)

    eta = 0.1
    w1 -= eta*dw1
    w2 -= eta*dw2
    w3 -= eta*dw3
    w4 -= eta*dw4
    b1 -= eta*db1
    b2 -= eta*db2
    b3 -= eta*db3
    b4 -= eta*db4
    return loss

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""
def affine_forward(A, W, b):
    '''
    Arguments:
        A {np.ndarray} -- data with size (n,d)
        W {np.ndarray} -- weights with size (d,d')
        b {np.ndarray} -- bias with size (d',)
    Returns:
        Z      -- affine output with size (n,d')
        cache  -- tuple of the original inputs
    '''
    Z = A @ W + b
    cache = (A, W, b)
    return Z, cache

def affine_backward(dZ, cache):
    '''
    Arguments:
        dZ {np.ndarray} -- gradient of Z
        cache {tuple}   -- cache of the forward operation
    Returns:
        dA, dW, dB -- gradients with respect to loss
    '''
    A = cache[0]
    W = cache[1]
    b = cache[2]
    dA = dZ @ W.T
    dW = A.T @ dZ
    dB = np.sum(dZ, axis=0)
    return dA, dW, dB

def relu_forward(Z):
    return np.clip(Z, 0, np.inf), Z

def relu_backward(dA, cache):
    mask = np.clip(cache, 0, np.inf)
    mask[np.where(mask != 0)] = 1
    return dA * mask

def cross_entropy(F, y):
    """
    Arguments:
        F {np.ndarray} -- logits with size (n, num_classes)
        y {np.ndarray} -- (actual class label of data with size (n,)
    Returns:
        loss, dF -- gradient of the logits
    """
    n = F.shape[0]
    exp_F = np.exp(F)
    # The ith element of Fiyi is F[i, true_label_idx_of_row_i]
    # Fiyi.shape = (n,)
    Fiyi = F[np.arange(n), y.astype(int)]
    # Sum all scores of the ith row.
    log_sigma_exp_Fik = np.log(np.sum(exp_F, axis=1))
    loss = np.sum(Fiyi - log_sigma_exp_Fik)
    loss /= -n
    # The y[i]-th element of i-th row in dF is 1, 0 otherwise.
    dF = np.zeros(F.shape)
    dF[np.arange(n), y.astype(int)] = 1
    # (10,3) divides (10,1) to get the probability.
    dF -= exp_F / np.sum(exp_F, axis=1).reshape(n,1)
    dF /= -n
    return loss, dF
