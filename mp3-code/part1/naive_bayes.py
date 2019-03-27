import numpy as np

class NaiveBayes(object):
    def __init__(self,num_class,feature_dim,num_value):
        """Initialize a naive bayes model.

        This function will initialize prior and likelihood, where
        prior is P(class) with a dimension of (# of class,)
            that estimates the empirical frequencies of different classes in the training set.
        likelihood is P(F_i = f | class) with a dimension of
            (# of features/pixels per image, # of possible values per pixel, # of class),
            that computes the probability of every pixel location i being value f for every class label.

        Args:
            num_class(int): number of classes to classify
            feature_dim(int): feature dimension for each example (784 in this case)
            num_value(int): number of possible values for each pixel (0 to 255)
        """

        self.num_value = num_value
        self.num_class = num_class
        self.feature_dim = feature_dim

        self.prior = np.zeros((num_class))
        self.likelihood = np.zeros((feature_dim,num_value,num_class))

    def train(self, train_set, train_label):
        """ Train naive bayes model (self.prior and self.likelihood) with training dataset.
            self.prior(numpy.ndarray): training set class prior (in log) with a dimension of (# of class,),
            self.likelihood(numpy.ndarray): traing set likelihood (in log) with a dimension of
                (# of features/pixels per image, # of possible values per pixel, # of class).
            You should apply Laplace smoothing to compute the likelihood.

        Args:
            train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
            train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
        """

        # YOUR CODE HERE

        #estimate prior

        train_size = len(train_label)
        for label in train_label:
            for i in range(10):
                if label == i:
                    self.prior[i] += 1
                    break
        num_samples = self.prior
        self.prior = self.prior/train_size # get percentage
        self.prior = np.log(self.prior) # take log

        #estimate liklihood
        print(len(train_set) == len(train_label))
        for i in range(len(train_set)):
            image = train_set[i]
            curr_label = train_label[i]
            for j in range(len(image)):
                self.likelihood[j][image[j]][curr_label] += 1

        #Laplacian Smoothing
        k = 1
        n_k = self.num_class * k
        self.likelihood += k
        num_samples += n_k
        for a in range(self.feature_dim):
            for b in range(self.num_value):
                for c in range(self.num_class):
                    self.likelihood[a][b][c] /= num_samples[c]

        self.likelihood = np.log(self.likelihood)

    def test(self,test_set,test_label):
        """ Test the trained naive bayes model (self.prior and self.likelihood) on testing dataset,
            by performing maximum a posteriori (MAP) classification.
            The accuracy is computed as the average of correctness
            by comparing between predicted label and true label.

        Args:
            test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
            test_label(numpy.ndarray): testing labels with a dimension of (# of examples, )

        Returns:
            accuracy(float): average accuracy value
            pred_label(numpy.ndarray): predicted labels with a dimension of (# of examples, )
        """

        # YOUR CODE HERE

        accuracy = 0
        pred_label = np.zeros((len(test_set)))

        for i in range(len(test_set)):
            test_img = test_set[i]
            classes = np.zeros(self.num_class)
            for j in range(self.num_class):
                classes[j] += self.prior[j]
                for k in range(self.feature_dim):
                    classes[j] += self.likelihood[k][test_img[k]][j]
            argmax = -float("inf")
            max_idx = 0
            for a in range(self.num_class):
                if classes[a] > argmax:
                    argmax = classes[a]
                    max_idx = a

            pred_label[i] = max_idx
            if max_idx == test_label[i]:
                accuracy += 1

        accuracy /= len(test_set)

        return accuracy, pred_label


    def save_model(self, prior, likelihood):
        """ Save the trained model parameters
        """

        np.save(prior, self.prior)
        np.save(likelihood, self.likelihood)

    def load_model(self, prior, likelihood):
        """ Load the trained model parameters
        """

        self.prior = np.load(prior)
        self.likelihood = np.load(likelihood)

    def intensity_feature_likelihoods(self, likelihood):
        """
        Get the feature likelihoods for high intensity pixels for each of the classes,
            by sum the probabilities of the top 128 intensities at each pixel location,
            sum k<-128:255 P(F_i = k | c).
            This helps generate visualization of trained likelihood images.

        Args:
            likelihood(numpy.ndarray): likelihood (in log) with a dimension of
                (# of features/pixels per image, # of possible values per pixel, # of class)
        Returns:
            feature_likelihoods(numpy.ndarray): feature likelihoods for each class with a dimension of
                (# of features/pixels per image, # of class)
        """
        # YOUR CODE HERE


        feature_likelihoods = np.zeros((likelihood.shape[0],likelihood.shape[2]))
        #print(feature_likelihoods.shape)
        split = likelihood[:,-128:,:]
        #print(feature_likelihoods.shape)
        feature_likelihoods = np.sum(split,axis=1)
        #print(feature_likelihoods.shape)

        return feature_likelihoods
