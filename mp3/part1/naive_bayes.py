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
		    feature_dim(int): feature dimension for each example
		    num_value(int): number of possible values for each pixel
		"""

		self.num_value = num_value
		self.num_class = num_class
		self.feature_dim = feature_dim

		self.prior = np.zeros((num_class))
		self.likelihood = np.zeros((feature_dim,num_value,num_class))

		self.prior_filename = "prior"
		self.likelihood_filename = "likelihood"

	def train(self,train_set,train_label):
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
		self.prior = np.zeros(self.prior.shape)
		self.likelihood = np.zeros(self.likelihood.shape)

		# Calculate the prior based on apperance in the train_label data (50000, ) set.
		# Normalize to [0,1]
		for label in train_label:
			self.prior[label] += 1
		self.prior /= np.sum(self.prior)

		# Calculte the likelihood matrix.
		# Dim:
		# 	0 - (728,) Which pixel we are look at.
		# 	1 -	(256,) Which value is being tested.
		# 	2 - (12,)  Which class is being tested.
		for pictureIdx in range(len(train_set)): #[0,50000], each picture.
			for pixValIdx in range(len(train_set[pictureIdx])): # [0,728], pixVal
				# Number of times pixel pixValIdx has value f in training examples from this class
				self.likelihood[pixValIdx, train_set[pictureIdx][pixValIdx], train_label[pictureIdx]] += 1
		k = 1 # laplace smoothing
		v = 256 # Number of possible values
		self.likelihood += k
		self.likelihood /= (len(train_set) + k*v)


		self.save_model(self.prior_filename, self.likelihood_filename)

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

		pass

		self.load_model(self.prior_filename, self.likelihood_filename)

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
	    split = likelihood[:,-128:,:]
	    return np.sum(split, axis=1)
