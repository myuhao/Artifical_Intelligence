import numpy as np

class MultiClassPerceptron(object):
	def __init__(self,num_class,feature_dim):
		"""Initialize a multi class perceptron model.

		This function will initialize a feature_dim weight vector,
		for each class.

		The LAST index of feature_dim is assumed to be the bias term,
			self.w[:,0] = [w1,w2,w3...,BIAS]
			where wi corresponds to each feature dimension,
			0 corresponds to class 0.

		Args:
		    num_class(int): number of classes to classify
		    feature_dim(int): feature dimension for each example
		"""

		self.w = np.zeros((feature_dim+1,num_class))

	def train(self,train_set,train_label):
		""" Train perceptron model (self.w) with training dataset.

		Args:
		    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
		    train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
		"""

		# YOUR CODE HERE
		# Initialize the bias to 1.
		self.w[-1,:] = 1

		for i in range(10):
			for train_idx in range(len(train_set)):
				train_sample = train_set[train_idx]
				sample_class = train_label[train_idx]
				# Get the weight of each class using inner product of weight and the sample.
				class_weights = np.dot(train_sample, self.w[:-1,:])
				# Add the bias.
				class_weights += self.w[-1,:]
				# Get the class index.
				class_idx = np.argmax(class_weights)
				# Wrong prediction
				if class_idx != sample_class:
					etaf = 0.001 * train_sample
					self.w[:-1,class_idx] -= etaf
					self.w[:-1,sample_class] += etaf



	def test(self,test_set,test_label):
		""" Test the trained perceptron model (self.w) using testing dataset.
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

		return accuracy, pred_label

	def save_model(self, weight_file):
		""" Save the trained model parameters
		"""

		np.save(weight_file,self.w)

	def load_model(self, weight_file):
		""" Load the trained model parameters
		"""

		self.w = np.load(weight_file)

