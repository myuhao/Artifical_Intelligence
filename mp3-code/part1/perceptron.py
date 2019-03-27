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
		self.num_class = num_class
		self.feature_dim = feature_dim

	def train(self,train_set,train_label):
		""" Train perceptron model (self.w) with training dataset. 

		Args:
		    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
		    train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
		"""

		# YOUR CODE HERE

		learn_rate = 0.001
		epoch_num = 30
		self.w[self.feature_dim , :] += 0.5

		train_order = []
		for i in range(len(train_label)):
			train_order.append(i)


		for j in range(epoch_num):
			# randomize test set
			np.random.shuffle(train_order)
			print("epoch: {0}" .format(j+1))
			for order in train_order:
				data_0 = train_set[order]
				label = train_label[order]
				data = np.concatenate((data_0, [1]))

				max_label = np.argmax(np.dot(data, self.w))

				if max_label == label:
					# label correct do nothing
					pass

				else:
					# label incorrect
					uf = np.concatenate((data_0, [0])) * learn_rate
					# update correct label weight
					self.w[:, label] += uf
					# update misclassified label weight
					self.w[:, max_label] -= uf


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

		for i in range(len(test_set)):
			sample_0 = test_set[i]
			label = test_label[i]

			sample = np.concatenate((sample_0, [1]))
			max_label = np.argmax(np.dot(sample, self.w))

			pred_label[i] = max_label

			if max_label == label:
				accuracy += 1

		accuracy /= len(test_set)

		return accuracy, pred_label

	def save_model(self, weight_file):
		""" Save the trained model parameters 
		""" 

		np.save(weight_file, self.w)

	def load_model(self, weight_file):
		""" Load the trained model parameters 
		""" 

		self.w = np.load(weight_file)


