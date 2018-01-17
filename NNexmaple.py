import numpy as np
import time
import matplotlib.pyplot as plt

class SLC():
    def __init__(self):
        self.N = 100 # number of points per class
        self.D = 2 # dimensionality
        self.K = 3 # number of classes
        self.h = 100 # size of hidden layer
        self.X = np.zeros((self.N * self.K, self.D)) # data matrix (each row = single example)
        self.y = np.zeros(self.N * self.K, dtype='uint8') # class labels
        self.lamda = 1e-3 #regulation constant
        self.step_size = 1e-0
        for j in xrange(self.K):
            ix = range(self.N*j, self.N*(j+1))
            r = np.linspace(0.0,1, self.N) # radius
            t = np.linspace(j*4,(j+1)*4, self.N) + np.random.randn(self.N)*0.2 # theta
            self.X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            self.y[ix] = j
        # lets visualize the data:
        #plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=40, cmap=plt.cm.Spectral)
        #plt.show()
        self.W = 0.01 * np.random.randn(self.D, self.h)
        self.b = np.zeros((1, self.h))
        self.W2 = 0.01 * np.random.randn(self.h, self.K)
        self.b2 = np.zeros((1, self.K))

    def activation(self, data):
        return np.maximum(0, data) # ReLU activation


    def compute_scores(self):
        self.scores = np.dot(self.X, self.W) + self.b

    def compute_NN_scores(self):
        scores_1_input = np.dot(self.X, self.W) + self.b
        self.hidden_layer_1 = self.activation(scores_1_input)
        self.NNscores = np.dot(self.hidden_layer_1, self.W2) + self.b2

    def compute_softmax_loss(self):
        exp_scores = np.exp(self.scores)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        probs = self.probs
        log_probs = -np.log(probs[range(self.X.shape[0]), self.y])
        data_loss = np.sum(log_probs) / self.X.shape[0]
        reg_loss = 0.5 * self.lamda * np.sum(self.W * self.W)
        self.loss = data_loss + reg_loss

    def compute_softmax_loss_NN(self):
        exp_scores = np.exp(self.NNscores)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        probs = self.probs
        corect_logprobs = -np.log(probs[range(self.X.shape[0]),self.y])
        data_loss = np.sum(corect_logprobs)/self.X.shape[0]
        reg_loss = 0.5 * self.lamda * np.sum(self.W * self.W) + 0.5 * self.lamda * np.sum(self.W2 * self.W2)
        self.loss = data_loss + reg_loss

    def compute_a_gradient(self):
        grad = self.probs
        grad[range(self.X.shape[0]), self.y] -= 1
        grad /= self.X.shape[0]
        self.dW = np.dot(self.X.T, grad)
        self.db = np.sum(grad, axis=0, keepdims=True)
        self.dW += self.lamda * self.W

    def compute_a_gradient_NN(self):
        grad = self.probs
        grad[range(self.X.shape[0]), self.y] -= 1
        grad /= self.X.shape[0]
        self.dW2 = np.dot(self.hidden_layer_1.T, grad)
        self.db2 = np.sum(grad, axis=0, keepdims=True)
        dhidden = np.dot(grad, self.W2.T)
        #backpropagate the ReLU function
        dhidden[dhidden <= 0] = 0
        self.dW = np.dot(self.X.T, dhidden)
        self.db = np.sum(dhidden, axis=0, keepdims=True)

        self.dW += self.lamda * self.W
        self.dW2 += self.lamda * self.W2

    def param_update(self):
        self.W += -self.step_size * self.dW
        self.b += -self.step_size * self.db

    def param_update_NN(self):
        self.W += -self.step_size * self.dW
        self.b += -self.step_size * self.db
        self.W2 += -self.step_size * self.dW2
        self.b2 += -self.step_size * self.db2

    def train_classifier(self):
        current_time = time.time()
        for i in xrange(250):
            self.compute_scores()
            self.compute_softmax_loss()
            if i % 10 == 0:
                print "iteration %d: loss %f" % (i, self.loss)
            self.compute_a_gradient()
            self.param_update()
        duration = time.time() - current_time

        prediction = np.argmax(self.scores, axis=1)
        print "training time: " + str(duration)
        print "training accuracy: %.3f" % (np.mean(prediction == self.y))

    def train_NN(self):

        for i in xrange(15000):
            self.compute_NN_scores()
            self.compute_softmax_loss_NN()
            if i % 1000 == 0:
                print "iteration %d: loss %f" % (i, self.loss)
            self.compute_a_gradient_NN()
            self.param_update_NN()

        hidden_layer_1 = np.maximum(0, np.dot(self.X, self.W) + self.b)
        scores = np.dot(self.hidden_layer_1, self.W2) + self.b2
        predicted_class = np.argmax(scores, axis=1)
        print 'training accuracy: %.2f' % (np.mean(predicted_class == self.y))


if __name__ == '__main__':
    nn = SLC()
    nn.train_classifier()
