import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#   Activation functions
def sigmoid(x):
    return 1 / np.sum([1, np.exp(-x)])


def sigmoid_dev(x):
    return np.exp(-x) / np.sum([1, np.exp(-x)]) ** 2


def relu(x):
    return max(0,x)


def relu_dev(x):
    if x > 0:
        return 1
    else:
        return 0


#   Loss functions
def MSE(y, y_out, n=1):
    return (1/n) * (y-y_out) ** 2


def MSE_dev(y,y_out, n=1):
    return (-2/n) * (y-y_out)


def binary_crossentropy(y, y_out):
    return -1 * (y * np.log(y_out) + (1-y)*np.log(1-y_out))


def binary_crossentropy_dev(y, y_out):
    return binary_crossentropy(y, y_out) * (1 - binary_crossentropy(y, y_out))


def binary_crossentropy_dev2(y, y_out):
    return (y_out - y)/ (y_out * (1-y_out))

#Neural network
class NeuralNetwork:

    def __init__(self,
                 X_train=None,
                 Y_train=None,
                 layers = None,
                 learn_rate=0.001,
                 loss = 'MSE',
                 seed = 10,
                 n_samples=1,
                 weight=None
                 ):
        """
        Function to initialize NN
        :param inputs: Input vector in shape (1,N)
        :param layers: list of int for layers with input
        :param outputs: Output vector
        :param learn_rate: Learning rate
        :param loss: Loss function
        :param seed: Seed for weights initialization
        """
        self.X = X_train
        self.Y = Y_train
        self.I = None
        self.layers = layers
        self.O = None
        self.learn_rate = learn_rate
        self.loss = loss
        self.seed = seed
        self.weight = weight
        self.n_samples = n_samples
        self.n_counter = 0

        self.N = len(layers)
        self.recc = 1
        self.loss = []
        self.mean_loss = []

        self.B = {}
        self.W = {}     #Weights
        self.Z = {}     #Dot products
        self.A = {}     #Activation potential

        self.initialize_weights()
        self.load_data()

    #Function initializes weights for layers
    #with normal distribution
    def initialize_weights(self):
        np.random.seed(self.seed)

        if self.weight:
            for i in range(1, self.N):
                self.W[i] = np.full((self.layers[i], self.layers[i - 1]), self.weight)
                self.B[i] = np.full((self.layers[i], 1), 0)
                #print(f'Vahy {i} vrstva: {self.W[i]}')
        else:
            for i in range(1,self.N):
                self.W[i] = np.random.randn(self.layers[i], self.layers[i-1])   #(x,y)
                self.B[i] = np.random.randn(self.layers[i], 1)                  #(x,1)
                #print(f'Vahy {i} vrstva: {self.W[i]}')

    #Method loads the data to train
    def load_data(self):
        data_file = "C:/Users/grofc/Desktop/Projekt Ing/NN_backpropagation/" \
                    "data_toxicity/qsar_fish_toxicity.csv"
        data_header = ["ClCO", "SM1", "GATS11", "NDSCH", "NDSSC", "MLOGP", "Response"]
        data = pd.read_csv(data_file, sep=";", names=data_header)
        # print(data.head())
        # print(data.isna().sum())

        #Select only 3 Input variables
        X = data.drop(columns=["Response", "MLOGP", "NDSSC", "NDSCH"]).copy()

        self.Y = data["Response"].copy()
        self.X = X.to_numpy()

        #Load first training item
        self.I = self.X[0]
        self.O = self.Y[0]

    #Method returns weights for specific layer
    def get_weights(self, layer):
        if layer >= self.N:     #At Zero weights= Input!!!
            return

        return self.W[layer]

    #Method returns dots product for specific layer
    def get_dots(self, layer):
        if layer > self.N:
            return

        return self.A[layer-1]

    #Method returns Input for training
    def get_input(self):
        return self.I

    #Method returns expected Output
    def get_output(self):
        return self.O

    #Method returns mean loss
    def get_mean_loss(self):
        return self.mean_loss


    #Forwarding of the NN
    def forward(self):
        self.I = self.X[self.n_counter]
        self.O = self.Y[self.n_counter]

        #Reset N samples counter
        self.n_counter += 1
        if self.n_counter == self.n_samples:
            self.n_counter = 0

        self.Z[0] = np.array(self.I).reshape(1, -1)
        self.A[0] = np.array(self.I).reshape(1, -1)

        for i in range(1,self.N):
            if i == 1:
                self.Z[i] = np.matmul(self.I, self.W[i].T) + self.B[i].T
            else:
                self.Z[i] = np.matmul(self.A[i-1], self.W[i].T) + self.B[i].T

            if i == 1:
                self.A[i] = np.vectorize(relu)(self.Z[i])
                #self.A[i] = sigmoid(self.Z[i])
            else:
                self.A[i] = self.Z[i].copy() #We use identinty activation function for output!
            print(self.A[i])

        #Reset backpropagation flag
        self.recc =1

    def backpropagation(self, backprop = 1):

        if self.recc == self.N:
            return
        else:

            #Calculation of error
            if self.recc == 1:
                error = MSE(self.O, self.A[self.N-1])
                self.loss.append(float(error))

                if (self.n_counter + 1) == self.n_samples:
                    mean_losses = sum(self.loss[-self.n_samples:]) / len(self.loss[-self.n_samples:])
                    self.mean_loss.append(mean_losses)
                if error == np.nan or error == np.inf:
                    return
                #print(f"Error is {error.flatten()}")

            n = self.N - self.recc

            # Using only activation potential!!!
            if self.recc == 1:
                backprop = MSE_dev(self.O, self.A[n])
            else:
                backprop = backprop * self.W[n+1]
                backprop = backprop * np.vectorize(relu_dev)(self.Z[n])
                #backprop = backprop *sigmoid_dev(self.Z[n])

            self.W[n] = (self.W[n].T - self.learn_rate * backprop * self.A[n - 1].T).T
            #self.B[n] = self.B[n] - self.learn_rate * backprop.T

            self.recc = self.recc + 1
            self.backpropagation(backprop)

    # Function visualize losses over epochs
    def visualize_loss(self):
        fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
        plt.plot(self.loss, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.title("Loss over epochs")
        plt.show()

    def visualize_MSE(self):
        fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
        self.mean_loss.append(np.sum(self.loss)/len(self.X))
        print(f'Total loss {self.loss}')

        plt.plot(self.mean_loss, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.title("Loss over epochs")
        plt.show()

    def fit_one(self, epochs=1):

        while(epochs):
            self.forward()
            self.backpropagation()
            epochs = epochs -1
            #print(self.A[self.N-1].reshape(1,1))
        self.visualize_loss()

    def fit(self, epochs=10):
        #Update weights num times
        while(epochs):

            for i in range(0, len(self.X)):
                self.I = self.X[i]
                self.O = self.Y[i]
                self.forward()
                self.backpropagation()

            epochs = epochs - 1

            self.loss = []
            self.visualize_MSE()


if __name__ == "__main__":
    layers = [3, 5, 1]
    learning_rate = 0.001
    seed = 1

    #Loading data

    # data_file = "C:/Users/grofc/Desktop/Projekt Ing/NN_backpropagation/data_toxicity/qsar_fish_toxicity.csv"
    # data_header = ["ClCO", "SM1", "GATS11", "NDSCH", "NDSSC", "MLOGP", "Response"]
    # data = pd.read_csv(data_file, sep=";", names=data_header)
    # print(data.head())
    #
    # print(data.isna().sum())
    #
    # X = data.drop(columns=["Response", "MLOGP", "NDSSC", "NDSCH"]).copy()
    # Y = data["Response"].copy()
    #
    # X = X.to_numpy()

    n = NeuralNetwork(X_train=None, Y_train=None, layers=layers, learn_rate=learning_rate, seed=seed)
    n.fit_one(epochs=2)
