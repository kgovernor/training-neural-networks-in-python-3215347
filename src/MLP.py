import numpy as np

class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By default it's 1.0."""

    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias).""" 
        self.weights = np.random.rand(inputs+1)*2 - 1
        self.bias = bias

    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        x_sum = np.dot(np.append(x, self.bias), self.weights)
        return self.sigmoid(x_sum)
    
    def set_weights(self, w_init):
        # w_init is a list of floats. Organize it as you'd like.
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        # return the output of the sigmoid function applied to x.
        sig = 1/(1+ np.exp(-x))
        return sig


# test code
neuron = Perceptron(inputs=2)
neuron.set_weights([10, 10, -15])  # AND

print("AND Gate:")
print("0 0 = {0:.10f}".format(neuron.run([0, 0])))
print("0 1 = {0:.10f}".format(neuron.run([0, 1])))
print("1 0 = {0:.10f}".format(neuron.run([1, 0])))
print("1 1 = {0:.10f}".format(neuron.run([1, 1])))

# test code
neuron = Perceptron(inputs=2)
neuron.set_weights([20, 20, -15])  # AND

print("\nOR Gate:")
print("0 0 = {0:.10f}".format(neuron.run([0, 0])))
print("0 1 = {0:.10f}".format(neuron.run([0, 1])))
print("1 0 = {0:.10f}".format(neuron.run([1, 0])))
print("1 1 = {0:.10f}".format(neuron.run([1, 1])))
