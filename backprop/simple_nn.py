
# coding: utf-8

# In[2]:


import numpy as np


# ## Building a Simple Feed-Forward Neural Network
# 
# We will start from the bottom up by defining a single neuron. A neuron should be able to do the following:
# 
# 1. Calculate the value of its net input from its input weights and the given input values.
# 2. Calculate the value of its output from its activation function and the given net input.
# 3. Update its weights according to some delta value.

# In[3]:


class Neuron:
    def __init__(self, bias, activationFunc):
        # A neuron has both a bias and a sequence of weights.
        # We will leave the weights uninitialized for now.
        self.bias = bias
        self.activationFunc = activationFunc
        self.weights = []
    
    # Here we calculate the total input value 'net' for this neuron.
    # 'net' is simply the dot product of the given input vector with this neuron's weight vector.
    def calculate_net_input(self, inputs):
        assert inputs.shape == self.weights.shape
        return np.dot(self.weights, inputs) + self.bias
    
    # Here we call our neuron's activation function with the given 'net' input value.
    def activate(self, net_input):
        return self.activationFunc(net_input)
    
    # Convenience function that will both calculate net inputs and return the result of activation.
    def calculate_output(self, inputs):
        net_input = self.calculate_net_input(inputs)
        return self.activate(net_input)
    
    # And finally we allow for the weights to be updated according to some vector of delta values.
    # 'learning_rate' is an arbitrary scalar that will be multiplied against the deltas.
    def update(self, w_grads, b_grad, learning_rate):
        self.weights -= learning_rate * w_grads
        self.bias -= learning_rate * b_grad


# Now we'll move on to defining a single *neuron layer*. A layer is an arbitrary group of non-connected neurons that are each connected only to neurons in other layers. It can be thought of as a *partition* in terms of graph theory where each neuron is a component disjoint from the others in the layer. Our layer should be able to do the following:
# 
# 1. Initialize each of its neurons according to the shape of its input.
# 2. "Connect" its output to the next layer in front of it.
# 3. *Feed forward* its outputs to that layer.
# 4. Collect the error for each of its neurons.

# In[4]:


class NeuronLayer:
    def __init__(self, neuron_count, bias, activationFunc):
        # Use bias and activationFunc to create the neurons for our
        # layer. Everything else can stay unintialized for now.
        self.neuron_count = neuron_count
        self.outputs = np.zeros(neuron_count)
        self.input_dims = -1
        self.w_grads = []
        self.b_grads = []
        # 'receiverFunc' will be the 
        self.receiverFunc = None
        self.neurons = [Neuron(bias, activationFunc) for i in xrange(neuron_count)]
            
    # Sets the shape of our layer's expected input by initializing all of
    # the neurons' weight vectors to size 'input_dims'. Weight values are
    # initially set to uniform random values between -1 and 1.
    def initialize(self, input_dims, weight_values=None):
        assert weight_values == None or len(weight_values) == input_dims
        self.input_dims = input_dims
        for (i, n) in enumerate(self.neurons):
            n.weights = weight_values.copy() if weight_values != None else np.random.uniform(-0.5, 0.5, input_dims)
    
    # Connects 'receiver' to this layer to receive the results of 'feed_forward'.
    # 'receiver' may be either a NeuronLayer, in which case it will be initialized
    # to mach this layer's output shape, or it can be any function that expects a single
    # argument 'inputs' where 'inputs' is an array of length 'neuron_count' for this layer.
    # Note: if receiver is not a NeuronLayer, 'receiver_weight_values' is ignored.
    def connect(self, receiver, receiver_weight_values=None):
        assert receiver != None
        if isinstance(receiver, NeuronLayer):
            self.receiverFunc = lambda x: receiver.feed_forward(x)
            receiver.initialize(self.neuron_count, receiver_weight_values)
        else:
            self.receiverFunc = receiver
            
    # Computes the output value for each neuron given 'inputs' and then
    # "feeds forward" the results by calling 'receiverFunc' (if this layer
    # is connected to a receiver).
    def feed_forward(self, inputs):
        assert len(inputs) == self.input_dims
        for (i, n) in enumerate(self.neurons):
            self.outputs[i] = n.calculate_output(inputs)
        if self.receiverFunc != None:
            self.receiverFunc(self.outputs)
    
    # Calculates the mean squared error for each neuron in this layer given the vector of expected output values.
    def calculate_error(self, expected_outputs):
        assert len(expected_outputs) == len(self.outputs)
        return [0.5 * (expected_outputs[i] - self.outputs[i])**2 for (i, e) in enumerate(self.neurons)]
    
    # Updates the weights and biases for all of this layer's neurons from the given.
    # w_grads should be a matrix of size NxM, where N is output dimensions of this layer and M is the input dimensions.
    # b_grads should be a vector of size N, where N is the output dimensions of this layer.
    def update(self, w_grads, b_grads, learning_rate):
        assert w_grads.shape == (self.neuron_count, self.input_dims) and len(b_grads) == self.neuron_count
        self.w_grads = w_grads
        self.b_grads = b_grads
        for (i, n) in enumerate(self.neurons):
            n.update(w_grads[i], b_grads[i], learning_rate)
            
    def get_weights(self):
        return [(n.weights, n.bias) for n in self.neurons]
    
    def dump_state(self):
        print 'weights, bias: {0}'.format(self.get_weights())
        print 'gradients: {0} + {1}'.format(self.w_grads, self.b_grads)
        print 'outputs: {0}'.format(self.outputs)


# Now that we have our basic components, it's time for the fun part; the *nerual network*. Here we will combine multiple layers to form a fully connected, feed-forward network that we can train to fit some data.
# 
# In principle, we could generalize this code to allow for an arbitrary number of hidden layers. We'll keep things simple, though, and just implement a basic 3-layer FFNN.
# 
# *Note: "3-layer" includes the "input layer" which need not be represented by an actual NeuronLayer in the implementation. Traditionally, the "input layer" is just a way of visualizing how the inputs are fed into the hidden layer.*
# 
# Our network should do the following:
# 
# 1. Initialize and connect each neuron layer.
# 2. Implement backpropagation to allow for training.
# 3. Provide a function to "train" the network on any number of training examples.
# 4. Provide a function to "predict" one or more output values given some input (feed-forward only, no back prop).
# 5. Learn a model that will cure cancer.
# 
# Ok, maybe that last one is a bit ambitious. But we can be optimistic! :D
# 
# Item number (2) is where things get interesting... we need to backpropagate the error from our feed-forward output through the network. Each weight should be updated with respect to its contribution to the total error. We compute this using partial derivatives.
# 
# *Note: all multiplication $\times$ operations here are element-wise, unless otherwise specified.*
# 
# So for each weight $i$, we'll have:
# 
# $w'_i = w_i - \alpha \times \frac{\partial E_{total}}{\partial w_i}$
# 
# where $w'_i$ is our new weight value and $E_{total} = \sum_{i=0}^{N}\frac{1}{2} \times (o'_i - o_i)^2$
# where $o'_i$ is our expected output value and $o_i$ is our actual output value for some output $i$.
# 
# For each layer, the partial derivative of the errors with respect to a single weight can be computed as:
# 
# $\frac{\partial E_{total}}{\partial w_i} = \frac{\partial E_{total}}{\partial o_i}\times \frac{\partial o_i}{\partial net_i}\times\frac{\partial net_i}{\partial w_i}$
# 
# For the output layer:
# 
# $\frac{\partial E_{total}}{\partial o_i} = -(o'_i - o_i)$
# 
# $\frac{\partial o_i}{\partial net_i} = o_i\times (1 - o_i)$
# 
# $\frac{\partial net_i}{\partial w_i} = in_i$
# 
# where $in_i$ is the input from the hidden layer at $i$
# 
# For the hidden layer:
# 
# $\frac{\partial E_{total}}{\partial h_i} = \sum_{j}\frac{\partial E_{o_j}}{\partial h_i}$
# 
# where $h_i$ is the output for the hidden layer at node $i$ and $E_{o_j}$ is the error for the output layer at node $j$.
# 
# $\frac{\partial E_{o_j}}{\partial h_i} = \frac{\partial E_{o_j}}{\partial o_j}\times \frac{\partial o_j}{\partial net_{oj}}\times\frac{\partial net_{oj}}{\partial h_i}$
# 
# We already have the "delta terms" $\frac{\partial E_{o_j}}{\partial o_j}\times \frac{\partial o_j}{\partial net_{oj}}$ from computing the error for the ouptut layer, so all we need is $\frac{\partial net_{oj}}{\partial h_i}$ for each hidden neuron $i$. Notice that we have two dimensions here, $i$ (hidden layer index) and $j$ (output layer index). We can collect all of the weights in a $N\times M$ matrix $W$ (weights from hidden layer to output layer), where $N$ is the number of hidden dimensions and $M$ is the number of output dimensions.
# 
# Then our vector of partial derivatives for error with respect to each hidden layer output can be computed as:
# 
# $\frac{\partial E_{total}}{\partial h_i} = \sum_{j} W_{ij}\times D$
# 
# where $D$ is our vector of delta values from the output layer computation.
# 
# $\frac{\partial net_{oj}}{\partial h_i} = W_{ij}$
# 
# for each hidden layer neuron and output layer neuron pair.
# 
# You will find the implementation of this in the `_backpropagate_error` function in `SimpleNN` below.

# In[5]:


# Now our neural net.
class SimpleNN:
    
    def __init__(self, input_dims, hidden_dims, output_dims, learn_rate=0.05, **args):
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.learn_rate = learn_rate
        
        # Process optional arguments
        hidden_layer_weights = args['hidden_layer_weights'] if 'hidden_layer_weights' in args else None
        output_layer_weights = args['output_layer_weights'] if 'output_layer_weights' in args else None
        hidden_layer_bias = args['hidden_layer_bias'] if 'hidden_layer_bias' in args else 0.0
        output_layer_bias = args['output_layer_bias'] if 'output_layer_bias' in args else 0.0
        
        # Define activation function for this FFNN implementation.
        # Logistic "sigmoid" activation is a standard choice for basic neural nets.
        logistic_activation = lambda x: 1 / (1 + np.exp(-x))
        
        # Initialize our layers
        self.hidden_layer = NeuronLayer(hidden_dims, hidden_layer_bias, logistic_activation)
        self.output_layer = NeuronLayer(output_dims, output_layer_bias, logistic_activation)
        self.hidden_layer.initialize(input_dims, hidden_layer_weights)
        self.hidden_layer.connect(self.output_layer, output_layer_weights)
        
    # Evaluates the model at its current state by feeding the given inputs into the network
    # and calculating the error in comparison to 'expected_outputs'. If 'return_mse' is True,
    # the mean squred error (MSE) for each output value is returned. Otherwise, the error per-output
    # is returned as an array.
    def evaluate(self, inputs, expected_outputs, return_mse=True):
        self.hidden_layer.feed_forward(inputs)
        errors = self.output_layer.calculate_error(expected_outputs)
        return errors if not return_mse else np.average(errors)
            
    def predict(self, inputs):
        inputs = np.array(inputs)
        self.hidden_layer.feed_forward(inputs)
        return self.output_layer.outputs
    
    def train(self, training_inputs, training_outputs):
        training_inputs = np.array(training_inputs)
        training_outputs = np.array(training_outputs)
        # If only one sample was given (in 1D form), reshape into 2D array with 1 row.
        if len(training_inputs.shape) == 1:
            training_inputs = np.reshape(training_inputs, (1, training_inputs.shape[0]))
            training_outputs = np.reshape(training_outputs, (1, training_outputs.shape[0]))
        
        history = []
        for (inputs, outputs) in zip(training_inputs, training_outputs):
            mse = self.evaluate(inputs, outputs)
            history.append((mse, self.output_layer.outputs.copy()))
            
            # Backpropagate error through network; we'll define this function below.
            self._backpropagate_error(inputs, outputs)
            
        return history
    
    def dump_state(self):
        print '{0} hidden dimensions, {1} output dimensions, {2} learning rate'.format(self.hidden_layer.neuron_count, self.output_layer.neuron_count, self.learn_rate)
        print '=== Hidden Layer ==='
        print self.hidden_layer.dump_state()
        print '=== Output Layer ==='
        print self.output_layer.dump_state()
    
    # Propagate output error with respect to expected_outputs back through the network using
    # the backpropagation algorithm.
    def _backpropagate_error(self, inputs, expected_outputs):
        output_layer_deltas = self._calculate_output_layer_deltas(expected_outputs)
        hidden_layer_deltas = self._calculate_hidden_layer_deltas(output_layer_deltas)
        output_layer_grads = self._calculate_gradients(output_layer_deltas, self.hidden_layer.outputs, self.hidden_dims, self.output_dims)
        self.output_layer.update(output_layer_grads, output_layer_deltas, self.learn_rate)
        hidden_layer_grads = self._calculate_gradients(hidden_layer_deltas, inputs, self.input_dims, self.hidden_dims)
        self.hidden_layer.update(hidden_layer_grads, hidden_layer_deltas, self.learn_rate)
        
#         for (i, o) in enumerate(self.output_layer.neurons):
#             for (j, w) in enumerate(o.weights):
#                 pd_error_wrt_weight = output_layer_deltas[i]*self.hidden_layer.outputs[j]
#                 o.weights[j] -= self.learn_rate * pd_error_wrt_weight
#             o.bias -= self.learn_rate * output_layer_deltas[i]
#         for (i, h) in enumerate(self.hidden_layer.neurons):
#             for (j, w) in enumerate(h.weights):
#                 pd_error_wrt_weight = hidden_layer_deltas[i]*inputs[j]
#                 h.weights[j] -= self.learn_rate * pd_error_wrt_weight
#             h.bias -= self.learn_rate * hidden_layer_deltas[i]
                
    def _calculate_output_layer_deltas(self, expected_outputs):
        deltas = np.zeros(self.output_dims)
        for (i, n) in enumerate(self.output_layer.neurons):
            pd_error_wrt_output = -(expected_outputs[i] - self.output_layer.outputs[i])
            pd_output_wrt_net_input = self.output_layer.outputs[i] * (1 - self.output_layer.outputs[i])
            deltas[i] = pd_error_wrt_output * pd_output_wrt_net_input
        return deltas
    
    def _calculate_hidden_layer_deltas(self, output_layer_deltas):
        deltas = np.zeros(self.hidden_dims)
        for (i, h) in enumerate(self.hidden_layer.neurons):
            pd_err_wrt_output = 0
            for (j, o) in enumerate(self.output_layer.neurons):
                pd_net_input_wrt_hidden_output = o.weights[i]
                pd_err_wrt_output += output_layer_deltas[j] * pd_net_input_wrt_hidden_output
            pd_hidden_output_wrt_net_input = self.hidden_layer.outputs[i] * (1 - self.hidden_layer.outputs[i])
            deltas[i] = pd_err_wrt_output * pd_hidden_output_wrt_net_input
        return deltas
    
    def _calculate_gradients(self, deltas, inputs, input_dims, output_dims):
        w_grads = np.zeros((output_dims, input_dims))
        for i in xrange(output_dims):
            for j in xrange(input_dims):
                pd_error_wrt_weight = deltas[i]*inputs[j]
                w_grads[i,j] = pd_error_wrt_weight
        return w_grads


# In[17]:


# Now let's test it! We'll start with something simple that even a single Perceptron could do.
x_train = [[0,0], [0,1], [1,0], [1,1]]
y_train = [[0], [1], [1], [0]]
nn = SimpleNN(input_dims = 2, hidden_dims = 10, output_dims = 1, learn_rate=0.5)
print 'Before training'
nn.dump_state()

print 'Training...'
mse = 100000
num_itr = 0
while mse > 0.0001 and num_itr < 5000:
    history = nn.train(x_train, y_train)
    mse = np.average([h[0] for h in history])
    num_itr += 1
print 'Done'
    
print nn.predict([0,1])
print nn.predict([1,1])
print mse
nn.dump_state()

