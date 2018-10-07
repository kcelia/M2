import torch as T

T.set_default_tensor_type('torch.FloatTensor')

class Loss:
    def get_loss_value(self, predict, y):
        raise NotImplementedError()

    def forward(self, predict, y):
        return self.get_loss_value(predict, y)

    def backward(self, predict, y):
        raise NotImplementedError()

    def __call__(self, predict, y):
        return self.get_loss_value(predict, y)

class Module:
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, y):
        raise NotImplementedError()

    def backward_update_gradient(self, x, delta):
        raise NotImplementedError()
    
    def update_parameters(self, epsilon):
        raise NotImplementedError()
    
    def backward_delta(self, x, delta):
        raise NotImplementedError()
    
    def zero_grad(self):
        raise NotImplementedError()
    
    def initialize_parameters(self):
        raise NotImplementedError()

    def initialize(self):
        raise NotImplementedError()

    def __call__(self, x):
        self.activation = self.forward(x)
        return self.activation

class Linear(Module):
    def __init__(self, input_s, output_s, bias=1):
        self.input_s = input_s
        self.output_s = output_s
        self.bias = bias
        self.initialize_parameters()
    
    def forward(self, x):
        if self.bias:
            #FIXME
            x = T.cat((x, T.ones(batch_size, [1](x.dim()-1))), 1)
            self.z = x.mm(self.w.t()).view(-1, self.output_s)
            return self.z
        else:
            self.z = x.mm(self.w.t()).view(-1, self.output_s)
            return self.z
    
    def backward_update_gradient(self, x, delta):
        self.grad += delta.unsqueeze(1).expand(-1, x.size(0)).mm(x).squeeze()

    def update_parameters(self, epsilon):
        self.w -= epsilon * self.grad

    def backward_delta(self, x, delta):
        return delta.unsqueeze(1).mm(x).mm(self.w)
    
    def zero_grad(self):
        self.grad = T.zeros(self.output_s, self.input_s + self.bias).float()
    
    def initialize_parameters(self):
        self.w = T.randn(self.output_s, self.input_s + self.bias).float()
        self.zero_grad()

class Sigmoid(Loss):
    def forward(self, x):
        self.z = 1 / (1 + T.exp(-x))
        return self.z

    def backward_delta(self, x):
        length = x.shape[0]
        result = T.zeros(length)
        for i in range(length):
            result[i] = T.exp(-x[i]) / (1 + T.exp(-x)).pow(2)
        return result

class Tanh(Module):
    def forward(self, x):
        self.a = T.tanh(x)
        return self.a

    def backward_delta(self, x, delta):
        return ((1 - self.a ** 2) * delta)
        
class MSE(Loss):  
    def get_loss_value(self, y, ypred):
        return T.norm(y - ypred).pow(2) / y.shape[0]

    def backward(self, y, ypred):
        return (2 / y.shape[0]) * (ypred - y).sum(0).view(-1)
    
class Hinge(Loss):
    def get_loss_value(self, y, ypred):
        return (((y * ypred) < 0).float() * (-y * ypred)).sum(0) / y.shape[0]
    
    def backward(self, y, ypred):
        return (((y * ypred) < 0).float() * -y).sum(0) / y.shape[0]
        
class BinaryCrossEntropy(Loss):
    @staticmethod
    def __safe_numerical_values(t):
        t[t != t] = 0.
        t[t == float('inf')]  =  1e15
        t[t == float('-inf')] = -1e15
        return t

    def get_loss_value(self, predict, y):
        predict = T.max(predict, T.ones(predict.shape) * 1e-15)
        predict = T.min(predict, T.ones(predict.shape) - 1e-15)
        t = -(y * T.log(predict) + (1 - y) * T.log(1 - predict))
        t = BinaryCrossEntropy.__safe_numerical_values(t)
        return t.sum() / float(predict.size()[1])

    def backward(self, predict, y):
        epsillon = T.ones(predict.shape) * 1e-15
        predict = T.max(predict, epsillon)
        predict = T.min(predict, T.ones(predict.shape) - epsillon)
        divisor = T.max(predict * (1 - predict), epsillon)
        return - (predict - y) / divisor


