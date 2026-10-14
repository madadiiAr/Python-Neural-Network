#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 17:00:06 2026

@author: Amirreza
"""
import numpy as np

class Neural_Network() :
    def __init__(self, input_size: int, hiddens_size: [int, ...], output_size: int, activation: str = "Sigmoid", alpha=0.01) :
        """
        SUMMARY.

        Parameters
        ----------
        input_size : int
            how many inputs.
        hiddens_size : [int, ...]
            a list. each item shows a layer of hidden neurons. each number shows the number of neurons in that layer.
        output_size : int
            how many outputs.
        activation : str, optional
            type of activation. can be ... or ... or ... . The default is "Sigmoid".

        Returns
        -------
        None
        """
        
        self.sizes = [input_size] + hiddens_size + [output_size]
        self.alpha = alpha
        self.activation = activation
        
        # each row -> a hidden node. each object -> input1, input2, ...
        self.weights = []
        self.biases = []

        for i, item in enumerate(self.sizes[1:]) :
            self.weights.append(np.random.uniform(-1, 1, size=(item, self.sizes[i])))
            self.biases.append(np.random.uniform(-1, 1, size=(item,)))
        
    def Forward(self, inputs) :
        inputs = np.array(inputs, np.float64)
        results = [inputs]
        last_a = [inputs]
        for layer in range(1, len(self.sizes)) :    
            inner_layer = layer -1
            
            results.append(self.weights[inner_layer] @ last_a[-1] + self.biases[inner_layer])
            last_a.append(self.Activate(self.activation, results[-1]))
            
        self.last_z = results
        self.last_a = last_a
        return last_a[-1]
            
    def _sigmoid(self, x, backward=False) :
        if not backward :
            return 1 / (1 + np.exp(-x))  
        else :
            return (self._sigmoid(x) * (1 - self._sigmoid(x)))
            
    def _tanh(self, x, backward=False) : 
        if not backward :
            return np.tanh(x)
        else :
            return 1 - self._tanh(x)**2
        
    def _relu(self, x, backward=False) :
        if not backward :
            return np.maximum(0, x)
        else :
            return np.where(x > 0, 1.0, 0.0)
        
    def _leaky_relu(self, x, alpha=0.01, backward=False) :
        if not backward :
            return np.where(x<0, alpha*x, x)
        else :
            return np.where(x>0, 1, alpha)
        
    def _soft_plus(self, x, backward=False) :
        if not backward : 
            return np.log(1+np.exp(x))
        else :
            return self._sigmoid(x)
    
    def _linear(self, x, backward=False) :
        if not backward :
            return x
        else :
            return 1
    
    def _silu(self, x, backward=False) :
        if not backward :
            return self._sigmoid(x) * x
        else :
            return (self._sigmoid(x) + x * self._sigmoid(x) * (1 - self._sigmoid(x)))
        
    def Activate(self, typE, x, backward=False) :
        res = None
        typE = typE.lower()
        if typE == "sigmoid" :
            res = self._sigmoid(x, backward = backward)
        elif typE == "tanh" :
            res = self._tanh(x, backward = backward)
        elif typE == "relu" :
            res = self._relu(x, backward = backward)
        elif typE == "leaky-relu" :
            res = self._leaky_relu(x, alpha=self.alpha, backward = backward)
        elif typE == "soft-plus" :
            res = self._soft_plus(x, backward = backward)
        elif typE == "linear" :
            res = self._linear(x, backward = backward)
        elif typE == "silu" :
            res = self._silu(x, backward = backward)
        return res
    
    def Backward(self, answers, learning_rate) :
        answers = np.array(answers, np.float64)
        self.lr = learning_rate
        self.deltas = [None] * len(self.sizes)
        delta = self.last_a[-1] - answers
        delta = delta * self.Activate(self.activation, self.last_z[-1], backward=True)
        self.deltas[-1] = delta
        for layer in range(len(self.sizes)-1, 0, -1) :
            if layer != len(self.sizes)-1 :
                self.deltas[layer] = (self.weights[layer].T @ self.deltas[layer+1]) * self.Activate(self.activation, self.last_z[layer], True)
            self.weights[layer-1] -= self.lr * np.outer(self.deltas[layer], self.last_a[layer-1])
            self.biases[layer-1] -= self.lr * self.deltas[layer]
            
    def Train(self, data, epoches, lr) :
        """
        data : [[case1: [input1, input2, ...], [expected_outputs]], [case2], ...]
        """
        
        for e in range(epoches) :
            for c in data :
                inputs, outputs = c
                self.Forward(inputs)
                self.Backward(outputs, lr)
                
if __name__ == "__main__":
    TABLE = [
    ([0, 0], [0]),   # sum=0, carry=0
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),   # sum=0, carry=1
]
    
nn = Neural_Network(2, [5,5], 1, activation="leaky-relu")
nn.Train(TABLE, 300000, 0.05)

print("--- predictions ---")
for inp, target in TABLE:
    out = nn.Forward(inp)[0]
    print(f"input {inp}  target {target[0]}  output {out:.4f}")