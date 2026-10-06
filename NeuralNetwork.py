#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:49:56 2026

@author: Amirreza
"""

import random as r

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
            self.weights.append([])
            self.weights[-1] = [[r.uniform(-1,1) for _ in range(self.sizes[i])] for __ in range(item)]

            self.biases.append([])
            self.biases[-1] = [r.uniform(-1,1) for _ in range(item)]
        
        
    def Forward(self, inputs) :
        results = [inputs]
        last_a = [inputs]
        tmp_res = 0
        for layer in range(1, len(self.sizes)) :    
            inner_layer = layer -1
            results.append([])
            last_a.append([])
            for neuron in range(self.sizes[layer]) :
                tmp_res = self.biases[inner_layer][neuron]
                for i, inp in enumerate(last_a[-2]) :
                    tmp_res += inp * self.weights[inner_layer][neuron][i]
                results[-1].append(tmp_res)
                last_a[-1].append(self.Activate(self.activation, tmp_res))
        self.last_z = results
        self.last_a = last_a
        return last_a[-1]
            
    def _sigmoid(self, x, backward=False) :
        from math import exp 
        if not backward :
            if x >= 0:
                return 1 / (1 + exp(-x))
            else:
                return exp(x) / (1 + exp(x))
        else :
            return (self._sigmoid(x) * (1 - self._sigmoid(x)))
            
    def _tanh(self, x, backward=False) : 
        if not backward :
            from math import tanh
            return tanh(x)
        else :
            return 1 - self._tanh(x)**2
        
    def _relu(self, x, backward=False) :
        if not backward :
            return max(0, x)
        else :
            if x > 0 :
                return 1
            else :
                return 0
        
    def _leaky_relu(self, x, alpha=0.01, backward=False) :
         if not backward :
             if x < 0 :
                 return alpha * x
             else :
                 return x
         else :
             if x > 0 :
                 return 1
             else :
                 return alpha
        
    def _soft_plus(self, x, backward=False) :
        if not backward : 
            from math import log, exp 
            return log(1+exp(x))
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
        self.lr = learning_rate
        self.deltas = [[] for _ in range(len(self.sizes))]
        
        for i, output in enumerate(self.last_a[-1]) :
            delta = (output - answers[i]) * self.Activate(self.activation, self.last_z[-1][i], True)
            self.deltas[-1].append(delta)
        
        for Layer in range(len(self.sizes)-1, 0, -1) :
            for Neuron in range(self.sizes[Layer]) :
                delta = 0
                if Layer != len(self.sizes)-1 : 
                    for Next_Neuron in range(self.sizes[Layer+1]) :
                        delta += (self.deltas[Layer+1][Next_Neuron] * self.weights[Layer][Next_Neuron][Neuron])
                    delta *= self.Activate(self.activation, self.last_z[Layer][Neuron], backward=True)
                    self.deltas[Layer].append(delta)
                for inp in range(self.sizes[Layer-1]) :
                    self.weights[Layer-1][Neuron][inp] -= self.lr * self.deltas[Layer][Neuron] * self.last_a[Layer-1][inp]
                self.biases[Layer-1][Neuron] -= self.lr * self.deltas[Layer][Neuron]
                    
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
    
nn = Neural_Network(2, [2], 1, activation="leaky-relu")
nn.Train(TABLE, 100000, 0.05)

print("--- predictions ---")
for inp, target in TABLE:
    out = nn.Forward(inp)[0]
    print(f"input {inp}  target {target[0]}  output {out:.4f}")