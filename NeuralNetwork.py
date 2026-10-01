#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:49:56 2026

@author: Amirreza
"""

from math import exp
import random as r

class Neural_Network() :
    def __init__(self, input_size: int, hidden_size: int, output_size: int) :
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # each row -> a hidden node. each object -> input1, input2, ...
        self.w1 = [[r.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.w2 = [[r.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
        
        self.b1 = [r.uniform(-1, 1) for _ in range(hidden_size)]
        self.b2 = [r.uniform(-1, 1) for _ in range(output_size)]
    
    def Forward(self, inp) :
        o1 = []
        o2 = []
        self.last_input = inp
        
        for hidden in range(self.hidden_size) :
            z = [k for k in inp]
            
            for i in range(len(z)) :
                z[i] *= self.w1[hidden][i]
            
            o1.append(self._sigmoid((self.b1[hidden] + sum(z))))
        
        for output in range(self.output_size) :
            z = [k for k in o1]
            
            for i in range(len(z)) :
                z[i] *= self.w2[output][i]
            
            o2.append(self._sigmoid((self.b2[output] + sum(z))))       
            
        self.o1, self.o2 = o1, o2
        return o2
            
    def _sigmoid(self, x):
        if x >= 0:
            return 1 / (1 + exp(-x))
        else:
            return exp(x) / (1 + exp(x))
            
    def Backward(self, answers, learning_rate) :
        self.lr = learning_rate
        self.delta1 = []
        self.delta2 = []
        for i, output in enumerate(self.o2) :
            delta = (output-answers[i]) * (output * (1-output))
            self.delta2.append(delta)
        
        for hidden in range(self.hidden_size) :
            err = 0
            for output in range(self.output_size) :
                err += self.delta2[output] * self.w2[output][hidden]
            delta = err * self.o1[hidden] * (1 - self.o1[hidden])
            self.delta1.append(delta)
            
        for output in range(self.output_size) :
            for hidden in range(self.hidden_size) :
                self.w2[output][hidden] -= self.lr * self.delta2[output] * self.o1[hidden]
            self.b2[output] -= self.lr * self.delta2[output]
        
        for hidden in range(self.hidden_size) :
            for inp in range(self.input_size) :
                self.w1[hidden][inp] -= self.lr * self.last_input[inp] * self.delta1[hidden]
            self.b1[hidden] -= self.lr * self.delta1[hidden]
            
    def Train(self, data, epoches, lr) :
        """
        data : [[case1: [input1, input2, ...], [expected_outputs]], [case2], ...]
        """
        
        for e in range(epoches) :
            for c in data :
                inputs, outputs = c
                self.Forward(inputs)
                self.Backward(outputs, lr)