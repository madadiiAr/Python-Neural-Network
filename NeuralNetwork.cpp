#include <cmath>
#include <iostream>
#include <vector>
#include <cstddef>
#include <random>

// -------------------- Sigmoid activation --------------------

double Activation_Sigmoid(double x, bool Backward=false) {
    if (Backward) {
        return x * (1 - x);
    }else {
        return 1/(1 + std::exp(-x));
    }
}

class NeuralNetwork {
// -------------------- Neuron definition --------------------
public:
    std::vector<int> size_;
    std::vector<std::vector<std::vector<double>>> weights_ = {{}};
    std::vector<std::vector<double>> biases_ = {{}};
    std::vector<std::vector<double>> last_z_;
    std::vector<std::vector<double>> last_a_;
    std::vector<std::vector<double>> deltas_;

    NeuralNetwork(std::vector<int> sizes)
    {
        std::mt19937 generator(42);
        std::uniform_real_distribution<double> dist(-1.0,1.0);

        size_ = sizes;
        for (std::size_t layer=1; layer<sizes.size(); layer++) {
            weights_.push_back({});
            biases_.push_back({});

            for (std::size_t neuron=0; neuron<sizes[layer]; neuron++) {
                weights_[layer].push_back({});

                for (std::size_t input=0; input<size_[layer-1];input++)
                {
                    weights_[layer][neuron].push_back(dist(generator));
                }
            
                biases_[layer].push_back(dist(generator));
            }
        }
    }

    double Neuron(const std::vector<double>&   inputs,
                const std::vector<double>&     weights,
                const double                   bias) 
    {                
        double z = bias;

        for (std::size_t i = 0; i < inputs.size(); i++) {
            z += inputs[i] * weights[i];
        }

        double a = Activation_Sigmoid(z);
        last_z_.back().push_back(z);
        last_a_.back().push_back(a);
        return a;
    }

    // -------------------- Layer definition --------------------

    std::vector<double> Layer(const std::vector<double>&              inputs,
                            const std::vector<std::vector<double>>&   weights,
                            const std::vector<double>&                biases)
    {
        std::vector<double> results;
        last_z_.push_back({});
        last_a_.push_back({});
        for (std::size_t i=0; i < biases.size(); i++){
            results.push_back(Neuron(inputs, weights[i], biases[i]));
        }
        return results;
    }

    // -------------------- Forward --------------------

    std::vector<double> Forward(const std::vector<double>& inputs)
    {
        last_z_ = {};
        last_a_ = {inputs};
        std::vector<double> last_results = inputs;
        for (std::size_t i=1; i < size_.size(); i++) {
            last_results = Layer(last_results, weights_[i], biases_[i]);
        }
        return last_results;
    }

    // -------------------- Backward --------------------

    void Backward(std::vector<double>& targets, double lr){
        deltas_.clear();
        deltas_.resize(size_.size());

        for (std::size_t neuron=0; neuron < size_.back(); neuron++) {
            double delta=0;
            delta = last_a_.back()[neuron] - targets[neuron];
            delta *=  Activation_Sigmoid(last_a_.back()[neuron], true);

            deltas_.back().push_back(delta);
        }

        for (std::size_t layer=size_.size()-2; layer >= 1; layer--) {
            for (std::size_t neuron=0; neuron < size_[layer]; neuron++){
                double delta=0;
                
                for (std::size_t next_neuron=0; next_neuron < size_[layer+1]; next_neuron++){
                    double x = weights_[layer+1][next_neuron][neuron];
                    x *= deltas_[layer+1][next_neuron];
                    delta += x;
                }

                delta *= Activation_Sigmoid(last_a_[layer][neuron], true);
                deltas_[layer].push_back(delta);
            }
        }

        for (std::size_t layer=size_.size()-1; layer >= 1; layer--) {
            for (std::size_t neuron=0; neuron < size_[layer]; neuron++){
                for (std::size_t previous_neuron=0; previous_neuron < size_[layer-1]; previous_neuron++){
                    weights_[layer][neuron][previous_neuron] -= lr * deltas_[layer][neuron] * last_a_[layer-1][previous_neuron];
                }
                biases_[layer][neuron] -= lr * deltas_[layer][neuron];
            }
        }
    }

    // -------------------- Train --------------------

    void Train(const std::vector<std::vector<std::vector<double>>>& table, std::size_t epoches, double lr) {
            for (std::size_t epoch=0; epoch < epoches; epoch++){
                for (const std::vector<std::vector<double>>& data_set: table) {
                    std::vector<double> inputs = data_set[0];
                    std::vector<double> targets= data_set[1];
                    Forward(inputs);
                    Backward(targets, lr);
                }
            }
    }
};

// -------------------- Main function --------------------

int main() {
    // -------------------- Values --------------------
    std::vector<std::vector<std::vector<double>>> table = {
        {{0.0, 0.0}, {0.0}},
        {{0.0, 1.0}, {1.0}},
        {{1.0, 0.0}, {1.0}},
        {{1.0, 1.0}, {0.0}}
    };

    // -------------------- Creating Network --------------------

    NeuralNetwork nn{{2, 5, 5, 1}};
    nn.Train(table, 300000, 0.01);

    // -------------------- Predictions --------------------

    std::cout << "------- Predictions -------" << std::endl;

    for (std::size_t index=0; index < table.size(); index++) {
        std::cout << "Row " << index+1 << ":" << std::endl << std::endl;

        std::vector<std::vector<double>>& dataset = table[index];
        std::vector<double> out= nn.Forward(dataset[0]);
        
        std::cout << "  Inputs :" << std::endl;
        std::cout << "      {";
        for (std::size_t input=0; input < dataset[0].size(); input++){
            std::cout << dataset[0][input];
            if (input < dataset[0].size()-1) {
                std::cout << ", ";
            }
        }
        std::cout << "}" << std::endl;

        std::cout << "  Outputs :" << std::endl;
        for (std::size_t output=0; output < dataset[1].size(); output++){
            std::cout << "      ";
            std::cout << "Expected " << dataset[1][output] << ", Got " << out[output] << std::endl;
        }

        if (index < table.size()-1) {
            std::cout << "-----" << std::endl;
        }
    }
}