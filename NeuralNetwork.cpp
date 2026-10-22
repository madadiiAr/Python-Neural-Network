#include <cmath>
#include <iostream>
#include <vector>
#include <cstddef>
#include <random>
#include <cctype>
#include <iomanip>
#include <chrono>

class NeuralNetwork {
private:
    std::vector<std::vector<std::vector<double>>> weights_ = {{}};
    std::vector<std::vector<double>> biases_ = {{}};
    std::vector<std::vector<double>> last_z_;
    std::vector<std::vector<double>> last_a_;
    std::vector<std::vector<double>> deltas_;
    std::vector<std::size_t> size_;
    std::string activation;
    double alpha;
    
    // -------------------- Activation --------------------
    
    double Activate(double x, bool Backward=false) {
        std::string lact = activation;
        for (char& c:lact) c = std::tolower(c);

        if (lact == "sigmoid")        return Sigmoid_(x, Backward);
        if (lact == "tanh")           return Tanh_(x, Backward);
        if (lact == "relu")           return ReLU_(x, Backward);
        if (lact == "leaky-relu")     return Leaky_ReLU(x, Backward);
        if (lact == "soft-plus")      return Soft_Plus_(x, Backward);
        if (lact == "linear")         return Linear_(x, Backward);
        if (lact == "silu")           return SiLU_(x, Backward);
    
        return x;
    }
    
    // -------------------- Sigmoid activation --------------------
    
    double Sigmoid_(double x, bool Backward=false) {
        double tmp = 1.0/(1.0 + std::exp(-x));
        return Backward? tmp*(1.0-tmp):tmp;
    }

    // -------------------- Tanh activation --------------------
    
    double Tanh_(double x, bool Backward=false) {
        double a = std::tanh(x);
        return Backward? 1.0-(a*a):a;
    }

    // -------------------- ReLU activation --------------------
    
    double ReLU_(double x, bool Backward=false) {
        return Backward? (x>0? 1.0:0.0) : std::max(0.0, x);
    }

    // -------------------- Leaky_ReLU_ activation --------------------
    
    double Leaky_ReLU(double x, bool Backward=false) {
        return Backward? (x>0? 1.0:alpha):(x>0? x:alpha*x);
    }

    // -------------------- Soft-Plus activation --------------------
    
    double Soft_Plus_(double x, bool Backward=false) {
        return Backward? Sigmoid_(x) : std::log(1+std::exp(x)) ;
    }

    // -------------------- Linear activation --------------------

    double Linear_(double x, bool Backward=false) {
        return Backward? 1.0 : x;
    }

    // -------------------- Silu activation --------------------

    double SiLU_(double x, bool Backward=false) {
        double sig = Sigmoid_(x);
        if (Backward) {
            return sig + x*sig*(1.0-sig);
        }
        return x * sig;
    }

public:    
    // -------------------- Neuron Network definition --------------------
    
    NeuralNetwork(std::size_t input_size, std::vector<std::size_t> hidden_sizes, std::size_t output_size, std::string activation="Sigmoid", double alpha=0.01)
    {
        size_ = {input_size};
        for (std::size_t hidden: hidden_sizes) {
            size_.push_back(hidden);
        }
        size_.push_back(output_size);
        
        this->activation = activation;
        this->alpha = alpha;
        
        std::mt19937 generator(42);
        std::uniform_real_distribution<double> dist(-1.0,1.0);
        
        for (std::size_t layer=1; layer<size_.size(); layer++) {
            weights_.push_back({});
            biases_.push_back({});
            
            for (std::size_t neuron=0; neuron<size_[layer]; neuron++) {
                weights_[layer].push_back({});
                
                for (std::size_t input=0; input<size_[layer-1];input++)
                {
                    weights_[layer][neuron].push_back(dist(generator));
                }
                
                biases_[layer].push_back(dist(generator));
            }
        }
    }

    // -------------------- Neuron definition --------------------
    double Neuron(const std::vector<double>&   inputs,
        const std::vector<double>&     weights,
        const double                   bias) 
        {                
            double z = bias;

            for (std::size_t i = 0; i < inputs.size(); i++) {
                z += inputs[i] * weights[i];
            }

            double a = Activate(z);
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

    void Backward(const std::vector<double>& targets, double lr){
        deltas_.clear();
        deltas_.resize(size_.size());

        for (std::size_t neuron=0; neuron < size_.back(); neuron++) {
            double delta=0;
            delta = last_a_.back()[neuron] - targets[neuron];
            delta *=  Activate(last_z_.back()[neuron], true);

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

                delta *= Activate(last_z_[layer][neuron], true);
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

    void Train(const std::vector<std::vector<std::vector<double>>>& table, std::size_t epochs, double lr) {
        std::cout << "Training ..." << std::endl;

        auto started_time = std::chrono::high_resolution_clock::now();
        
        std::size_t update_interval = epochs/100;
        if (update_interval == 0) update_interval = 1;

        for (std::size_t epoch=0; epoch < epochs; epoch++){
            for (const std::vector<std::vector<double>>& data_set: table) {
                const std::vector<double>& inputs = data_set[0];
                const std::vector<double>& targets= data_set[1];
                Forward(inputs);
                Backward(targets, lr);
            }

            if (((epoch+1)%update_interval) == 0 || epoch==epochs-1) {
                double progress = (double)(epoch+1)/epochs;
                std::size_t bar_length = 40;
                std::size_t bar_position = (int)(progress*bar_length);
            
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed_t = std::chrono::duration_cast<std::chrono::milliseconds> (now - started_time).count();
                double time_per_epoch = (double)elapsed_t/(epoch+1);
                double estimated_time = time_per_epoch * (epochs - (epoch+1));

                int rem_min = (int)(estimated_time/60000.0);
                int rem_sec = (int)(estimated_time/1000.0) % 60;

                std::cout << "\r[";
                for (std::size_t i=0; i < bar_length; i++) {
                    if (i < bar_position)           std::cout << "█";
                    else if (i == bar_position)     std::cout << "▒";
                    else                            std::cout << " ";
                }

                std::cout << "] " << std::setw(3) << (progress * 100.0) << "%"
                          << " | ETA:" << std::setfill('0') << std::setw(2) << rem_min << ":"
                          << std::setw(2) << rem_sec << std::setfill(' ') << std::flush;
            }

        }
        std::cout << std::endl << std::endl;
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

    NeuralNetwork nn{2, {5, 5}, 1, "sigmoid"};
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