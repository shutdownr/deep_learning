#include "algebra.h"

#include <string>
#include <fstream>
#include <vector>
#include <utility>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <array>
#include <algorithm>
#include <map>
#include <cmath>
#include <numeric>
#include <iterator>
#include <functional>

std::vector<std::pair<std::string, std::vector<int> > > read_csv(std::string filename){
    // Reads a CSV file into a vector of <string, vector<int> > pairs where
    // each pair represents <column name, column values>

    // Create a vector of <string, int vector> pairs to store the result
    std::vector<std::pair<std::string, std::vector<int> > > result;
    // Create an input filestream
    std::ifstream myFile(filename);
    // Make sure the file is open
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");
    // Helper vars
    std::string line, colname;
    int val;
    // Read the column names
    if(myFile.good())
    {
        // Extract the first line in the file
        std::getline(myFile, line);
        // Create a stringstream from line
        std::stringstream ss(line);
        // Extract each column name
        while(std::getline(ss, colname, ',')){         
            // Initialize and add <colname, int vector> pairs to result
            result.push_back({colname, std::vector<int> {}});
        }
    }

    // Read data, line by line
    while(std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);
        // Keep track of the current column index
        int colIdx = 0;
        // Extract each integer
        while(ss >> val){
            // Add the current integer to the 'colIdx' column's values vector
            result.at(colIdx).second.push_back(val);
            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();
            // Increment the column index
            colIdx++;
        }
    }

    // Close file
    myFile.close();

    return result;
}

// Convenience methods


// Prints a vector
void print_vector(std::vector<double> vector) {
    std::copy(vector.begin(), vector.end(),
          std::ostream_iterator<double>(std::cout, " "));
    std::cout << std::endl;
}
void print_vector(std::vector<int> vector) {
    std::copy(vector.begin(), vector.end(),
          std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
}
// Prints a matrix
void print_matrix(std::vector<std::vector<double> > matrix) {
    for(int i = 0; i<matrix.size(); i++) {
        print_vector(matrix[i]);
        std::cout << std::endl;
    }
}
void print_matrix(std::vector<std::vector<int> > matrix) {
    for(int i = 0; i<matrix.size(); i++) {
        print_vector(matrix[i]);
    }
}

// Prints a column of the dataset
void print_column(std::pair<std::string, std::vector<int> > column) {
    std::cout << column.first << ": ";
    print_vector(column.second);
}

// Prints a row of the dataset
void print_row(std::vector<std::pair<std::string, std::vector<int> > > dataset, int index) {
    std::cout << "Label: " << dataset[0].second[index] << std::endl;
    for (int column = 1; column < dataset.size(); column++) {
        std::cout << dataset[column].second[index];
    }
    std::cout << std::endl;
}

// Prints a full number (graphical plot)
void plot_number(std::vector<std::pair<std::string, std::vector<int> > > dataset, int index) {
    int dataset_width = 28;
    std::cout << "Label: " << dataset[0].second[index] << std::endl;
    for (int column = 1; column < dataset.size(); column++) {
        int value = dataset[column].second[index];
        bool is_blank = value < 50;
        char output = is_blank ? '.' : '8';
        std::cout << output;
        if (column % dataset_width == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

// Apply function to each element of a vector
static std::vector<double> apply(std::vector<double> vector, std::function<double(double)> function) {
    std::vector<double> output;
    for(int i=0; i<vector.size(); i++) {
        output.push_back(function(vector[i]));
    }
    return output;
}
// Apply function to each element of a matrix
static std::vector<std::vector<double> > apply(std::vector<std::vector<double> > matrix, std::function<double(double)> function) {
    std::vector<std::vector<double> > output;
    for(int i=0; i<matrix.size(); i++) {
        output.push_back(apply(matrix[i], function));
    }
    return output;
}


// Activation functions
// Rectified linear unit, max(x,0)
static double relu(double value) {
    return value > 0 ? value : 0;
}
static double relu_prime(double value) {
    return value > 0 ? 1 : 0;
}
// Hyperbolic tangent
// The non-prime function can be called with tanh(), so it's not implemented here
static double tanh_prime(double value) {
    return 1 - pow(tanh(value), 2);
}

// Sigmoid
static double sigmoid(double value) {
    return 1 / (1 + exp(-value));
}
static double sigmoid_prime(double value) {
    double sigm = sigmoid(value);
    return sigm * (1 - sigm);
}

// Loss functions
// Mean squared error
double mse(std::vector<std::vector<double> > outputs, std::vector<std::vector<double> > y_true) {
    auto deltas = subtract(outputs, y_true);
    auto squared_deltas = power(deltas, 2);
    double squared_delta_sum = sum(squared_deltas);
    return squared_delta_sum / size(outputs);
}
std::vector<std::vector<double> > mse_prime(std::vector<std::vector<double> > outputs, std::vector<std::vector<double> > y_true) {
    auto delta = subtract(outputs, y_true);
    double total_size = y_true.size() * y_true[0].size();
    delta = multiply(delta, 2/total_size);
    return delta;
}


// NN classes

// Abstract layer class
class Layer {
    public:
    // Layer id
    std::string id;
    // Activation function and its prime, used in forward pass / backward pass
    std::function<double(double)> activation_function;
    std::function<double(double)> activation_function_prime;

    std::vector<std::vector<double> > inputs;

    std::vector<std::vector<double> > weights;
    std::vector<std::vector<double> > biases;

    std::vector<std::vector<double> > intermediates;
    std::vector<std::vector<double> > outputs;

    double learning_rate;

    virtual std::vector<std::vector<double> > forward_pass(std::vector<std::vector<double> > inputs) = 0;
    virtual std::vector<std::vector<double> > backward_pass(std::vector<std::vector<double> > error) = 0;
    virtual void print() = 0;
};

// Fully connected layer
class DenseLayer: public Layer {
    public:
    // Constructor
    // TODO: Implement automatic input size detection based on input data for first layer. 
    //       Other layers should have their input size determined in the connect function of the network based on the previous layers.
    DenseLayer(std::string p_id, int number_of_inputs, int number_of_nodes, std::function<double(double)> p_activation_function, std::function<double(double)> p_activation_function_prime) {
        srand((unsigned)time(NULL));
        id = p_id;
        activation_function = p_activation_function;
        activation_function_prime = p_activation_function_prime;
        
        // Initialize weights and bias randomly
        std::vector<double> bias_vector;
        for(int i=0; i<number_of_nodes; i++) {
            double bias = ((double) rand() / (RAND_MAX)) - 0.5;
            bias_vector.push_back(bias);
        }
        // Biases are a 1xnumber_of_nodes matrix
        biases.push_back(bias_vector);

        for(int i=0; i<number_of_inputs; i++) {
            std::vector<double> weight_vector;
            for(int j=0; j<number_of_nodes; j++) {
                double weight = ((double) rand() / (RAND_MAX)) - 0.5;
                weight_vector.push_back(weight);
            }
            weights.push_back(weight_vector);
        }
    }
    
    // Forward pass, dot product of inputs and weights, with bias added and activation function applied, returns outputs
    std::vector<std::vector<double> > forward_pass(std::vector<std::vector<double> > inputs) {
        this->inputs = inputs;

        auto dot_product = dot(inputs, weights);
        auto biased_dot_product = add(dot_product, biases);

        intermediates = biased_dot_product;
        outputs = apply(biased_dot_product, activation_function);
        // Should be fine to return this as a reference, if an issue pops up, deepcopy instead
        return outputs;
    }

    // Backward pass, reverse the activation function (with its prime), update biases and weights and return the updated error
    std::vector<std::vector<double> > backward_pass(std::vector<std::vector<double> > error) {       
        // Reverse the activation function with its prime
        auto reverse_activation = apply(intermediates, activation_function_prime);
        auto reverse_activation_errors = multiply(reverse_activation, error);
        
        // Update biases
        biases = subtract(biases, multiply(reverse_activation_errors, learning_rate));

        // Calculate input error (error which is passed to the previous layer), do this before updating weights!
        auto input_error = dot(reverse_activation_errors, transpose(weights));

        // Calculate weight updates
        // Order of dot product matters since the matrices do not have the same shape
        auto weight_error = dot(transpose(inputs), reverse_activation_errors);
        weights = subtract(weights, multiply(weight_error, learning_rate));

        return input_error;
    }

    // Prints descriptive info
    void print() {
        std::cout << "Layer " << id << ":" << std::endl;
        std::cout << "Inputs:" << std::endl;
        print_matrix(inputs);
        std::cout << "Weights:" << std::endl;
        print_matrix(weights);
        std::cout << "Biases:" << std::endl;
        print_matrix(biases);
        std::cout << "Intermediates:" << std::endl;
        print_matrix(intermediates);
        std::cout << "Outputs:" << std::endl;
        print_matrix(outputs);
    }
};


// Neural network class, contains a vector of layers
class NeuralNetwork {
    public:

    // Layers within the network, order represents the data flow ([0] is the input layer. [-1] is the output layer)
    std::vector<Layer*> layers;    
    // Learning rate, this is passed to the layers in connect(), the network itself doesn't use the learning rate
    double learning_rate;
    // Calculates loss based on predicted outputs and true label, used for output
    std::function<double(std::vector<std::vector<double> >, std::vector<std::vector<double> >)> loss_function;
    // Loss function prime, used for backprop
    std::function<std::vector<std::vector<double> >(std::vector<std::vector<double> >, std::vector<std::vector<double> >)> loss_function_prime;

    NeuralNetwork(double p_learning_rate = 0.005, 
                std::function<double(std::vector<std::vector<double> >, std::vector<std::vector<double> >)> p_loss_function = mse, 
                std::function<std::vector<std::vector<double> >(std::vector<std::vector<double> >, std::vector<std::vector<double> >)> p_loss_function_prime = mse_prime) {
        learning_rate = p_learning_rate;
        loss_function = p_loss_function;
        loss_function_prime = p_loss_function_prime;
    }

    // Connects the entire network, call after finalizing the layers
    void connect() {
        if(layers.size() <= 1) {
            std::cout << "WARNING, LESS THAN TWO LAYERS, NOT CONNECTING THE NETWORK!!" << std::endl;
            return;
        }
        for(Layer* layer: layers) {
            layer->learning_rate = learning_rate;
        }
    }

    // Forward pass, predict outputs for input matrix, inputs
    std::vector<std::vector<double> > forward_pass(std::vector<std::vector<double> > inputs) {
        // Feed forward
        for(Layer* layer: layers) {
            inputs = layer->forward_pass(inputs);
        }
        return inputs;
    }

    // Backward pass, update weights and biases based on loss and learning rate
    void backward_pass(std::vector<std::vector<double> > error) {
        // Pass through layers in reverse order
        for(int i = layers.size() - 1; i>=0; i--) {
            error = layers[i]->backward_pass(error);
        }
    }

    // Predicts y for a given X
    std::vector<std::vector<std::vector<double> > > predict(std::vector<std::vector<std::vector<double> > > X) {
        std::vector<std::vector<std::vector<double> > > predictions;
        for(auto input: X) {
            predictions.push_back(forward_pass(input));
        }
        return predictions;
    }

    // Trains the network based on X and y 
    // TODO: Implement train / test split
    void train(std::vector<std::vector<std::vector<double> > > X, std::vector<std::vector<std::vector<double> > > y, int epochs) {
        for (int epoch = 1; epoch <= epochs; epoch++) {
            double total_loss = 0;
            for(int i=0; i<X.size(); i++) {
                auto outputs = forward_pass(X[i]);
                total_loss += loss_function(outputs, y[i]);
                auto errors = loss_function_prime(outputs, y[i]);
                backward_pass(errors);
            }
            // Mean loss
            total_loss /= X.size();
            std::cout << "Epoch " << epoch << "/" << epochs << "; Loss: " << total_loss << std::endl;
        }
    }
};

int main()
{
    // MNIST training case
    auto dataset = read_csv("../data/mnist_train.csv");

    std::vector<int> y_input;
    std::vector<std::vector<std::vector<double> > > X;
    for(int index=0; index<dataset[0].second.size(); index++) { 
        y_input.push_back(dataset[0].second[index]);
        std::vector<double> X_values;
        for(int column=1; column<dataset.size(); column++) {
            // Rescale the data to values between 0 and 1 by dividing by 255 
            X_values.push_back(dataset[column].second[index] / 255.0);
        }
        std::vector<std::vector<double> > X_matrix;
        X_matrix.push_back(X_values);
        X.push_back(X_matrix);
    }

    // Turn y into categoricals (sparse)
    std::vector<std::vector<std::vector<double> > > y;
    for(int y_value: y_input) {
        // i from 0 to 9 to include all digits
        std::vector<double> y_value_sparse;
        for(int i=0; i<10; i++) {
            y_value_sparse.push_back(i==y_value ? 1 : 0);
        }
        std::vector<std::vector<double> > y_matrix;
        y_matrix.push_back(y_value_sparse);
        y.push_back(y_matrix);
    }

    std::vector<std::vector<std::vector<double> > > X_train;
    std::vector<std::vector<std::vector<double> > > X_test;
    std::vector<std::vector<std::vector<double> > > y_train;
    std::vector<std::vector<std::vector<double> > > y_test;
    for(int i=0; i<1000; i++) {
        X_train.push_back(X[i]);
        y_train.push_back(y[i]);
    }
    for(int i=1000; i<1005; i++) {
        X_test.push_back(X[i]);
        y_test.push_back(y[i]);
    }


    // Simple XOR training case
    // std::vector<std::vector<std::vector<double> > > X = {{{0,0}}, {{0,1}}, {{1,1}}, {{1,0}}};
    // std::vector<std::vector<std::vector<double> > > y = {{{0}}, {{1}}, {{0}}, {{1}}};
    // std::vector<std::vector<std::vector<double> > > X_test = {{{0,0}}, {{1,1}}, {{1,0}}, {{0,1}}};

    // Use a pretty high learning rate for now to compensate for smaller training set size
    NeuralNetwork nn = NeuralNetwork(0.1, mse, mse_prime);

    // TODO: Sigmoid and tanh work, relu doesn't, investigate this
    DenseLayer l1 = DenseLayer("1", 28*28, 100, sigmoid, sigmoid_prime);
    DenseLayer l2 = DenseLayer("2", 100, 50, sigmoid, sigmoid_prime);
    DenseLayer l3 = DenseLayer("3", 50, 10, sigmoid, sigmoid_prime);
    nn.layers.push_back(&l1);
    nn.layers.push_back(&l2);
    nn.layers.push_back(&l3);


    nn.connect();
    std::cout << "training nn..." << std::endl;
    nn.train(X_train, y_train, 30);

    auto predictions = nn.predict(X_test);
    for(int i=0; i<predictions.size(); i++) {
        print_matrix(y_test[i]);
        print_matrix(predictions[i]);
    }
}