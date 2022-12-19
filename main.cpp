#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream
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

// Prints a full number
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

// Activation functions
// Rectified linear unit, max(x,0)
static double relu(double value) {
    return std::max(0.0, value);
}
static double relu_prime(double value) {
    return value > 0 ? 1 : 0;
}
// Hyperbolic tangent
// The non-prime function can be called with tanh(), so it's not implemented here
static double tanh_prime(double value) {
    return 1 - pow(tanh(value), 2);
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


// Linear algebra operations
// Transpose matrix
static std::vector<std::vector<double> > transpose(std::vector<std::vector<double> > matrix) {
    std::vector<std::vector<double> > new_matrix;
    for(int x=0; x < matrix[0].size(); x++) {
        new_matrix.push_back(std::vector<double>(matrix.size()));
    }

    // i - row index; j - column index
    for(int i = 0; i<matrix.size(); i++) {        
        for (int j=0; j<matrix[i].size(); j++) {
            new_matrix[j][i] = matrix[i][j];
        }
    }
    return new_matrix;
}

// Adds two vectors
static std::vector<double> add(std::vector<double> a, std::vector<double> b) {
    std::vector<double> output;
    for(int i=0; i<a.size(); i++) {
        output.push_back(a[i] + b[i]);
    }
    return output;
}
// Adds matrix a to matrix b
static std::vector<std::vector<double> > add(std::vector<std::vector<double> > a, std::vector<std::vector<double> > b) {
    std::vector<std::vector<double> > output;
    for(int i=0; i<a.size(); i++) {
        output.push_back(add(a[i], b[i]));
    }
    return output;
}
// Subtracts vector b from vector a
static std::vector<double> subtract(std::vector<double> a, std::vector<double> b) {
    std::vector<double> output;
    for(int i=0; i<a.size(); i++) {
        output.push_back(a[i] - b[i]);
    }
    return output;
}
// Subtracts matrix a from matrix b
static std::vector<std::vector<double> > subtract(std::vector<std::vector<double> > a, std::vector<std::vector<double> > b) {
    std::vector<std::vector<double> > output;
    for(int i=0; i<a.size(); i++) {
        output.push_back(subtract(a[i], b[i]));
    }
    return output;
}
// Multiplies a vector by a scalar
static std::vector<double> multiply(std::vector<double> vector, double scalar) {
    std::vector<double> output;
    for(double value: vector) {
        output.push_back(value * scalar);
    }
    return output;
}
// Multiplies two vectors
static std::vector<double> multiply(std::vector<double> a, std::vector<double> b) {
    std::vector<double> output;
    for(int i=0; i<a.size(); i++) {
        output.push_back(a[i] * b[i]);
    }
    return output;
}
// Multiplies a matrix by a scalar
static std::vector<std::vector<double> > multiply(std::vector<std::vector<double> > matrix, double scalar) {
    std::vector<std::vector<double> > output;
    for(int i=0; i<matrix.size(); i++) {
        output.push_back(multiply(matrix[i], scalar));
    }
    return output;
}
// Multiplies two matrices (element-wise multiplication)
static std::vector<std::vector<double> > multiply(std::vector<std::vector<double> > a, std::vector<std::vector<double> > b) {
    std::vector<std::vector<double> > output;
    for(int i=0; i<a.size(); i++) {
        output.push_back(multiply(a[i], b[i]));
    }
    return output;
}

// Dot product of a vector and a scalar
static double dot(std::vector<double> vector, double scalar) {
    double total = 0;
    for(double value: vector) {
        total += value * scalar;
    }
    return total;
}
// Dot product of two vectors
static double dot(std::vector<double> a, std::vector<double> b) {
    double sum = 0;
    for(int i=0; i<a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
// Dot product of a matrix and a scalar
static double dot(std::vector<std::vector<double> > matrix, double scalar) {
    double total = 0;
    for(auto row: matrix) {
        for(double value: row) {
            total += value * scalar;
        }
    }
    return total;
}
// Dot product of a matrix and a vector 
static std::vector<double> dot(std::vector<std::vector<double> > matrix, std::vector<double> vector) {
    std::vector<double> output;
    for(int i=0; i<matrix.size(); i++) {
        double row_total = 0;
        auto row = matrix[i];
        auto scalar = vector[i];
        if(row.size() != vector.size()) {
            std::cout << "WARNING, VECTOR AND MATRIX ARE NOT EQUAL SIZE!";
        }
        for(double value: row) {
            row_total += value * scalar;
        }
        output.push_back(row_total);
    }
    return output;
}
// Dot product of a matrix and a matrix, number of columns of a must be equal to number of rows of b 
static std::vector<std::vector<double> > dot(std::vector<std::vector<double> > a, std::vector<std::vector<double> > b) {
    std::vector<std::vector<double> > output;
    // Row index
    for(int i=0; i<a.size(); i++) {
        auto row = a[i];
        output.push_back(std::vector<double>());
        // Column index
        for(int j=0; j<b[0].size(); j++) {
            std::vector<double> column;
            for(int k=0; k<b.size(); k++) {
                column.push_back(b[k][j]);
            }
            output[i].push_back(dot(row, column));
        }
    }
    return output;
}

// NN classes

// Abstract layer class
class Layer {
    public:
    std::string id;
    // std::vector<Node*> nodes;
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
        std::vector<double> bias_vector;
        // Initialize weights and bias randomly
        for(int i=0; i<number_of_nodes; i++) {
            double bias = ((double) rand() / (RAND_MAX)) - 0.5;
            bias_vector.push_back(bias);
        }
        for(int i=0; i<number_of_inputs; i++) {
            std::vector<double> weight_vector;
            for(int j=0; j<number_of_nodes; j++) {
                double weight = ((double) rand() / (RAND_MAX)) - 0.5;
                weight_vector.push_back(weight);
            }
            weights.push_back(weight_vector);
        }
        // Biases are a 1xnumber_of_nodes matrix
        biases.push_back(bias_vector);
    }

    // Forward pass, calls the forward_pass method on all nodes, returns outputs
    std::vector<std::vector<double> > forward_pass(std::vector<std::vector<double> > inputs) {
        this->inputs = inputs;

        auto dot_product = dot(inputs, weights);
        auto biased_dot_product = add(dot_product, biases);

        intermediates = biased_dot_product;
        outputs = apply(biased_dot_product, activation_function);
        // Should be fine to return this as a reference, if an issue pops up, deepcopy instead
        return outputs;
    }

    // Backward pass, propagate error and update weights
    std::vector<std::vector<double> > backward_pass(std::vector<std::vector<double> > error) {
        // Reverse the activation function with its prime
        auto reverse_activation = apply(intermediates, activation_function_prime);
        auto reverse_activation_errors = multiply(reverse_activation, error);
        // Update biases

        biases = subtract(biases, multiply(reverse_activation_errors, learning_rate));

        // Calculate weight updates
        // Order of dot product matters since the matrices do not have the same shape
        auto weight_error = dot(transpose(inputs), reverse_activation_errors);
        weights = subtract(weights, multiply(weight_error, learning_rate));

        // Calculate input error (error which is passed to the previous layer)
        auto input_error = dot(reverse_activation_errors, transpose(weights));
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



class NeuralNetwork {
    public:
    std::vector<Layer*> layers;    
    double learning_rate;

    NeuralNetwork(double p_learning_rate = 0.005) {
        learning_rate = p_learning_rate;
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

    // Forward pass, predict outputs for input vector X
    std::vector<std::vector<double> > forward_pass(std::vector<std::vector<double> > inputs) {
        // Feed forward
        for(Layer* layer: layers) {
            inputs = layer->forward_pass(inputs);
        }

        // TODO: Check whether outputs have to be rescaled
        // Do that here if needed

        return inputs;
    }

    // Backward pass, update weights and biases based on loss and learning rate
    void backward_pass(std::vector<std::vector<double> > error) {
        // Pass through layers in reverse order
        for(int i = layers.size() - 1; i>=0; i--) {
            error = layers[i]->backward_pass(error);
        }
    }

    // TODO: Implement predict function

    // TODO: Move loss function outside
    // Use mse as a loss for now, add more losses later / if necessary
    // Calculates loss based on predicted outputs and true label
    double loss_function(std::vector<std::vector<double> > outputs, std::vector<std::vector<double> > y_true) {
        // TODO: Rework this to use a functional api instead of nested for loops
        double squared_delta_sum = 0;
        int total_size = 0;
        for(int i=0; i<outputs.size(); i++) {
            total_size += outputs[i].size();
            for(int j=0; j< outputs[i].size(); j++) {
                double delta = outputs[i][j] - y_true[i][j];
                delta *= delta;
                squared_delta_sum += delta;
            }
        }
        double loss = squared_delta_sum / total_size;
        return loss;
    }

    std::vector<std::vector<double> > loss_function_prime(std::vector<std::vector<double> > outputs, std::vector<std::vector<double> > y_true) {
        auto delta = subtract(outputs, y_true);
        double total_size = y_true.size() * y_true[0].size();
        delta = multiply(delta, 2/total_size);
        return delta;
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
    // auto dataset = read_csv("../data/mnist_train.csv");

    // std::vector<int> y_input;
    // std::vector<std::vector<std::vector<double> > > X;

    // for(int index=0; index<dataset.size(); index++) { 
    //     y_input.push_back(dataset[0].second[index]);
    //     std::vector<double> X_values;
    //     for(int column=1; column<dataset.size(); column++) {
    //         X_values.push_back(dataset[column].second[index]);
    //     }
    //     std::vector<std::vector<double> > X_matrix;
    //     X_matrix.push_back(X_values);
    //     X.push_back(X_matrix);
    // }

    // // Turn y into categoricals (sparse)
    // std::vector<std::vector<std::vector<double> > > y;
    // for(int y_value: y_input) {
    //     // i from 0 to 9 to include all digits
    //     std::vector<double> y_value_sparse;
    //     for(int i=0; i<10; i++) {
    //         y_value_sparse.push_back(i==y_value ? 1 : 0);
    //     }
    //     std::vector<std::vector<double> > y_matrix;
    //     y_matrix.push_back(y_value_sparse);
    //     y.push_back(y_matrix);
    // }

    // std::vector<std::vector<std::vector<double> > > X_train;
    // std::vector<std::vector<std::vector<double> > > y_train;
    // for(int i=0; i<500; i++) {
    //     X_train.push_back(X[i]);
    //     y_train.push_back(y[i]);
    // }

    // Simple XOR training case
    std::vector<std::vector<std::vector<double> > > X = {{{0,0}}, {{0,1}}, {{1,1}}, {{1,0}}};
    std::vector<std::vector<std::vector<double> > > y = {{{0}}, {{1}}, {{0}}, {{1}}};

    NeuralNetwork nn = NeuralNetwork(0.1);

    DenseLayer l1 = DenseLayer("1", 2, 5, tanh, tanh_prime);
    DenseLayer l2 = DenseLayer("2", 5, 1, tanh, tanh_prime);
    nn.layers.push_back(&l1);
    nn.layers.push_back(&l2);


    nn.connect();
    std::cout << "training nn..." << std::endl;
    nn.train(X, y, 1000);
}