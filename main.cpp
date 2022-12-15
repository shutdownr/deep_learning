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
#include <Eigen/Core>

using Eigen::MatrixXd;

std::vector<std::pair<std::string, std::vector<int> > > read_csv(std::string filename){
    // Reads a CSV file into a vector of <string, vector<int>> pairs where
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


void print_vector(std::vector<int> vector) {
    std::copy(vector.begin(), vector.end(),
          std::ostream_iterator<int>(std::cout, " "));
}

void print_column(std::pair<std::string, std::vector<int> > column) {
    std::cout << column.first << ": ";
    print_vector(column.second);
    std::cout << std::endl;
    return;
}

void print_row(std::vector<std::pair<std::string, std::vector<int> > > dataset, int index) {
    std::cout << "Label: " << dataset[0].second[index] << std::endl;
    for (int column = 1; column < dataset.size(); column++) {
        std::cout << dataset[column].second[index];
    }
    std::cout << std::endl;
}

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

static int softmax(std::vector<double> predictions) {
    return std::distance(predictions.begin(), std::max_element(predictions.begin(), predictions.end()));
}

static double relu(double value) {
    return std::max(0.0, value);
}
static double relu_prime(double value) {
    return value > 0 ? 1 : 0;
}

static std::vector<std::vector<double> > transpose(std::vector<std::vector<double> > matrix) {
    // Transpose matrix
    // i - row index; j - column index
    std::vector<std::vector<double> > new_matrix;
    for(int x=0; x < matrix[0].size(); x++) {
        new_matrix.push_back(std::vector<double>(matrix.size()));
    }

    for(int i = 0; i<matrix.size(); i++) {        
        for (int j=0; j<matrix[i].size(); j++) {
            new_matrix[j][i] = matrix[i][j];
        }
    }
    return new_matrix;
}

static double dot(std::vector<double> vector, double scalar) {
    double total = 0;
    for(double value: vector) {
        total += value * scalar;
    }
    return total;
}
static double dot(std::vector<std::vector<double> > matrix, double scalar) {
    double total = 0;
    for(auto row: matrix) {
        for(double value: row) {
            total += value * scalar;
        }
    }
    return total;
}
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

class Node;

class Connection {
    public:
    // Connection is directed from source node -> target node
    // Target node
    Node *target;
    // Source node
    Node *origin;
    // Weight of the connection
    double weight;

    Connection(Node *p_target, Node *p_origin, double p_weight) {
        target = p_target;
        origin = p_origin;
        weight = p_weight;
    }
};

class Node {
    std::string id;
    // Maps previous nodes and their respective weights
    std::vector<Connection*> prev;
    // Maps next nodes and their respective weights
    std::vector<Connection*> next;
    
    public:
    // Activation function used by the node
    std::function<double(double)> activation_function;

    // Bias
    double bias;
    // Output value before calling the activation function
    double intermediate;
    // Output value after calling the activation function
    double output;

    // Constructor
    Node(std::string p_id, std::function<double(double)> p_activation_function, double p_bias = 1) {
        id = p_id;
        activation_function = p_activation_function;
        // TODO: Implement random bias initialization (maybe at a layer basis to allow for different initialization functions?)
        bias = p_bias;
    }

    // Prints node connections
    void print() {
        std::cout << "Node" << id << ":" << std::endl;
        std::cout << "Incoming connections:" << std::endl;
        for(Connection* c: prev) {
            std::cout << c->origin->id << " -> " << c->target->id << std::endl;
        }

        std::cout << "Outgoing connections:" << std::endl;
        for(Connection* c: next) {
            std::cout << c->origin->id << " -> " << c->target->id << std::endl;
        }
    }

    std::vector<double> get_inputs() {
        std::vector<double> inputs;
        for(Connection* conn: prev) {
            inputs.push_back(conn->origin->output);
        }
        return inputs;
    }

    std::vector<double> get_weights() {
        std::vector<double> weights;
        for(Connection* conn: prev) {
            weights.push_back(conn->weight);
        }
        return weights;
    }

    void update_weights(double error) {
        for(Connection* conn: prev) {
            conn-> weight = conn-> weight - error;
        }
    }

    // Transforms inputs from prev nodes with weights, bias and activation function, forward pass
    double feedforward() {
        std::vector<double> inputs = get_inputs();
        std::vector<double> weights = get_weights();

        std::vector<double> weighted_inputs;
        weighted_inputs.reserve(inputs.size());
        std::transform(inputs.begin(), inputs.end(),
                        weights.begin(), std::back_inserter(weighted_inputs), 
                        std::multiplies<double>());

        double output = std::accumulate(weighted_inputs.begin(), weighted_inputs.end(), 0);
        output += bias;
        intermediate = output;
        output = activation_function(output);
        this->output = output;

        return output;
    }

    // Connect this node to another, using a Connection object
    void connect(Node* node) {
        // TODO: Connection has a default weight of 1 (as of now, look into this later and allow for weight initializers)
        Connection* connection = new Connection(node, this, 1.0);
        next.push_back(connection);
        node->prev.push_back(connection);
    }
};

// Abstract layer class
class Layer {
    public:
    std::string id;
    std::vector<Node*> nodes;
    std::function<double(double)> activation_function;
    std::function<double(double)> activation_function_prime;

    virtual void feedforward() = 0;
    virtual std::vector<double> backprop(std::vector<double> error, double learning_rate) = 0;
    virtual void connect(Layer* layer) = 0;

    // Initializes the layer with given inputs. This is to be used for the input layer
    void initialize(std::vector<double> inputs) {
        if(inputs.size() != nodes.size()) {
            std::cout << "WARNING, INPUT SIZE NOT EQUAL TO NODE SIZE, NOT INITIALIZING LAYER " << id << " !!" << std::endl;
            return;
        }
        for(int i = 0; i < inputs.size(); i++) {
            Node* node = nodes[i];
            double input = inputs[i];
            node->output = input;
        }
    }
};

// Fully connected layer
class DenseLayer: public Layer {
    std::vector<double> reverse_activation_function(std::vector<double> error, std::vector<double> intermediates) {
        std::vector<double> output;
        for(int i=0; i<intermediates.size(); i++) {
            output.push_back(activation_function_prime(intermediates[i]) * error[i]);
        }
        return output;
    }
    public:
    DenseLayer(std::string p_id, int number_of_nodes, std::function<double(double)> p_activation_function, std::function<double(double)> p_activation_function_prime) {
        id = p_id;
        activation_function = p_activation_function;
        activation_function_prime = p_activation_function_prime;
        for(int i=0; i<number_of_nodes; i++) {
            Node* new_node = new Node(id + "_" + std::to_string(i), activation_function);
            nodes.push_back(new_node);
        }
    }

    // Forward pass, calls the feedforward method on all nodes
    void feedforward() {
        for(Node* node: nodes) {
            node->feedforward();
        }
    }

    // Backward pass, propagate error and update weights
    std::vector<double> backprop(std::vector<double> error, double learning_rate) {
        // Weight and input matrices
        std::vector<std::vector<double> > weights;
        std::vector<std::vector<double> > inputs;
        std::vector<double> intermediates;
        for(int i=0; i<nodes.size(); i++) {
            auto node = nodes[i];
            weights.push_back(node->get_weights());
            inputs.push_back(node->get_inputs());
            intermediates.push_back(node->intermediate);

            // Update bias of each node
            node->bias -= learning_rate * error[i];
        }

        std::vector<double> output_error = reverse_activation_function(error, intermediates);

        auto weights_t = transpose(weights);
        auto inputs_t = transpose(inputs);

        // Verify this in python
        auto input_error = dot(weights_t, output_error);
        auto weight_error = dot(inputs_t, output_error);

        for(int i=0; i<weight_error.size(); i++) {
            nodes[i]->update_weights(weight_error[i] * learning_rate);
        }
        return input_error;
    }

    // Connect this layer to another, connecting individual nodes
    // This is a DenseLayer, so connect all nodes to all following nodes
    void connect(Layer* layer) {
        auto target_nodes = layer->nodes;
        for(Node* node: nodes) {
            for(Node* target_node: target_nodes) {
                node->connect(target_node);
            }
        }
    }

    // Prints descriptive info
    void print() {
        std::cout << "Layer " << id << ":" << std::endl;
        for(Node* node: nodes) {
            node->print();
            std::cout << std::endl; 
        }
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
        for(int i=0; i<layers.size()-1; i++) {
            Layer* source_layer = layers[i];
            Layer* target_layer = layers[i+1];
            source_layer->connect(target_layer);
        }
    }

    // Forward pass, predict outputs for input vector X
    std::vector<double> forward_pass(std::vector<double> X) {
        // Initialize the input layer
        layers.front()->initialize(X);
        
        // Feed forward
        for(Layer* layer: layers) {
            layer->feedforward();
        }

        // Return outputs of each node 
        Layer* output_layer = layers.back();
        std::vector<Node*> nodes = output_layer->nodes;
        std::vector<double> outputs;
        outputs.reserve(nodes.size());
        std::transform(nodes.begin(), nodes.end(), std::back_inserter(outputs), [](Node* node) { return node->output; });

        // Check whether outputs have to be rescaled
        // Do that here if needed

        return outputs;        
    }

    // Backward pass, update weights and biases based on loss and learning rate
    void backward_pass(std::vector<double> error, double learning_rate) {
        // Pass through layers in reverse order
        for(int i = layers.size() - 1; i>=0; i--) {
            layers[i]->backprop(error, learning_rate);
        }
    }


    // Use mse as a loss for now, add more losses later / if necessary
    // Calculates loss based on predicted outputs and true label
    double loss_function(std::vector<double> outputs, std::vector<double> y_true) {
        double squared_delta_sum = 0;
        for(int i=0; i<outputs.size(); i++) {
            double delta = outputs[i] - y_true[i];
            delta *= delta;
            squared_delta_sum += delta;
        }
        double loss = squared_delta_sum / outputs.size();
        return loss;
    }
    std::vector<double> loss_function_prime(std::vector<double> outputs, std::vector<double> y_true) {
        std::vector<double> losses;
        for(int i=0; i<outputs.size(); i++) {
            losses.push_back(2 * (outputs[i] - y_true[i]) / outputs.size());
        }
        return losses;
    }

    // Trains the network based on X and y 
    // TODO: Implement train / test split
    void train(std::vector<std::vector<double> > X, std::vector<std::vector<double>> y) {
        // Run passes once for now
        auto outputs = forward_pass(X[0]);
        auto errors = loss_function_prime(outputs, y[0]);
        backward_pass(errors, learning_rate);
    }
};



int main()
{
    NeuralNetwork nn = NeuralNetwork();
    DenseLayer l1 = DenseLayer("1", 2, relu, relu_prime);
    DenseLayer l2 = DenseLayer("2", 4, relu, relu_prime);
    // Change this later to allow for softmax as an output function
    DenseLayer l3 = DenseLayer("3", 10, relu, relu_prime);

    nn.layers.push_back(&l1);
    nn.layers.push_back(&l2);
    nn.layers.push_back(&l3);

    nn.connect();
    l2.print();
    
    
}


