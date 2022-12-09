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


static std::string test(int &number) {
    return std::to_string(number);
}

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
    
    // Activation function used by the node
    std::function<double(double)> activation_function;
    // Bias (constant)
    double bias = 1;

    double output;

    // Constructor
    public:
    Node(std::string p_id, std::function<double(double)> p_activation_function, double p_bias = 1) {
        id = p_id;
        activation_function = p_activation_function;
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

    // Transforms inputs from prev nodes with weights, bias and activation function, forward pass
    double feedforward() {
        double output = 0;
        for(Connection* prev_node: prev) {
            double weight = prev_node->weight;
            double input = prev_node->origin->output;
            output += weight * input;
        }
        output += this->bias;
        output = this->activation_function(output);
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

class Layer {
    public:
    std::string id;
    std::vector<Node*> nodes;
    std::function<double(double)> activation_function;

    virtual void feedforward() = 0;
    virtual void connect(Layer* layer) = 0;
};

class DenseLayer: public Layer {
    public:
    DenseLayer(std::string p_id, int number_of_nodes, std::function<double(double)> p_activation_function) {
        id = p_id;
        activation_function = p_activation_function;
        for(int i=0; i<number_of_nodes; i++) {
            Node* new_node = new Node(id + "_" + std::to_string(i), activation_function);
            nodes.push_back(new_node);
        }
    }

    void print() {
        std::cout << "Layer " << id << ":" << std::endl;
        for(Node* node: nodes) {
            node->print();
            std::cout << std::endl; 
        }
    }

    void feedforward() {
        for(Node* node: nodes) {
            node->feedforward();
        }
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
};



class NeuralNetwork {
    public:

    std::vector<Layer*> layers;

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

    void forward_pass() {
        for(Layer* layer: layers) {
            layer->feedforward();
        }
    }

    void train(std::vector<std::vector<int> > X, std::vector<int> y) {

    }
};



int main()
{
    NeuralNetwork nn = NeuralNetwork();
    DenseLayer l1 = DenseLayer("1", 2, relu);
    DenseLayer l2 = DenseLayer("2", 4, relu);
    // Change this later to allow for softmax as an output function
    DenseLayer l3 = DenseLayer("3", 10, relu);

    nn.layers.push_back(&l1);
    nn.layers.push_back(&l2);
    nn.layers.push_back(&l3);

    nn.connect();
    l2.print();
}


