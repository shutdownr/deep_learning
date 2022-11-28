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

static int softmax(std::vector<int> predictions) {
    return std::distance(predictions.begin(), std::max_element(predictions.begin(), predictions.end()));
}

static double relu(double value) {
    return std::max(0.0, value);
}

class Node {
    public:

    int id;
    std::map<Node, int> prev;
    std::map<Node, int> next;
};

class Graph {
    std::vector<Node> nodes;
};

class Layer {
    public:


};



class NeuralNetwork {
    public:

    std::vector<Layer> layers;

    void train(std::vector<std::vector<int> > X, std::vector<int> y) {

    }
};



int main()
{
    // int a = 5;
    // std::string b = test(a);
    // auto data = read_csv("data/mnist_train.csv");
    // calls Sum::operator() for each number
    // plot_number(data, 1);
    std::vector<int> vect { 10, 20, 30 };

    std::cout << softmax(vect);
    // std::for_each(data.begin(), data.end(), unpack_csv);
 
    // std::cout << data << std::endl;
}


