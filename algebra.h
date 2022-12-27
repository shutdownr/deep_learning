#include <cmath>
#include <functional>
#include <vector>
#include <stdexcept>

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

// Computes the size of a matrix (number of elements) 
static int size(std::vector<std::vector<double> > matrix) {
    int size = 0;
    for(auto vector: matrix) {
        size += vector.size();
    }
    return size;
}

// Computes the sum of a vector
static double sum(std::vector<double> vector) {
    double sum = 0;
    for(double value: vector) {
        sum += value;
    }
    return sum;
}
// Computes the sum of a matrix
static double sum(std::vector<std::vector<double> > matrix) {
    double total_sum = 0;
    for(auto vector: matrix) {
        total_sum += sum(vector);
    }
    return total_sum;
}

// Adds two vectors
static std::vector<double> add(std::vector<double> a, std::vector<double> b) {
    std::vector<double> output;
    for(int i=0; i<a.size(); i++) {
        output.push_back(a[i] + b[i]);
    }
    return output;
}
// Adds a vector to a matrix
static std::vector<std::vector<double> > add(std::vector<std::vector<double> > matrix, std::vector<double> vector) {
    std::vector<std::vector<double> > output;
    for(int i=0; i<matrix.size(); i++) {
        output.push_back(add(matrix[i], vector));
    }
    return output;
}
// Adds matrix a to matrix b
static std::vector<std::vector<double> > add(std::vector<std::vector<double> > a, std::vector<std::vector<double> > b) {
    if (a.size() == b.size()) {
        std::vector<std::vector<double> > output;
        for(int i=0; i<a.size(); i++) {
            output.push_back(add(a[i], b[i]));
        }
        return output;
    } else if(b.size() == 1) {
        return add(a, b[0]);
    }
    throw std::invalid_argument("Matrix sizes don't align");
}
// Subtracts vector b from vector a
static std::vector<double> subtract(std::vector<double> a, std::vector<double> b) {
    std::vector<double> output;
    for(int i=0; i<a.size(); i++) {
        output.push_back(a[i] - b[i]);
    }
    return output;
}
// Subtracts a vector from a matrix
static std::vector<std::vector<double> > subtract(std::vector<std::vector<double> > matrix, std::vector<double> vector) {
    std::vector<std::vector<double> > output;
    for(int i=0; i<matrix.size(); i++) {
        output.push_back(subtract(matrix[i], vector));
    }
    return output;
}
// Subtracts matrix a from matrix b
static std::vector<std::vector<double> > subtract(std::vector<std::vector<double> > a, std::vector<std::vector<double> > b) {
    if (a.size() == b.size()) {
        std::vector<std::vector<double> > output;
        for(int i=0; i<a.size(); i++) {
            output.push_back(subtract(a[i], b[i]));
        }
        return output;
    } else if(b.size() == 1) {
        return subtract(a, b[0]);
    }
    throw std::invalid_argument("Matrix sizes don't align");
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
    for(auto vector: matrix) {
        output.push_back(multiply(vector, scalar));
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

// Power function of a vector
static std::vector<double> power(std::vector<double> vector, double exponent) {
    std::vector<double> output;
    for(double value: vector) {
        output.push_back(pow(value, exponent));
    }
    return output;
}

// Power function of a matrix
static std::vector<std::vector<double> > power(std::vector<std::vector<double> > matrix, double exponent) {
    std::vector<std::vector<double> > output;
    for(auto vector: matrix) {
        output.push_back(power(vector, exponent));
    }
    return output;
}