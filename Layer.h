#pragma once
#include "Matrix.h"

class Layer {
	Matrix<double> neurons;
public:
	Layer() = default;

	Layer(std::vector<double> neurons) {
		std::vector<std::vector<double>> mat(1,std::vector<double>(neurons.size()));
		mat[0] = neurons;

		this->neurons = Matrix<double>(mat);
	}
	
	Layer(Matrix<double> neurons) {
		this->neurons = neurons;
	}

	Layer(const Layer& l) {
		neurons = l.neurons;
	}

	int numOfNeurons() const {
		return neurons.cols();
	}

	Matrix<double>& getNeurons() {
		return neurons;
	}

	double& operator[](int i) {
		return neurons[0][i];
	}

	void print() const {
		std::cout << neurons;
	}
};
