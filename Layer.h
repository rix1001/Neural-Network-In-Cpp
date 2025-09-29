#pragma once
#include "Matrix.h"

class Layer {
	Matrix<double> neurons;
public:
	Layer() = default;

	Layer(const std::vector<double>& neurons) {
		this->neurons = Matrix<double>({ neurons });
	}
	
	Layer(const Matrix<double>& neurons) {
		this->neurons = neurons;
	}

	Layer(const Layer& l) {
		neurons = l.neurons;
	}

	Layer& operator=(const Layer& l) {
		neurons = l.neurons;
		return *this;
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
