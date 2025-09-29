#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Layer.h"
#include <random>
#include <algorithm>
#include <fstream>


#define LEARNING_RATE 0.001
#define BATCH_SIZE 100
#define EPOCHS 1
#define ALPHA 0.01


//ACTIVATION FUNCTIONS

inline double ReLU(const double& x) {
	return std::max(0.0, x);
}

inline double leakyReLU(const double& x) {
	return std::max(ALPHA * x, x);
}

inline double softmax(const double& x, const double& sum) {
	return exp(x) / sum;
}


class NeuralNetwork {
	std::vector<Layer> layers;
	std::vector<Matrix<double>> weights;
	std::vector<Matrix<double>> biases;
	std::vector<double> target;

	class invalid_size {};
public:
	NeuralNetwork(const std::vector<int>& topology) {
		int size = topology.size();

		layers.resize(size);

		weights.resize(size - 1);
		biases.resize(size - 1);

		std::random_device rd;
		std::mt19937 mt(rd());

		double val;

		for (int i = 0;i < size - 1;i++) {
			weights[i] = Matrix<double>(topology[i], topology[i + 1]);

			val = sqrt(2.0 / topology[i]);                    //He normal inicialization
			std::normal_distribution<double> ud(0.0, val);


			for (int r = 0;r < topology[i];r++) {
				for (int c = 0;c < topology[i + 1];c++) {
					weights[i][r][c] = ud(mt);
				}
			}

			biases[i] = Matrix<double>(1, topology[i + 1]);
		}
	}

	NeuralNetwork(const NeuralNetwork& nn) {
		layers = nn.layers;
		weights = nn.weights;
		biases = nn.biases;
	}

	void setInput(const std::vector<double>& input) {
		layers[0] = input;
	}

	void setTarget(const std::vector<double>& target) {
		this->target = target;
	}

	void setWeights(const std::vector<Matrix<double>>& weights) {
		this->weights = weights;
	}

	void setBiases(const std::vector<Matrix<double>>& biases) {
		this->biases = biases;
	}

	void printLayers() const {
		std::cout << "Input Layer:\n";
		layers[0].print();

		int L = layers.size();

		for (int i = 1;i < L - 1;i++) {
			std::cout << "Hidden layer #" << i << ":\n";
			layers[i].print();
		}

		std::cout << "Output Layer:\n";

		layers[L - 1].print();
	}


	void printWeights() const {
		for (int i = 0;i < weights.size();i++) {
			std::cout << "WEIGHTS #" << i + 1 << ":\n";
			std::cout << weights[i] << '\n';
		}
	}


	void printBiases() const {
		for (int i = 0;i < biases.size();i++) {
			std::cout << "BIAS #" << i + 1 << ":\n";
			std::cout << biases[i] << '\n';
		}
	}


	void printBiasesAsVectors() {
		int cols;
		
		for (int i = 0;i < biases.size();i++) {
			std::cout << "BIAS #" << i + 1 << ":\n";

			std::cout << "{{";

			cols = biases[i].cols();

			for (int c = 0;c < cols;c++) {
				std::cout << biases[i][0][c];

				if (c < cols - 1) {
					std::cout << ',';
				}
			}

			std::cout << "}}\n";
		}
	}


	void printBiasesAsVectors(const std::string& fileName) {
		int cols;

		std::ofstream out(fileName);

		for (int i = 0;i < biases.size();i++) {
			out << "BIAS #" << i + 1 << ":\n";

			out << "{{";

			cols = biases[i].cols();

			for (int c = 0;c < cols;c++) {
				out << biases[i][0][c];

				if (c < cols - 1) {
					out << ',';
				}
			}

			out << "}}\n";
		}
	}

	void printWeightsAsVectors() {
		int rows, cols;
		int w = weights.size();


		for (int i = 0;i < w;i++) {
			rows = weights[i].rows();
			cols = weights[i].cols();

			std::cout << "====================\n";
			std::cout << "WEIGHTS #" << i + 1 << '\n';
			std::cout << "====================\n";

			std::cout << '{';

			for (int r = 0;r < rows;r++) {
				std::cout << '{';

				for (int c = 0;c < cols;c++) {
					std::cout << weights[i][r][c];

					if (c != cols - 1) {
						std::cout << ',';
					}
					else {
						std::cout << "}";
					}
				}

				if (r != rows - 1) {
					std::cout << ',';
				}
				else {
					std::cout << '}';
				}

				std::cout << '\n';
			}
		}
	}

	void printWeightsAsVectors(const std::string& fileName) {
		int rows, cols;
		int w = weights.size();

		std::ofstream out(fileName);

		for (int i = 0;i < w;i++) {
			rows = weights[i].rows();
			cols = weights[i].cols();

			out << "====================\n";
			out << "WEIGHTS #" << i + 1 << '\n';
			out << "====================\n";

			out << '{';

			for (int r = 0;r < rows;r++) {
				out << '{';

				for (int c = 0;c < cols;c++) {
					out << weights[i][r][c];

					if (c != cols - 1) {
						out << ',';
					}
					else {
						out << "}";
					}
				}

				if (r != rows - 1) {
					out << ',';
				}
				else {
					out << '}';
				}

				out << '\n';
			}
		}
	}

	void printOutput() {
		int L = layers.size();

		std::cout << "-------------------------\n";

		layers[L - 1].print();

		std::cout << '\n';

		int n = layers[L - 1].numOfNeurons();

		int idx = -1;
		double max = 0;

		for (int i = 0;i < n;i++) {
			if (layers[L - 1][i] > max) {
				max = layers[L - 1][i];
				idx = i;
			}
		}

		std::cout << "Most activated neuron: " << idx << ' ' << max * 100 << "%\n";
		std::cout << "-------------------------\n";

	}

	int mostActivatedNeuron() {
		int L = layers.size();

		int n = layers[L - 1].numOfNeurons();

		int idx = -1;
		double max = 0;

		for (int i = 0;i < n;i++) {
			if (layers[L - 1][i] > max) {
				max = layers[L - 1][i];
				idx = i;
			}
		}

		return idx;
	}

	double mostActivatedNeuronPercentage() {
		int L = layers.size();

		int n = layers[L - 1].numOfNeurons();

		double max = 0;

		for (int i = 0;i < n;i++) {
			if (layers[L - 1][i] > max) {
				max = layers[L - 1][i];
			}
		}

		return max * 100;
	}

	void feedForward() {
		int size = layers.size();

		int cols;

		for (int i = 0;i < size - 2;i++) {  //all the hidden layers will use the leaky ReLU activation function
			layers[i + 1] = (layers[i].getNeurons() * weights[i]) + biases[i];

			cols = layers[i + 1].numOfNeurons();

			for (int c = 0;c < cols;c++) {
				layers[i + 1][c] = leakyReLU(layers[i + 1][c]);
			}
		}


		//output layer will use the softmax activation function

		layers[size - 1] = (layers[size - 2].getNeurons() * weights[size - 2]) + biases[size - 2];

		cols = layers[size - 1].numOfNeurons();

		double sum = 0;

		double maximum = layers[size-1][0];

		for (int c = 1;c < cols;c++) {           //finding the neuron with the biggest value
			if (layers[size - 1][c] > maximum) {  
				maximum = layers[size - 1][c];
			}
		}

		for (int c = 0;c < cols;c++) {
			sum += exp(layers[size - 1][c]-maximum);     //preventing possible overflow with the -maximum
		}

		for (int c = 0;c < cols;c++) {
			layers[size - 1][c] = softmax(layers[size - 1][c]-maximum, sum);
		}
	}


	double crossEntropy() {
		int n = layers.size();

		int cols = layers[n - 1].numOfNeurons();

		if (target.size() != cols) {
			throw invalid_size();
		}

		double error = 0;

		double a, b, x;

		for (int c = 0;c < cols;c++) {
			a = target[c];
			b = layers[n - 1][c];

			x = -a * log(b);

			error += x;

		}

		return error;
	}


													//if the update flag is ON, the weights and biases will be
													//updated instantly, otherwise the weight and bias gradients will be returned and
													//the weights and biases can be updated later with the updateParameters method
	std::vector<std::vector<Matrix<double>>> backPropagation(bool update = false) {
		int L = layers.size();
		int cols = layers[L - 1].numOfNeurons();

		double softmaxDerived;
		double errorDerived;
		double leakyReLUderived;

		std::vector<Matrix<double>> weightGradients(L - 1);

		std::vector<Matrix<double>> deltas(L - 1);
		deltas[L - 2] = Matrix<double>(1, cols);

		//OUTPUT LAYER
		for (int c = 0;c < cols;c++) {
			deltas[L - 2][0][c] = layers[L - 1][c] - target[c];
		}


		weightGradients[L - 2] = layers[L - 2].getNeurons().transpose() * deltas[L - 2];  

		//HIDDEN LAYERS
		for (int l = L - 2;l > 0;l--) {
			cols = layers[l].numOfNeurons();

			deltas[l - 1] = deltas[l] * weights[l].transpose();

			for (int c = 0;c < cols;c++) {
				leakyReLUderived = layers[l][c] > 0 ? 1 : ALPHA;

				deltas[l - 1][0][c] *= leakyReLUderived;
			}

			weightGradients[l - 1] = layers[l - 1].getNeurons().transpose() * deltas[l - 1];
		}

		if (update) {
			int rows;

			for (int i = 0;i < L - 1;i++) {
				rows = weights[i].rows();
				cols = weights[i].cols();

				for (int r = 0;r < rows;r++) {
					for (int c = 0;c < cols;c++) {
						weights[i][r][c] = weights[i][r][c] - LEARNING_RATE * weightGradients[i][r][c];
					}
				}

				for (int c = 0;c < cols;c++) {
					biases[i][0][c] = biases[i][0][c] - LEARNING_RATE * deltas[i][0][c];
				}
			}

			return {};              //dont need to return the weightGradients and biasGradients if we already updated the weights and biases
		}

		return { weightGradients,deltas };  //deltas is the same as the gradients of the biases
	}


	void updateParameters(std::vector<Matrix<double>>& newWeights, std::vector<Matrix<double>>& newBiases, double avg = 1) {
		int rows;                                                                           //if we use a batch we will have
		int cols;                                                                           //to avg the total of the newWeights and newBiases

		int L = layers.size();

		for (int i = 0;i < L - 1;i++) {
			rows = weights[i].rows();
			cols = weights[i].cols();

			for (int r = 0;r < rows;r++) {
				for (int c = 0;c < cols;c++) {
					weights[i][r][c] = weights[i][r][c] - LEARNING_RATE * (newWeights[i][r][c] / avg);
				}
			}

			for (int c = 0;c < cols;c++) {
				biases[i][0][c] = biases[i][0][c] - LEARNING_RATE * (newBiases[i][0][c] / avg);
			}
		}
	}


	//training the neural network on a dataset
	void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs) { 
		if (inputs.size() != outputs.size()) {
			throw invalid_size();
		}

		std::vector<int> indexes(inputs.size());

		for (int i = 0;i < inputs.size();i++) {
			indexes[i] = i;
		}

		std::random_device rd;
		std::mt19937 mt(rd());


		int size = inputs.size() / BATCH_SIZE;
		int L = layers.size();

		std::vector<Matrix<double>> totalWeights(L - 1);
		std::vector<Matrix<double>> totalBiases(L - 1);

		for (int i = 0;i < L - 1;i++) {
			totalWeights[i] = Matrix<double>(weights[i].rows(), weights[i].cols());
			totalBiases[i] = Matrix<double>(1, biases[i].cols());
		}

		std::vector<std::vector<Matrix<double>>> weightsAndBiases(2, std::vector<Matrix<double>>(L - 1));

		clock_t start, end, curStart;
		start = clock();

		int idx;

		for (int i = 0;i < EPOCHS;i++) {
			curStart = clock();

			std::cout << "========\n";
			std::cout << "EPOCH #" << i + 1 << '\n';
			std::cout << "========\n";

			std::shuffle(indexes.begin(), indexes.end(), mt);   //shuffling the inputs at the start of every epoch, so its not the same
																//batch everytime

			for (int j = 0;j < size;j++) {
				for (int k = 0;k < BATCH_SIZE;k++) {
					idx = indexes[j * BATCH_SIZE + k];

					this->setInput(inputs[idx]);
					this->setTarget(outputs[idx]);
					this->feedForward();

					weightsAndBiases = this->backPropagation();

					for (int l = 0;l < L - 1;l++) {
						totalWeights[l] += weightsAndBiases[0][l];
						totalBiases[l] += weightsAndBiases[1][l];
					}
				}

				//end of batch , time to update the weights and biases with the average of the totalWeights and totalBiases

				this->updateParameters(totalWeights, totalBiases, BATCH_SIZE);

				for (int l = 0;l < L - 1;l++) {
					totalWeights[l].zero();
					totalBiases[l].zero();
				}
			}

			end = clock();

			std::cout << "Current Epoch Time: " << (double)(end - curStart) / CLOCKS_PER_SEC << " s\n";
			std::cout << "Total Time: " << (double)(end - start) / CLOCKS_PER_SEC << " s\n";
		}
	}
};


#endif