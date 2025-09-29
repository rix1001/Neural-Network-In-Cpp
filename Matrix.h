#pragma once
#include <vector>
#include <cmath>
#include <iostream>

template<class T>
class Matrix {
	int m_rows, m_cols;
	T** m_elems;
public:
	class incorrect_2D_vector {};
	class different_dimensions {};
	class not_compatible {};
	class invalid_determinant {};
	class invalid_inverse {};

	Matrix() {
		m_rows = 0;
		m_cols = 0;

		m_elems = nullptr;
	}


	Matrix(int n, int m) {
		m_rows = n;
		m_cols = m;

		m_elems = new T * [n];
		for (int i = 0;i < n;i++) {
			m_elems[i] = new T[m]{};
		}

		if (m_rows == m_cols) {
			for (int i = 0;i < m_rows;i++) {
				m_elems[i][i] = 1;
			}
		}
	}


	Matrix(int n, int m, T** mat) {
		m_rows = n;
		m_cols = m;

		m_elems = new T * [n];

		for (int i = 0;i < n;i++) {
			m_elems[i] = new T[m];

			for (int j = 0;j < m;j++) {
				m_elems[i][j] = mat[i][j];
			}
		}
	}


	Matrix(const Matrix& mat) {
		m_rows = mat.m_rows;
		m_cols = mat.m_cols;

		m_elems = new T * [m_rows];

		for (int i = 0;i < m_rows;i++) {
			m_elems[i] = new T[m_cols];

			for (int j = 0;j < m_cols;j++) {
				m_elems[i][j] = mat.m_elems[i][j];
			}
		}
	}

	Matrix(Matrix&& mat) noexcept {
		m_rows = mat.m_rows;
		m_cols = mat.m_cols;
		m_elems = mat.m_elems;

		mat.m_rows = 0;
		mat.m_cols = 0;
		mat.m_elems = nullptr;
	}


	Matrix(const std::vector<std::vector<T>>& v) {
		m_rows = v.size();

		for (int i = 0;i < m_rows;i++) {
			if (v[i].size() != v[0].size()) {
				throw incorrect_2D_vector();
			}
		}

		m_cols = v[0].size();

		m_elems = new T * [m_rows];

		for (int i = 0;i < m_rows;i++) {
			m_elems[i] = new T[m_cols];

			for (int j = 0;j < m_cols;j++) {
				m_elems[i][j] = v[i][j];
			}
		}
	}


	Matrix& operator=(const Matrix& mat) {
		m_rows = mat.m_rows;
		m_cols = mat.m_cols;

		m_elems = new T * [m_rows];

		for (int i = 0;i < m_rows;i++) {
			m_elems[i] = new T[m_cols];

			for (int j = 0;j < m_cols;j++) {
				m_elems[i][j] = mat.m_elems[i][j];
			}
		}

		return *this;
	}

	Matrix& operator=(Matrix&& mat) noexcept {
		m_rows = mat.m_rows;
		m_cols = mat.m_cols;
		m_elems = mat.m_elems;

		mat.m_rows = 0;
		mat.m_cols = 0;
		mat.m_elems = nullptr;

		return *this;
	}

	~Matrix() {
		for (int i = 0;i < m_rows;i++) {
			delete[]m_elems[i];
		}
		delete[]m_elems;
	}

	int rows() const {
		return m_rows;
	}

	int cols() const {
		return m_cols;
	}

	T*& operator[] (int i) {
		return m_elems[i];
	}

	Matrix operator+(const Matrix& mat) {
		if ((m_rows != mat.m_rows) || (m_cols != mat.m_cols)) {
			throw different_dimensions();
		}

		Matrix tmp(m_rows, m_cols);

		for (int i = 0;i < m_rows;i++) {
			for (int j = 0;j < m_cols;j++) {
				tmp[i][j] = m_elems[i][j] + mat.m_elems[i][j];
			}
		}

		return tmp;
	}

	Matrix& operator+=(const Matrix& mat) {
		if ((m_rows != mat.m_rows) || (m_cols != mat.m_cols)) {
			throw different_dimensions();
		}

		for (int i = 0;i < m_rows;i++) {
			for (int j = 0;j < m_cols;j++) {
				m_elems[i][j] += mat.m_elems[i][j];
			}
		}

		return *this;
	}

	Matrix operator-(const Matrix& mat) {
		if ((m_rows != mat.m_rows) || (m_cols != mat.m_cols)) {
			throw different_dimensions();
		}

		Matrix tmp(m_rows, m_cols);

		for (int i = 0;i < m_rows;i++) {
			for (int j = 0;j < m_cols;j++) {
				tmp[i][j] = m_elems[i][j] - mat.m_elems[i][j];
			}
		}

		return tmp;
	}

	Matrix& operator-=(const Matrix& mat) {
		if ((m_rows != mat.m_rows) || (m_cols != mat.m_cols)) {
			throw different_dimensions();
		}

		for (int i = 0;i < m_rows;i++) {
			for (int j = 0;j < m_cols;j++) {
				m_elems[i][j] -= mat.m_elems[i][j];
			}
		}

		return *this;
	}

	Matrix operator*(const Matrix& mat) {
		if (m_cols != mat.m_rows) {
			throw not_compatible();
		}

		Matrix tmp(m_rows, mat.m_cols);

		auto sum = m_elems[0][0];
		sum = 0;

		for (int i = 0;i < m_rows;i++) {
			for (int j = 0;j < mat.m_cols;j++) {
				sum = 0;

				for (int k = 0;k < m_cols;k++) {
					sum += (m_elems[i][k] * mat.m_elems[k][j]);
				}

				tmp[i][j] = sum;
			}
		}

		return tmp;
	}

	Matrix& operator*=(const Matrix& mat) {
		if (m_cols != mat.m_rows) {
			throw not_compatible();
		}

		Matrix tmp(m_rows, mat.m_cols);

		auto sum = m_elems[0][0];
		sum = 0;

		for (int i = 0;i < m_rows;i++) {
			for (int j = 0;j < mat.m_cols;j++) {
				sum = 0;

				for (int k = 0;k < m_cols;k++) {
					sum += (m_elems[i][k] * mat.m_elems[k][j]);
				}

				tmp[i][j] = sum;
			}
		}

		*this = tmp;

		return *this;
	}

	Matrix transpose() {
		std::vector<std::vector<T>> tmp(m_cols, std::vector<T>(m_rows));

		for (int i = 0;i < m_rows;i++) {
			for (int j = 0;j < m_cols;j++) {
				tmp[j][i] = m_elems[i][j];
			}
		}

		return tmp;
	}

	double determinant() {          //Gauss elimination
		if (m_rows != m_cols) {
			throw invalid_determinant();
		}

		Matrix<double> tmp(m_rows, m_cols);
		for (int r = 0;r < m_rows;r++) {
			for (int c = 0;c < m_cols; c++) {
				tmp[r][c] = m_elems[r][c];
			}
		}

		double pivot;
		double val;

		int sign = 1;

		int idx;

		for (int d = 0;d < m_rows - 1;d++) {
			pivot = tmp[d][d];

			if (pivot == 0) {
				idx = -1;

				for (int i = d + 1;i < m_rows;i++) {
					if (tmp[i][d] != 0) {
						pivot = tmp[i][d];
						idx = i;
						break;
					}
				}

				if (idx == -1) {
					continue;
				}

				for (int j = 0;j < m_cols;j++) {
					std::swap(tmp[d][j], tmp[idx][j]);
				}

				sign *= -1;
			}


			for (int r = d + 1;r < m_rows;r++) {
				val = tmp[r][d] / pivot;

				for (int c = d;c < m_cols;c++) {
					tmp[r][c] -= (val * tmp[d][c]);
				}
			}
		}

		double det = 1;

		for (int d = 0;d < m_rows;d++) {
			det *= tmp[d][d];
		}

		det *= sign;

		if (std::fabs(det) < 1e-10) {    //avoiding inaccuracy caused by double
			return 0;
		}

		return det;
	}

	Matrix<double> inverse() {
		if (m_rows != m_cols) {
			throw invalid_inverse();
		}

		double det = determinant();

		if (det == 0) {
			throw invalid_inverse();
		}

		double detInv = 1.0 / det;

		Matrix<double> invMatrix(m_rows, m_cols);

		Matrix<T> copy(*this);
		copy = copy.transpose();

		Matrix tmp(m_rows - 1, m_cols - 1);

		int idx1, idx2;

		for (int i = 0;i < m_rows;i++) {
			for (int j = 0;j < m_cols;j++) {
				idx1 = 0, idx2 = 0;

				for (int r = 0;r < m_rows;r++) {
					if (r == i) {
						continue;
					}

					for (int c = 0;c < m_cols;c++) {
						if (c == j) {
							continue;
						}

						tmp.m_elems[idx1][idx2] = copy.m_elems[r][c];

						if (idx2 == m_cols - 2) {
							idx1++;
							idx2 = 0;
						}
						else {
							idx2++;
						}
					}
				}

				invMatrix[i][j] = tmp.determinant();


				if ((i + j) % 2 == 1) {
					invMatrix[i][j] *= -1;
				}

				invMatrix[i][j] *= detInv;

				if (invMatrix[i][j] == -0) {
					invMatrix[i][j] = 0;
				}
			}
		}


		return invMatrix;
	}

	Matrix operator^(unsigned int n) {
		if (m_rows != m_cols) {
			throw not_compatible();
		}

		Matrix tmp(*this);

		Matrix sol(m_rows, m_cols);

		while (n != 0) {
			if (n % 2 == 1) {
				sol *= tmp;
			}
			tmp *= tmp;
			n /= 2;
		}

		return sol;
	}

	Matrix& operator^=(unsigned int n) {
		if (m_rows != m_cols) {
			throw not_compatible();
		}

		Matrix tmp(*this);

		*this = Matrix(m_rows, m_cols);

		while (n != 0) {
			if (n % 2 == 1) {
				*this *= tmp;
			}

			tmp *= tmp;
			n /= 2;
		}


		return *this;
	}

	Matrix HadamardProduct(const Matrix& mat) {
		if (m_rows != mat.m_rows || m_cols != mat.m_cols) {
			throw different_dimensions();
		}

		Matrix<T> tmp(m_rows, m_cols);

		for (int i = 0;i < m_rows;i++) {
			for (int j = 0;j < m_cols;j++) {
				tmp[i][j] = m_elems[i][j] * mat.m_elems[i][j];
			}
		}

		return tmp;
	}

	void zero() {
		for (int i = 0;i < m_rows;i++) {
			for (int j = 0;j < m_cols;j++) {
				m_elems[i][j] = 0;
			}
		}
	}

	std::ostream& print(std::ostream& os) const {
		for (int i = 0;i < m_rows;i++) {
			for (int j = 0;j < m_cols;j++) {
				os << m_elems[i][j] << ' ';
			}
			os << '\n';
		}

		return os;
	}
};

template<class T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& m) {
	return m.print(os);
}
