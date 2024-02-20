#pragma once

#include <iostream>
#include <cmath>

#include <cmath>

#include <cmath>
#include <initializer_list>

template <int size>
class Vector {
public:
	float elements[size]; // research using double from now on

	// Default constructor initializes all elements to 0.0f
	Vector() {
		for (int i = 0; i < size; ++i) {
			elements[i] = 0.0f;
		}
	}

	// Constructor with initializer list
	Vector(const std::initializer_list<float>& list) {
		int i = 0;
		for (const auto& val : list) {
			elements[i++] = val;
		}
	}

	template <typename... Args>
	Vector(Args... args) {
		static_assert(sizeof...(Args) == size, "Incorrect number of arguments for Vector constructor");
		InitializeElements(0, args...);
	}

	// Private helper function to initialize elements
	template <typename First, typename... Rest>
	void InitializeElements(int index, First first, Rest... rest) {
		elements[index] = static_cast<float>(first);
		InitializeElements(index + 1, rest...);
	}

	// Base case for the recursion
	static void InitializeElements(int index) {}

	// Access elements with out-of-range checks
	float& operator[](int index) {
		if (index >= 0 && index < size) {
			return elements[index];
		}
		throw std::out_of_range("Index out of range");
	}

	// Const version for read-only access
	const float& operator[](int index) const {
		if (index >= 0 && index < size) {
			return elements[index];
		}
		throw std::out_of_range("Index out of range");
	}

	Vector<size> operator+(const Vector<size>& other) const {
		Vector<size> result;
		for (int i = 0; i < size; ++i) {
			result.elements[i] = elements[i] + other.elements[i];
		}
		return result;
	}

	Vector<size> operator-(const Vector<size>& other) const {
		Vector<size> result;
		for (int i = 0; i < size; ++i) {
			result.elements[i] = elements[i] - other.elements[i];
		}
		return result;
	}

	Vector<size> operator*(float scalar) const {
		Vector<size> result;
		for (int i = 0; i < size; ++i) {
			result.elements[i] = elements[i] * scalar;
		}
		return result;
	}

	// Dot Product of the vector
	float Dot(const Vector<size>& other) const {
		float dotProduct = 0.0f;
		for (int i = 0; i < size; ++i) {
			dotProduct += elements[i] * other.elements[i];
		}
		return dotProduct;
	}

	// Normalize the vector
	Vector<size> Normalize() const {
		float mag = Magnitude();
		Vector<size> result;
		for (int i = 0; i < size; ++i) {
			result[i] = elements[i] / mag;
		}
		return result;
	}

	// Magnitude of the vector
	float Magnitude() const {
		float sum = 0.0f;
		for (int i = 0; i < size; ++i) {
			sum += elements[i] * elements[i];
		}
		return std::sqrt(sum);
	}

	// Inverse the vector
	Vector<size> Inverse() const
	{
		Vector<size> result;
		for (int i = 0; i < size; i++)
		{
			result.elements[i] = -elements[i];
		}
		return result;
	}
};


class AABB {
public:
	Vector<3> minExtent;
	Vector<3> maxExtent;

	// Constructor taking minimum and maximum extents
	AABB(const Vector<3>& Min, const Vector<3>& Max) : minExtent(Min), maxExtent(Max) {}

	// Getter functions for the bounding box dimensions
	float Top() const { return maxExtent[1]; }
	float Bottom() const { return minExtent[1]; }
	float Left() const { return minExtent[0]; }
	float Right() const { return maxExtent[0]; }
	float Front() const { return maxExtent[2]; }
	float Back() const { return minExtent[2]; }
};

class Matrix4by4 {
public:
	Vector<4> columns[4];

	Matrix4by4() {
		for (Vector<4>& column : columns)
		{
			column = Vector<4>(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}

	// Constructors taking column vectors
	Matrix4by4(const Vector<4>& column1, const Vector<4>& column2, const Vector<4>& column3, const Vector<4>& column4) {
		columns[0] = column1;
		columns[1] = column2;
		columns[2] = column3;
		columns[3] = column4;
	}

	// Copy constructor
	Matrix4by4(const Matrix4by4& other) {
		for (int i = 0; i < 4; ++i) {
			columns[i] = other.columns[i];
		}
	}

	// Set a specific column with another matrix
	Matrix4by4& SetColumn(int columnNumber, const Matrix4by4& matrix) {
		columns[columnNumber] = matrix.columns[columnNumber];
		return *this;
	}

	// Set a specific column with a Vector<4>
	Matrix4by4& SetColumn(int columnNumber, const Vector<4> vector) {
		columns[columnNumber] = vector;
		return *this;
	}

	// Get a specific column as a new matrix
	Matrix4by4 GetColumn(int columnNumber) const {
		return Matrix4by4(
			Vector<4>(columns[0][columnNumber], 0.0f, 0.0f, 0.0f),
			Vector<4>(0.0f, columns[1][columnNumber], 0.0f, 0.0f),
			Vector<4>(0.0f, 0.0f, columns[2][columnNumber], 0.0f),
			Vector<4>(0.0f, 0.0f, 0.0f, columns[3][columnNumber])
		);
	}

	// Invert the translation and rotation part of the matrix
	Matrix4by4 InvertTR() const {
		Matrix4by4 rv = Identity();

		// Copy the transpose of the rotation part
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				rv.columns[i][j] = columns[j][i];
			}
		}

		// Set negation of the translation part
		rv.columns[0][3] = -columns[0][3];
		rv.columns[1][3] = -columns[1][3];
		rv.columns[2][3] = -columns[2][3];

		return rv;
	}

	// Identity matrix
	static Matrix4by4 Identity() {
		return Matrix4by4(
			Vector<4>(1.0f, 0.0f, 0.0f, 0.0f),
			Vector<4>(0.0f, 1.0f, 0.0f, 0.0f),
			Vector<4>(0.0f, 0.0f, 1.0f, 0.0f),
			Vector<4>(0.0f, 0.0f, 0.0f, 1.0f)
		);
	}

	// Matrix-vector multiplication
	Vector<4> operator *(const Vector<4>& vector) const {
		Vector<4> result;
		for (int i = 0; i < 4; ++i) {
			result[i] = columns[i].Dot(vector);
		}
		return result;
	}

	// Matrix-matrix multiplication
	Matrix4by4 operator *(const Matrix4by4& matrix) const {
		Matrix4by4 result;

		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				result.columns[i][j] = columns[0][i] * matrix.columns[0][j] +
					columns[1][i] * matrix.columns[1][j] +
					columns[2][i] * matrix.columns[2][j] +
					columns[3][i] * matrix.columns[3][j];
			}
		}

		return result;
	}

	Matrix4by4 operator -(const Matrix4by4& matrix) {
		Matrix4by4 result;

		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				result.columns[i][j] = columns[i][j] - matrix.columns[i][j];

		return result;
	}

	// Invert the translation part
	Matrix4by4 TranslationInverse() const {
		Matrix4by4 rv = Identity();

		// Set negation of the translation part
		rv.columns[0][3] = -columns[0][3];
		rv.columns[1][3] = -columns[1][3];
		rv.columns[2][3] = -columns[2][3];

		return rv;
	}

	// Invert the scale part
	Matrix4by4 ScaleInverse() const {
		Matrix4by4 rv = Identity();

		// Set reciprocal of the diagonal scale elements
		rv.columns[0][0] = 1.0f / columns[0][0];
		rv.columns[1][1] = 1.0f / columns[1][1];
		rv.columns[2][2] = 1.0f / columns[2][2];

		return rv;
	}
};

class Quat {
private:
	float x;
	float y;
	float z;
	float w;

public:
	Quat() : x(0.0f), y(0.0f), z(0.0f), w(1.0f) {}

	Quat(const float newX, const float newY, const float newZ, const float newW)
		: x(newX), y(newY), z(newZ), w(newW) {}

	Quat(const float angle, Vector<3> axis) {
		const float halfAngle = angle / 2;
		w = std::cos(halfAngle);
		x = axis[0] * std::sin(halfAngle);
		y = axis[1] * std::sin(halfAngle);
		z = axis[2] * std::sin(halfAngle);
	}

	void Set(const float newX, const float newY, const float newZ, const float newW) {
		x = newX;
		y = newY;
		z = newZ;
		w = newW;
	}

	void SetAxis(Vector<3> axis) {
		x = axis[0];
		y = axis[1];
		z = axis[2];
	}

	Vector<3> GetAxis() const {
		return Vector<3>(x, y, z);
	}

	Vector<4> GetAxisAngle() const {
		Vector<4> result;

		const float halfAngle = std::acos(w);
		result[3] = halfAngle * 2;

		const float sinHalfAngle = std::sin(halfAngle);
		if (sinHalfAngle != 0.0f) {
			result[0] = x / sinHalfAngle;
			result[1] = y / sinHalfAngle;
			result[2] = z / sinHalfAngle;
		}
		else {
			// Handle division by zero gracefully
			result[0] = result[1] = result[2] = 0.0f;
		}

		return result;
	}

	Quat Inverse() const {
		Quat result;
		result.w = w;
		result.SetAxis(GetAxis().Inverse());
		return result;
	}

	void Scale(const float s) {
		x *= s;
		y *= s;
		z *= s;
		w *= s;
	}

	static Quat Identity() {
		return Quat(0.0f, 0.0f, 0.0f, 1.0f);
	}

	static Quat Normalize(const Quat& quat) {
		Quat result;
		const float num2 = ((quat.x * quat.x) + (quat.y * quat.y)) + (quat.z * quat.z) + (quat.w * quat.w);
		const float num = 1.0f / std::sqrt(num2);

		result.x = quat.x * num;
		result.y = quat.y * num;
		result.z = quat.z * num;
		result.w = quat.w * num;

		return result;
	}

	// Addition operator
	Quat operator+(const Quat& other) const {
		return Quat(x + other.x, y + other.y, z + other.z, w + other.w);
	}

	// Multiplication operator (quaternion * quaternion)
	Quat operator*(const Quat& other) const {
		return Quat(
			w * other.x + x * other.w + y * other.z - z * other.y,
			w * other.y - x * other.z + y * other.w + z * other.x,
			w * other.z + x * other.y - y * other.x + z * other.w,
			w * other.w - x * other.x - y * other.y - z * other.z
		);
	}

	// Scaling operator (quaternion * scalar)
	Quat operator*(const float scalar) const {
		return Quat(x * scalar, y * scalar, z * scalar, w * scalar);
	}

	// Normalize the quaternion
	Quat& Normalize() {
		const float num2 = ((x * x) + (y * y)) + (z * z) + (w * w);
		const float num = 1.0f / std::sqrt(num2);

		x *= num;
		y *= num;
		z *= num;
		w *= num;

		return *this;
	}

	// Equality operator
	bool operator==(const Quat& other) const {
		return x == other.x && y == other.y && z == other.z && w == other.w;
	}

	// Inequality operator
	bool operator!=(const Quat& other) const {
		return !(*this == other);
	}
};
