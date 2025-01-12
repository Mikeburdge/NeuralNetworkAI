#pragma once

class SingletonBase {
public:
	static SingletonBase& GetInstance() {
		static SingletonBase instance;
		return instance;
	}

	// Other class methods or variables can be defined here

protected:
	SingletonBase() {} // Private constructor to prevent direct instantiation
	SingletonBase(const SingletonBase&) = delete; // Delete copy constructor
	void operator=(const SingletonBase&) = delete; // Delete assignment operator
};
