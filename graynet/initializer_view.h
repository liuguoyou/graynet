#pragma once

#include <initializer_list>

template<typename T>
class initializer_view {
public:
	initializer_view() :
		first_(nullptr), last_(nullptr) {}
	initializer_view(const initializer_view &) = default;
	initializer_view(std::initializer_list<T> list) :
		first_(list.begin()), last_(list.end()) {}

	template<typename Container>
	initializer_view(const Container &container) :
		first_(container.data()), last_(container.data() + container.size()) {}

	size_t size() const { return last_ - first_; }
	const T *begin() const { return first_; }
	const T *end() const { return last_; }

private:
	const T *first_;
	const T *last_;
};
