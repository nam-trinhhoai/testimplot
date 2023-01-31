#ifndef _IS_SAME_TYPE_H_
#define _IS_SAME_TYPE_H_

template<typename T, typename U>
struct isSameType {
	static const bool value = false;
};

template<typename T>
struct isSameType<T, T> { //specialization
	static const bool value = true;
};

#endif
