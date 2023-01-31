/*
 * Matrix2D.h
 *
 *  Created on: 10 janv. 2018
 *      Author: j0483271
 */


// tips https://stackoverflow.com/questions/24702235/c-stdmap-holding-any-type-of-value

#include <vector>
#include "imageformats.h"

#ifndef MURATAPP_SRC_VIEW_CANVAS2D_MATRIX2D_H_
#define MURATAPP_SRC_VIEW_CANVAS2D_MATRIX2D_H_

class Matrix2DInterface {
public:
	virtual ~Matrix2DInterface() {};
	virtual const double getDouble(unsigned int x, unsigned int y) = 0;
	virtual const unsigned int width() = 0;
	virtual const unsigned int height() = 0;
	virtual void* data() = 0;
    virtual const ImageFormats::QSampleType getType() = 0;
};

template<typename T>
class Matrix2DLine : Matrix2DInterface {
public:
    Matrix2DLine(int w=1, int h=1);
	virtual ~Matrix2DLine();
	const T get(unsigned int x, unsigned int y);
	void set(const T val, unsigned int x, unsigned int y);
	T* getLine(unsigned int y);
	T* getData();
	const double getDouble(unsigned int x, unsigned int y);
	const unsigned int width();
	const unsigned int height();
    const ImageFormats::QSampleType getType();
	void* data();
    void transpose();


protected:
	std::vector<T> tab; // tab that will contain data
	int w;  // width
	int h;  // height
};

#include "Matrix2D.hpp"

#endif /* MURATAPP_SRC_VIEW_CANVAS2D_MATRIX2D_H_ */
