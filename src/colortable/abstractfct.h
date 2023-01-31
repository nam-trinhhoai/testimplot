/*
 * AbstractFct.h
 *
 *  Created on: 16 mai 2018
 *      Author: j0334308
 */

#ifndef ABSTRACTFCT_H_
#define ABSTRACTFCT_H_

#include <QSize>

#include "colortable.h"


class QPainter;
class AbstractFct {
public:
	enum FUNCTION_TYPE{
		LINEAR,
		BINARY,
		BINLINEAR,
		LOG,
		TRIANGLE1,
		TRIANGLE2,
		UNDEF
	};

	AbstractFct(int p1, int p2,int colorTableSize, bool inverted);
	AbstractFct();
	AbstractFct(int colorTableSize);
	virtual ~AbstractFct();

	virtual void paint(QPainter *,const QSize &size, int histogramSize){};

	virtual void reset(){};

	virtual AbstractFct * clone(){return nullptr;};

	virtual int get(int x){return x;};

	virtual FUNCTION_TYPE getType() const{return FUNCTION_TYPE::UNDEF;}


	int getParam1() {
		return p1;
	}

	virtual void setParam1(int p) {
		p1 = p;
	}

	int getParam2() {
		return p2;
	}

	virtual void setParam2(int p) {
		p2 = p;
	}

	bool isInverted() {
		return inverted;
	}

	void setInverted(bool b) {
		inverted = b;
	}

protected:
	 int p1;
	 int p2;

	 int colorTableSize;

	 bool inverted = false;
};

#endif
