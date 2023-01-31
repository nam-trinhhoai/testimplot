/*
 * AbstractFct.cpp
 *
 *  Created on: 16 mai 2018
 *      Author: j0334308
 */

#include "abstractfct.h"


AbstractFct::AbstractFct(int p1, int p2,int colorTableSize, bool inverted)
{
	this->p1=p1;
	this->p2=p2;
	this->colorTableSize=colorTableSize;
	this->inverted=inverted;
}
AbstractFct::AbstractFct() {
	p1=0;
	p2=0;
	colorTableSize=0;
}

AbstractFct::AbstractFct(int colorTableSize) {
	this->p1=0;
	this->p2=colorTableSize-1;
	this->colorTableSize=colorTableSize;
}

AbstractFct::~AbstractFct() {
	// TODO Auto-generated destructor stub
}

