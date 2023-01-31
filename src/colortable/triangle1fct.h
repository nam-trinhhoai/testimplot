#ifndef TRIANGLE1FCT_H_
#define TRIANGLE1FCT_H_

#include "abstractfct.h"
#include <QPainter>
#include "lutrenderutil.h"

class Triangle1Fct: public AbstractFct {

public:
	Triangle1Fct(int p1, int p2,int colorTableSize, bool inverted);
	Triangle1Fct();
	Triangle1Fct(int colorTableSize);
	virtual ~Triangle1Fct();

	FUNCTION_TYPE getType() const override{return FUNCTION_TYPE::TRIANGLE1;}

	void reset() override;
	virtual AbstractFct * clone();
	int get(int x) override;
	void paint(QPainter * e,const QSize &size, int histogramSize) override;

};

#endif
