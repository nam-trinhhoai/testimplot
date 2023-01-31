#ifndef TRIANGLE2FCT_H_
#define TRIANGLE2FCT_H_

#include "abstractfct.h"
#include <QPainter>
#include "lutrenderutil.h"

class Triangle2Fct: public AbstractFct {

public:
	Triangle2Fct(int p1, int p2,int colorTableSize, bool inverted);
	Triangle2Fct();
	Triangle2Fct(int colorTableSize);
	virtual ~Triangle2Fct();

	FUNCTION_TYPE getType() const override{return FUNCTION_TYPE::TRIANGLE2;}
	void reset() override;
	virtual AbstractFct * clone();
	int get(int x) override;
	void paint(QPainter * e,const QSize &size, int histogramSize) override;

};

#endif
