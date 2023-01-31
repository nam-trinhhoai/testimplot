#ifndef LINEARFCT_H_
#define LINEARFCT_H_

#include "abstractfct.h"
#include <QPainter>
#include "lutrenderutil.h"

class LinearFct: public AbstractFct {

public:
	LinearFct(int p1, int p2,int colorTableSize, bool inverted);
	LinearFct();
	LinearFct(int colorTableSize);
	virtual ~LinearFct();

	FUNCTION_TYPE getType() const override{return FUNCTION_TYPE::LINEAR;}

	void reset() override;
	virtual AbstractFct * clone();
	int get(int x) override;
	void paint(QPainter * e,const QSize &size, int histogramSize) override;

};

#endif
