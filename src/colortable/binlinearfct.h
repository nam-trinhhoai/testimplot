#ifndef BINLINEARFCT_H_
#define BINLINEARFCT_H_

#include "abstractfct.h"
#include <QPainter>
#include "lutrenderutil.h"

class BinLinearFct: public AbstractFct {

public:
	BinLinearFct(int p1, int p2,int colorTableSize, bool inverted);
	BinLinearFct();
	BinLinearFct(int colorTableSize);
	virtual ~BinLinearFct();

	FUNCTION_TYPE getType() const override{return FUNCTION_TYPE::BINLINEAR;}
	void reset() override;
	virtual AbstractFct * clone();
	int get(int x) override;
	void paint(QPainter * e,const QSize &size, int histogramSize) override;

};

#endif
