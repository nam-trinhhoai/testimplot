#ifndef BINFCT_H_
#define BINFCT_H_

#include "abstractfct.h"
#include <QPainter>
#include "lutrenderutil.h"

class BinFct: public AbstractFct {

public:
	BinFct(int p1, int p2,int colorTableSize, bool inverted);
	BinFct();
	BinFct(int colorTableSize);
	virtual ~BinFct();


	FUNCTION_TYPE getType() const override{return FUNCTION_TYPE::BINARY;}

	void reset() override;
	virtual AbstractFct * clone();
	int get(int x) override;
	void paint(QPainter * e,const QSize &size, int histogramSize) override;

};

#endif
