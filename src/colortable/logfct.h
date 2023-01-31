#ifndef LOGFCT_H_
#define LOGFCT_H_

#include "abstractfct.h"
#include <QPainter>
#include "lutrenderutil.h"

class LogFct: public AbstractFct {

public:
	LogFct(int p1, int p2,int colorTableSize, bool inverted);
	LogFct();
	LogFct(int colorTableSize);
	virtual ~LogFct();

	FUNCTION_TYPE getType() const override{return FUNCTION_TYPE::LOG;}
	void reset() override;
	virtual AbstractFct * clone();
	int get(int x) override;
	void paint(QPainter * e,const QSize &size, int histogramSize) override;
private:
	std::vector<int> discreteSegment(int x1, int y1, int x2, int y2);
};

#endif
