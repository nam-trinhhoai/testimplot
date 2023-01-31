/*
 * lookuptable.h
 *
 *  Created on: 16 mai 2018
 *      Author: j0334308
 */

#ifndef LOOKUPTABLE_H_
#define LOOKUPTABLE_H_

#include "colortable.h"
#include "abstractfct.h"
#include <QSize>

class QPainter;

class LookupTable {
    Q_GADGET
public:
	LookupTable();
	LookupTable(const  ColorTable &table);
	LookupTable(const  LookupTable & );
	LookupTable& operator=(const LookupTable&);

	virtual ~LookupTable();

	int size()const;
	std::string getName() const;
	const ColorTable& getColorTable() const;

	std::array<int, 4>  getColors(int i) const;
	int  getAlpha(int i) const;
	void setAlpha(int from,int to, int alpha);
	void setInterpolatedAlpha(int from,int to,bool sign ,int alpha);
	void paintFunction(QPainter * p,const QSize &size, int histogramSize) const;

	int getFunctionParam1() const;
	int getFunctionParam2() const;

	void setFunctionParam1(int val);
	void setFunctionParam2(int val);

	void setFunctionType(AbstractFct::FUNCTION_TYPE type);
	AbstractFct::FUNCTION_TYPE getFunctionType() const;

	bool isFunctionInverted() const;
	void setFunctionInverted(bool);

    void razTransp();
   	void razFunction();


private:
	int clipIndex(int val) const;
private:
	ColorTable m_table;
	AbstractFct *m_function;


};

#endif /* QTLARGEIMAGEVIEWER_SRC_COLORTABLE_LOOKUPTABLE_H_ */
