/*
 * lookuptable.cpp
 *
 *  Created on: 16 mai 2018
 *      Author: j0334308
 */

#include "lookuptable.h"
#include "colortableregistry.h"
#include "linearfct.h"
#include "binlinearfct.h"
#include "binfct.h"
#include "logfct.h"
#include "triangle1fct.h"
#include "triangle2fct.h"
#include <QDebug>
#include <iostream>

LookupTable::LookupTable()
{
	m_table=ColorTableRegistry::DEFAULT();
	m_function=new LinearFct(m_table.size());
}
LookupTable::LookupTable(const ColorTable &table) {
	m_table=table;
	m_function=new LinearFct(table.size());
}

LookupTable::~LookupTable() {
	delete m_function;
}

const ColorTable& LookupTable::getColorTable() const
{
	return m_table;
}

void LookupTable::setFunctionType(AbstractFct::FUNCTION_TYPE type)
{
	delete m_function;
	switch(type)
	{
	case AbstractFct::FUNCTION_TYPE::BINARY:
		m_function=new BinFct(size());
		break;
	case AbstractFct::FUNCTION_TYPE::BINLINEAR:
		m_function=new BinFct(size());
		break;
	case AbstractFct::FUNCTION_TYPE::LOG:
		m_function=new LogFct(size());
		break;
	case AbstractFct::FUNCTION_TYPE::TRIANGLE1:
		m_function=new Triangle1Fct(size());
		break;
	case AbstractFct::FUNCTION_TYPE::TRIANGLE2:
		m_function=new Triangle2Fct(size());
		break;
	default:
		m_function=new LinearFct(size());
		break;
	}

	m_function->reset();
}
AbstractFct::FUNCTION_TYPE LookupTable::getFunctionType()const
{
	return m_function->getType();
}

bool LookupTable::isFunctionInverted() const
{
	return m_function->isInverted();
}
void LookupTable::setFunctionInverted(bool val)
{
	m_function->setInverted(val);
}


LookupTable::LookupTable(const LookupTable & par) {
	this->m_table = par.m_table;
	this->m_function=par.m_function->clone();
}

LookupTable& LookupTable::operator=(const LookupTable& par) {
	if (this != &par) {
		delete this->m_function;
		this->m_table = par.m_table;
		this->m_function=par.m_function->clone();
	}
	return *this;
}

void LookupTable::paintFunction(QPainter * p,const QSize &size, int histogramSize) const{
		m_function->paint(p,size,histogramSize);
}

int LookupTable::getFunctionParam1() const
{
	return m_function->getParam1();
}
int LookupTable::getFunctionParam2() const
{
	return m_function->getParam2();
}

void LookupTable::setFunctionParam1(int val)
{
	m_function->setParam1(val);
}
void LookupTable::setFunctionParam2(int val)
{
	m_function->setParam2(val);
}

std::array<int, 4>  LookupTable::getColors(int i) const
{
	const std::array<int, 4> &color = m_table.getColors(clipIndex(m_function->get(i)));

	std::array<int, 4> returnColor;
	returnColor[0]=color[0];
	returnColor[1]=color[1];
	returnColor[2]=color[2];
	returnColor[3] = getAlpha(i);
	return returnColor;
}

int LookupTable::getAlpha(int i) const
{
	return m_table.getColors(clipIndex(i/*m_function->get(i))*/))[3];
}

int LookupTable::size()const
{
	return m_table.size();
}

std::string LookupTable::getName() const
{
	return m_table.getName();
}

 int LookupTable::clipIndex(int val) const
 {
	 return std::min(std::max(0,val),m_table.size() - 1);
 }

void  LookupTable::setAlpha(int from ,int to, int alpha)
{
	//Interpolate alpha in etween
	int fromIndexExtact=clipIndex(from);
	int toIndexExact=clipIndex(to);

	for(int i=fromIndexExtact;i<=toIndexExact;i++)
	{
		m_table.setAlpha(i, alpha);
	}
}

void LookupTable::setInterpolatedAlpha(int from,int to,bool sign ,int alpha)
{
	int realFrom=from;
	int realTo=to;

	if(sign)
	{
		realTo+=1;
	}else
	{
		realFrom-=1;
	}

	int fromIndexExtact=clipIndex(realFrom);
	int toIndexExact=clipIndex(realTo);
	for(int i=fromIndexExtact;i<=toIndexExact;i++)
	{
		m_table.setAlpha(i, alpha);
	}
}


void LookupTable::razTransp()
{
	for(int i=0;i<m_table.size();i++)
	{
		m_table.setAlpha(i, 255);
	}

}

void LookupTable::razFunction()
{
	m_function->reset();
}


