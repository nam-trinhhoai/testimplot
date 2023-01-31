#include "statusbar.h"
#include <iomanip>
#include <sstream>
#include <QLabel>
#include <QLineEdit>
#include <QGridLayout>
#include <QSizeGrip>


StatusBar::StatusBar(QWidget *parent):QWidget(parent) {
	m_x=new QLineEdit(this);
	m_x->setReadOnly(true);
	m_y=new QLineEdit( this);
	m_y->setReadOnly(true);
	m_i=new QLineEdit( this);
	m_i->setReadOnly(true);
	m_j=new QLineEdit(this);
	m_j->setReadOnly(true);
	m_depth=new QLineEdit(this);
	m_depth->setReadOnly(true);
	m_value=new QLineEdit(this);
	m_value->setReadOnly(true);

	QGridLayout *layout=new QGridLayout(this);
//	layout->setMargin(0);
	layout->setContentsMargins(0,0,0,0);
	int index=0;
	m_labelWorldX=new QLabel("E:",this);
	layout->addWidget(m_labelWorldX,0,index++);
	layout->addWidget(m_x,0,index);
	layout->setColumnStretch(index,2);
	index++;

	m_labelWorldY=new QLabel("N:",this);
	layout->addWidget(m_labelWorldY,0,index++);
	layout->addWidget(m_y,0,index);
	layout->setColumnStretch(index,2);
	index++;

	layout->addWidget(new QLabel("I:",this),0,index++);
	layout->addWidget(m_i,0,index);
	layout->setColumnStretch(index,2);
	index++;

	layout->addWidget(new QLabel("J:",this),0,index++);
	layout->addWidget(m_j,0,index);
	layout->setColumnStretch(index,2);
	index++;

	layout->addWidget(new QLabel("Depth:",this),0,index++);
	layout->addWidget(m_depth,0,index);
	layout->setColumnStretch(index,2);
	index++;

	layout->addWidget(new QLabel("Value:",this),0,index++);
	layout->addWidget(m_value,0,index);
	layout->setColumnStretch(index,2);
	index++;


}
void StatusBar::setWorldCoordinateLabels(const QString &labelX, const QString &labelY)
{
	m_labelWorldX->setText(labelX);
	m_labelWorldY->setText(labelY);
}

void StatusBar::x(double x)
{
	std::stringstream ss;
	ss << std::fixed << std::setprecision(2) <<x;
	m_x->setText(ss.str().c_str());

}
void StatusBar::y(double x)
{
	std::stringstream ss;
	ss << std::fixed << std::setprecision(2) <<x;
	m_y->setText(ss.str().c_str());
}

void StatusBar::depth(double x)
{
	std::stringstream ss;
	ss << std::fixed << std::setprecision(2) <<x;
	m_depth->setText(ss.str().c_str());
}


void StatusBar::i(int x)
{
	std::stringstream ss;
	ss << std::fixed << std::setprecision(0) <<x;
	m_i->setText(ss.str().c_str());
}
void StatusBar::j(int x)
{
	std::stringstream ss;
	ss << std::fixed << std::setprecision(0) <<x;
	m_j->setText(ss.str().c_str());
}

void StatusBar::value(const QString &value)
{
	m_value->setText(value);
}

void StatusBar::clearI()
{
	m_i->setText("");
}
void StatusBar::clearJ()
{
	m_j->setText("");
}
void StatusBar::clearDepth()
{
	m_depth->setText("");
}


void StatusBar::clearValue()
{
	m_value->setText("");
}

StatusBar::~StatusBar() {
	// TODO Auto-generated destructor stub
}

