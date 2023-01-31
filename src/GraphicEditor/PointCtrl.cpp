#include "PointCtrl.h"


PointCtrl::PointCtrl()
{
	m_position= QPointF(0,0);
	m_ctrl1= QPointF(0,0);
	m_ctrl2= QPointF(0,0);
	m_nbPoints = 0;
}

PointCtrl::PointCtrl(const PointCtrl& other)
{
	this->m_position = other.m_position;
	this->m_ctrl1 = other.m_ctrl1;
	this->m_ctrl2 = other.m_ctrl2;
	this->m_nbPoints = other.m_nbPoints;
	this->m_grab1 = other.m_grab1;
	this->m_grab2 = other.m_grab2;
}

PointCtrl::PointCtrl(QPointF position, QPointF ctrl1)//, QGraphicsItem* item)
{
	m_position= position;
	m_ctrl1= ctrl1;
	m_ctrl2= ctrl1;
	m_nbPoints = 1;

}

PointCtrl::PointCtrl(QPointF position, QPointF ctrl1, QPointF ctrl2)// ,QGraphicsItem* item)
{
	m_position= position;
	m_ctrl1= ctrl1;
	m_ctrl2= ctrl2;
	m_nbPoints = 2;

}


PointCtrl::~PointCtrl()
{

}

PointCtrl& PointCtrl::operator=(const PointCtrl& other)
{
	this->m_position = other.m_position;
	this->m_ctrl1 = other.m_ctrl1;
	this->m_ctrl2 = other.m_ctrl2;
	this->m_nbPoints = other.m_nbPoints;
	this->m_grab1 = other.m_grab1;
	this->m_grab2 = other.m_grab2;

	return *this;
}
