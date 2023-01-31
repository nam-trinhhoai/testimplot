#include "qrect3d.h"

QRect3D::QRect3D(double x, double y, double z,double width, double height, double depth)
{
	m_width=width;
	m_height=height;
	m_depth=depth;

	m_x=x;
	m_y=y;
	m_z=z;
	m_valid =true;
}
QRect3D::QRect3D()
{
	m_width=0;
	m_height=0;
	m_depth=0;

	m_x=0;
	m_y=0;
	m_z=0;

	m_valid = false;
}


QRect3D::QRect3D(const QRect3D &par) {
	this->m_x = par.m_x;
	this->m_y = par.m_y;
	this->m_z = par.m_z;

	this->m_width = par.m_width;
	this->m_height = par.m_height;
	this->m_depth = par.m_depth;

	this->m_valid = par.m_valid;
}

QRect3D& QRect3D::operator=(const QRect3D &par) {
	if (this != &par) {
		this->m_x = par.m_x;
		this->m_y = par.m_y;
		this->m_z = par.m_z;

		this->m_width = par.m_width;
		this->m_height = par.m_height;
		this->m_depth = par.m_depth;
		this->m_valid = par.m_valid;
	}
	return *this;
}

bool QRect3D::merge(const QRect3D & in)
{

	bool modified=false;

	if( this->m_valid && in.m_valid)
	{

		if(in.x()<m_x)
		{
			m_x=in.x();
			modified=true;
		}
		if(in.y()<m_y)
		{
			m_y=in.y();
			modified=true;
		}
		if(in.z()<m_z)
		{
			m_z=in.z();
			modified=true;
		}

		if(in.width()>m_width)
		{
			m_width=in.width();
			modified=true;
		}
		if(in.height()>m_height)
		{
			m_height=in.height();
			modified=true;
		}
		if(in.depth()>m_depth)
		{
			m_depth=in.depth();
			modified=true;
		}
	}
	if(!this->m_valid && in.m_valid)
	{
		m_x=in.x();
		m_y=in.y();
		m_z=in.z();
		m_width=in.width();
		m_height=in.height();
		m_depth=in.depth();
		m_valid=true;
		modified = true;
	}

	return modified;
}

QRect3D::~QRect3D()
{

}
