#include "mousetrackingevent.h"

QEvent::Type MouseTrackingEvent::customEventType = QEvent::None;

MouseTrackingEvent::MouseTrackingEvent(double worldx, double worldy,double depth, SampleUnit depthUnit) :
		QEvent(MouseTrackingEvent::type())  {
	m_worldx=worldx;
	m_worldy=worldy;
	m_depth=depth;
	m_hasDepth=true;
	m_depthUnit=depthUnit;
}

MouseTrackingEvent::MouseTrackingEvent() : QEvent(MouseTrackingEvent::type())
{
	m_worldx=0;
	m_worldy=0;
	m_depth=0;
	m_hasDepth=false;
	m_depthUnit=SampleUnit::NONE;
}

MouseTrackingEvent::MouseTrackingEvent(const MouseTrackingEvent &par):QEvent(par) {
	this->m_worldx = par.m_worldx;
	this->m_worldy = par.m_worldy;
	this->m_depth = par.m_depth;
	this->m_hasDepth = par.m_hasDepth;
	this->m_depthUnit = par.m_depthUnit;
}

MouseTrackingEvent& MouseTrackingEvent::operator=(const MouseTrackingEvent &par) {
	if (this != &par) {
		this->m_worldx = par.m_worldx;
		this->m_worldy = par.m_worldy;
		this->m_depth = par.m_depth;
		this->m_hasDepth = par.m_hasDepth;
		this->m_depthUnit = par.m_depthUnit;
	}
	return *this;
}

void MouseTrackingEvent::setPos(double worldx, double worldy,double depth, SampleUnit depthUnit)
{
	m_worldx=worldx;
	m_worldy=worldy;
	m_depth=depth;
	m_hasDepth=true;
	m_depthUnit=depthUnit;
}

void MouseTrackingEvent::setPos(double worldx, double worldy)
{
	m_worldx=worldx;
	m_worldy=worldy;
	m_hasDepth=false;
}


MouseTrackingEvent::~MouseTrackingEvent() {
	// TODO Auto-generated destructor stub
}

