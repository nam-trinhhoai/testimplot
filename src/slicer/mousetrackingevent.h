#ifndef MouseTrackingEvent_H
#define MouseTrackingEvent_H

#include <QEvent>
#include "viewutils.h"

class MouseTrackingEvent: public QEvent {
public:
	MouseTrackingEvent(double worldx, double worldy, double depth, SampleUnit depthUnit);
	MouseTrackingEvent();

	MouseTrackingEvent(const MouseTrackingEvent&);
	MouseTrackingEvent& operator=(const MouseTrackingEvent&);

	virtual ~MouseTrackingEvent();

	static QEvent::Type type() {
		if (customEventType == QEvent::None) {
			int generatedType = QEvent::registerEventType();
			customEventType = static_cast<QEvent::Type>(generatedType);
		}
		return customEventType;
	}

	void setPos(double worldx, double worldy, double depth, SampleUnit depthUnit);
	void setPos(double worldx, double worldy);

	double worldX() const {
		return m_worldx;
	}

	double worldY() const {
		return m_worldy;
	}
	bool hasDepth() const {
		return m_hasDepth;
	}
	double depth() const {
		return m_depth;
	}
private:
	static QEvent::Type customEventType;

	double m_worldx;
	double m_worldy;

	double m_depth;
	bool m_hasDepth;
	SampleUnit m_depthUnit;
};

#endif
