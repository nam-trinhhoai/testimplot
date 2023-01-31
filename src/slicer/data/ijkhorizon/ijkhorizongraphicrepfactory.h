#ifndef IJKHorizonGraphicRepFactory_H
#define IJKHorizonGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"
#include "sliceutils.h"

class IJKHorizon;

class IJKHorizonGraphicRepFactory :public IGraphicRepFactory {
	Q_OBJECT
public:
	IJKHorizonGraphicRepFactory(IJKHorizon * data);
	virtual ~IJKHorizonGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	IJKHorizon * m_data;

};

#endif
