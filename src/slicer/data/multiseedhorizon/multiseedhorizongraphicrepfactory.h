#ifndef MultiSeedHorizonGraphicRepFactory_H
#define MultiSeedHorizonGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"
#include "sliceutils.h"

class MultiSeedHorizon;

class MultiSeedHorizonGraphicRepFactory :public IGraphicRepFactory {
	Q_OBJECT
public:
	MultiSeedHorizonGraphicRepFactory(MultiSeedHorizon * data);
	virtual ~MultiSeedHorizonGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	MultiSeedHorizon * m_data;

};

#endif
