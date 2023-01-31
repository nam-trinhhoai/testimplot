#ifndef MultiSeedRgtGraphicRepFactory_H
#define MultiSeedRgtGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"
#include "sliceutils.h"

class MultiSeedRgt;

class MultiSeedRgtGraphicRepFactory :public IGraphicRepFactory {
	Q_OBJECT
public:
	MultiSeedRgtGraphicRepFactory(MultiSeedRgt * data);
	virtual ~MultiSeedRgtGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	MultiSeedRgt * m_data;

};

#endif
