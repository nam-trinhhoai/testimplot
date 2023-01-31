#ifndef FixedLayerFromDatasetGraphicRepFactory_H
#define FixedLayerFromDatasetGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"

class FixedLayerFromDataset;

class FixedLayerFromDatasetGraphicRepFactory:public IGraphicRepFactory{
	Q_OBJECT
public:
	FixedLayerFromDatasetGraphicRepFactory(FixedLayerFromDataset * data);
	virtual ~FixedLayerFromDatasetGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	FixedLayerFromDataset * m_data;

};

#endif
