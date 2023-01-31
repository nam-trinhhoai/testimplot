#ifndef RgbLayerFromDatasetGraphicRepFactory_H
#define RgbLayerFromDatasetGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"

class RgbLayerFromDataset;

class RgbLayerFromDatasetGraphicRepFactory:public IGraphicRepFactory{
	Q_OBJECT
public:
	RgbLayerFromDatasetGraphicRepFactory(RgbLayerFromDataset * data);
	virtual ~RgbLayerFromDatasetGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	RgbLayerFromDataset * m_data;

};

#endif
