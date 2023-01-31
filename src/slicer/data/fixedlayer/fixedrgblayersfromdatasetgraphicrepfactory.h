#ifndef FixedRGBLayersFromDatasetGraphicRepFactory_H
#define FixedRGBLayersFromDatasetGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"

class FixedRGBLayersFromDataset;

class FixedRGBLayersFromDatasetGraphicRepFactory:public IGraphicRepFactory{
	Q_OBJECT
public:
	FixedRGBLayersFromDatasetGraphicRepFactory(FixedRGBLayersFromDataset * data);
	virtual ~FixedRGBLayersFromDatasetGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	FixedRGBLayersFromDataset * m_data;

};

#endif
