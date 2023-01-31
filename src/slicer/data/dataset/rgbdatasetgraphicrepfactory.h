#ifndef RgbDatasetGraphicRepFactory_H
#define RgbDatasetGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"
#include "sliceutils.h"

class RgbDataset;

class RgbDatasetGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	RgbDatasetGraphicRepFactory(RgbDataset * data);
	virtual ~RgbDatasetGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	RgbDataset * m_data;

};

#endif
