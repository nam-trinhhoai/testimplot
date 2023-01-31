#ifndef LayerDatasetGraphicRepFactory_H
#define LayerDatasetGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"
#include "sliceutils.h"

class LayerSlice;

class LayerDatasetGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	LayerDatasetGraphicRepFactory(LayerSlice * data);
	virtual ~LayerDatasetGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	LayerSlice * m_data;

};

#endif
