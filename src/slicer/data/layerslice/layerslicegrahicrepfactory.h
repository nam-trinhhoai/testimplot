#ifndef LayerSliceGraphicRepFactory_H
#define LayerSliceGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"

class LayerSlice;

class LayerSliceGraphicRepFactory:public IGraphicRepFactory{
	Q_OBJECT
public:
	LayerSliceGraphicRepFactory(LayerSlice * data);
	virtual ~LayerSliceGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	LayerSlice * m_data;

};

#endif
