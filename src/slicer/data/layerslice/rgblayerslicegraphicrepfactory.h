#ifndef RGBLayerSliceGraphicRepFactory_H
#define RGBLayerSliceGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"

class RGBLayerSlice;

class RGBLayerSliceGraphicRepFactory:public IGraphicRepFactory{
	Q_OBJECT
public:
	RGBLayerSliceGraphicRepFactory(RGBLayerSlice * data);
	virtual ~RGBLayerSliceGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	RGBLayerSlice * m_data;

};

#endif
