



#ifndef __RGBLAYERFREEHORIZONGRAPHICREPFACTORY__
#define __RGBLAYERFREEHORIZONGRAPHICREPFACTORY__

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"

class RGBLayerImplFreeHorizonOnSlice;


class RGBLayerFreeHorizonGraphicRepFactory:public IGraphicRepFactory{
	Q_OBJECT
public:
	RGBLayerFreeHorizonGraphicRepFactory(RGBLayerImplFreeHorizonOnSlice * data);
	virtual ~RGBLayerFreeHorizonGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	RGBLayerImplFreeHorizonOnSlice * m_data;

};

#endif
