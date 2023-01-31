#ifndef VideoLayerGraphicRepFactory_H
#define VideoLayerGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"
#include "sliceutils.h"

class VideoLayer;

class VideoLayerGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	VideoLayerGraphicRepFactory(VideoLayer * data);
	virtual ~VideoLayerGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	VideoLayer * m_data;

};

#endif
