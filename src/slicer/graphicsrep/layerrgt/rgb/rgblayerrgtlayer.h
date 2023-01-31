#ifndef RGBLayerRGTLayer_H
#define RGBLayerRGTLayer_H

#include "graphiclayer.h"

class QGraphicsItem;
class RGBQGLCUDAImageItem;
class RGBLayerRGTRep;

class RGBLayerRGTLayer : public GraphicLayer{
	  Q_OBJECT
public:
	  RGBLayerRGTLayer(RGBLayerRGTRep *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent);
	virtual ~RGBLayerRGTLayer();

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;




public slots:
	virtual void refresh() override;

protected slots:
	void minValueActivated(bool activated);
	void minValueChanged(float value);

protected:
	RGBQGLCUDAImageItem *m_item;
	RGBLayerRGTRep *m_rep;
};

#endif
