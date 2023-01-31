#ifndef RgbLayerFromDatasetLayer_H
#define RgbLayerFromDatasetLayer_H

#include "graphiclayer.h"
class QGraphicsItem;
class RgbLayerFromDatasetRep;
class QGraphicsScene;
class RGBQGLCUDAImageItem;

class RgbLayerFromDatasetLayer : public GraphicLayer{
	  Q_OBJECT
public:
	  RgbLayerFromDatasetLayer(RgbLayerFromDatasetRep *rep,QGraphicsScene *scene,
			 int defaultZDepth,QGraphicsItem *parent);
	virtual ~RgbLayerFromDatasetLayer();

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;

public slots:
	virtual void refresh() override;

protected:
	RGBQGLCUDAImageItem *m_item;
	RgbLayerFromDatasetRep *m_rep;
};

#endif
