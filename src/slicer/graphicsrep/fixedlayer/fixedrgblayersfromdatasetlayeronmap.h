#ifndef FixedRGBLayersFromDatasetLayerOnMap_H
#define FixedRGBLayersFromDatasetLayerOnMap_H

#include "graphiclayer.h"
class QGraphicsItem;
class RGBQGLCUDAImageItem;
class FixedRGBLayersFromDatasetRep;
class QGraphicsScene;
class QGLImageFilledHistogramItem;

class FixedRGBLayersFromDatasetLayerOnMap : public GraphicLayer{
	  Q_OBJECT
public:
	  FixedRGBLayersFromDatasetLayerOnMap(FixedRGBLayersFromDatasetRep *rep,QGraphicsScene *scene,
			 int defaultZDepth,QGraphicsItem *parent);
	virtual ~FixedRGBLayersFromDatasetLayerOnMap();

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;

public slots:
	virtual void refresh() override;

protected:
	RGBQGLCUDAImageItem *m_item;
	FixedRGBLayersFromDatasetRep *m_rep;
};

#endif
