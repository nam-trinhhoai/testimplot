#ifndef FixedLayersFromDatasetAndCubeLayerOnMap_H
#define FixedLayersFromDatasetAndCubeLayerOnMap_H

#include "graphiclayer.h"
class QGraphicsItem;
class RGTQGLCUDAImageItem;
class FixedLayersFromDatasetAndCubeRep;
class QGraphicsScene;
class QGLImageFilledHistogramItem;

class FixedLayersFromDatasetAndCubeLayerOnMap : public GraphicLayer{
	  Q_OBJECT
public:
	  FixedLayersFromDatasetAndCubeLayerOnMap(FixedLayersFromDatasetAndCubeRep *rep,QGraphicsScene *scene,
			 int defaultZDepth,QGraphicsItem *parent);
	virtual ~FixedLayersFromDatasetAndCubeLayerOnMap();

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;

public slots:
	virtual void refresh() override;

protected:
	RGTQGLCUDAImageItem *m_item;
	FixedLayersFromDatasetAndCubeRep *m_rep;
};

#endif
