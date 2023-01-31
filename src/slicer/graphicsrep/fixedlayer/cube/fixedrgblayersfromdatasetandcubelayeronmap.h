#ifndef FixedRGBLayersFromDatasetAndCubeLayerOnMap_H
#define FixedRGBLayersFromDatasetAndCubeLayerOnMap_H

#include "graphiclayer.h"
class QGraphicsItem;
class RGBInterleavedQGLCUDAImageItem;
class FixedRGBLayersFromDatasetAndCubeRep;
class QGraphicsScene;
class QGLImageFilledHistogramItem;

class FixedRGBLayersFromDatasetAndCubeLayerOnMap : public GraphicLayer{
	  Q_OBJECT
public:
	  FixedRGBLayersFromDatasetAndCubeLayerOnMap(FixedRGBLayersFromDatasetAndCubeRep *rep,QGraphicsScene *scene,
			 int defaultZDepth,QGraphicsItem *parent);
	virtual ~FixedRGBLayersFromDatasetAndCubeLayerOnMap();

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;

public slots:
	virtual void refresh() override;

protected slots:
	void minValueActivated(bool activated);
	void minValueChanged(float value);

protected:
	RGBInterleavedQGLCUDAImageItem *m_item;
	FixedRGBLayersFromDatasetAndCubeRep *m_rep;
};

#endif
