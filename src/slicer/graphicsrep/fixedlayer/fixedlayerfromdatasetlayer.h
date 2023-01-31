#ifndef FixedLayerFromDatasetLayer_H
#define FixedLayerFromDatasetLayer_H

#include "graphiclayer.h"
class QGraphicsItem;
class RGTQGLCUDAImageItem;
class FixedLayerFromDatasetRep;
class QGraphicsScene;
class QGLImageFilledHistogramItem;

class FixedLayerFromDatasetLayer : public GraphicLayer{
	  Q_OBJECT
public:
	  FixedLayerFromDatasetLayer(FixedLayerFromDatasetRep *rep,QGraphicsScene *scene,
			 int defaultZDepth,QGraphicsItem *parent);
	virtual ~FixedLayerFromDatasetLayer();

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;

    void showCrossHair(bool val);
    virtual void mouseMoved(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys)override;

public slots:
	virtual void refresh() override;

protected:
	RGTQGLCUDAImageItem *m_item;
	QGLImageFilledHistogramItem *m_histoItem;
	FixedLayerFromDatasetRep *m_rep;
};

#endif
