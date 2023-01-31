#ifndef StackLayerRGTLayer_H
#define StackLayerRGTLayer_H

#include "graphiclayer.h"
class QGraphicsItem;
class RGTQGLCUDAImageItem;
class StackLayerRGTRep;
class QGraphicsScene;
class QGLImageFilledHistogramItem;

class StackLayerRGTLayer : public GraphicLayer{
	  Q_OBJECT
public:
	 StackLayerRGTLayer(StackLayerRGTRep *rep,QGraphicsScene *scene,
			 int defaultZDepth,QGraphicsItem *parent);
	virtual ~StackLayerRGTLayer();

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
	StackLayerRGTRep *m_rep;
};

#endif
