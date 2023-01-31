#ifndef LayerRGTLayer_H
#define LayerRGTLayer_H

#include "graphiclayer.h"
class QGraphicsItem;
class RGTQGLCUDAImageItem;
class LayerRGTRep;
class QGraphicsScene;
class QGLImageFilledHistogramItem;

class LayerRGTLayer : public GraphicLayer{
	  Q_OBJECT
public:
	 LayerRGTLayer(LayerRGTRep *rep,QGraphicsScene *scene,
			 int defaultZDepth,QGraphicsItem *parent);
	virtual ~LayerRGTLayer();

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
	LayerRGTRep *m_rep;
};

#endif
