#ifndef StratiSliceFrequencyLayer_H
#define StratiSliceFrequencyLayer_H

#include "graphiclayer.h"
class QGraphicsItem;
class RGTQGLCUDAImageItem;
class StratiSliceFrequencyRep;
class QGraphicsScene;
class QGLImageFilledHistogramItem;

class StratiSliceFrequencyLayer : public GraphicLayer{
	  Q_OBJECT
public:
	  StratiSliceFrequencyLayer(StratiSliceFrequencyRep *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent);
	virtual ~StratiSliceFrequencyLayer();

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
	StratiSliceFrequencyRep *m_rep;
};

#endif
