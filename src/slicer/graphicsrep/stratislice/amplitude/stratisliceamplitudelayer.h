#ifndef StratiSliceAmplitudeLayer_H
#define StratiSliceAmplitudeLayer_H

#include "graphiclayer.h"
class QGraphicsItem;
class RGTQGLCUDAImageItem;
class StratiSliceAmplitudeRep;
class QGraphicsScene;
class QGLImageFilledHistogramItem;

class StratiSliceAmplitudeLayer : public GraphicLayer{
	  Q_OBJECT
public:
	StratiSliceAmplitudeLayer(StratiSliceAmplitudeRep *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent);
	virtual ~StratiSliceAmplitudeLayer();

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
	StratiSliceAmplitudeRep *m_rep;
};

#endif
