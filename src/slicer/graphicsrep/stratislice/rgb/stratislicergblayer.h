#ifndef StratiSliceRGBLayer_H
#define StratiSliceRGBLayer_H

#include "graphiclayer.h"

class QGraphicsItem;
class RGBQGLCUDAImageItem;
class StratiSliceRGBAttributeRep;

class StratiSliceRGBLayer : public GraphicLayer{
	  Q_OBJECT
public:
	  StratiSliceRGBLayer(StratiSliceRGBAttributeRep *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent);
	virtual ~StratiSliceRGBLayer();

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;




public slots:
	virtual void refresh() override;

protected:
	RGBQGLCUDAImageItem *m_item;
	StratiSliceRGBAttributeRep *m_rep;
};

#endif
