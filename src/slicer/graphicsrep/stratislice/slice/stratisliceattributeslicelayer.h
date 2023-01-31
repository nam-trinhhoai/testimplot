#ifndef StratiSliceAttributeSliceLayer_H
#define StratiSliceAttributeSliceLayer_H

#include "graphiclayer.h"
#include "sliceutils.h"

class QGraphicsItem;
class StratiSliceAttributeRepOnSlice;
class QGraphicsScene;
class QGLIsolineItem;
class IGeorefImage;

class StratiSliceAttributeSliceLayer : public GraphicLayer{
	  Q_OBJECT
public:
	  StratiSliceAttributeSliceLayer(StratiSliceAttributeRepOnSlice *rep,SliceDirection dir,const IGeorefImage * const transfoProvider,int startValue,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent);
	virtual ~StratiSliceAttributeSliceLayer();

	void setSliceIJPosition(int imageVal);

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;
public slots:
	virtual void refresh() override;

protected:
	QGLIsolineItem *m_lineItem;
	StratiSliceAttributeRepOnSlice *m_rep;

	const IGeorefImage * const m_transfoProvider;
};

#endif
