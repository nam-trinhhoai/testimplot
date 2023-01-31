#ifndef LayerRGTSliceLayer_H
#define LayerRGTSliceLayer_H

#include "graphiclayer.h"
#include "sliceutils.h"
class QGraphicsItem;
class LayerRGTRepOnSlice;
class QGraphicsScene;
class QGLIsolineItem;
class IGeorefImage;

class LayerRGTSliceLayer : public GraphicLayer{
	  Q_OBJECT
public:
    LayerRGTSliceLayer(LayerRGTRepOnSlice *rep,SliceDirection dir,const IGeorefImage * const transfoProvider,int startValue,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent);
	virtual ~LayerRGTSliceLayer();

	void setSliceIJPosition(int imageVal);

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;
public slots:
	virtual void refresh() override;

protected:
	QGLIsolineItem *m_lineItem;
	LayerRGTRepOnSlice *m_rep;

	const IGeorefImage * const m_transfoProvider;
};

#endif
