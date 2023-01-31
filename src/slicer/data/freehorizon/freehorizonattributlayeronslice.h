
#ifndef __FREEHORIZONATTRIBUTLAYERONSLICE__
#define __FREEHORIZONATTRIBUTLAYERONSLICE__

#include "graphiclayer.h"
#include "sliceutils.h"
#include "curve.h"

class QGraphicsItem;
class FreeHorizonAttributRepOnSlice;
class QGraphicsScene;
class QGLIsolineItem;
class IGeorefImage;

class FreeHorizonAttributLayerOnSlice : public GraphicLayer{
	  Q_OBJECT
public:
	  FreeHorizonAttributLayerOnSlice(FreeHorizonAttributRepOnSlice *rep,SliceDirection dir,
			int startValue,QGraphicsScene *scene,
			int defaultZDepth,QGraphicsItem *parent);
	virtual ~FreeHorizonAttributLayerOnSlice();

	void setSliceIJPosition(int imageVal);

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;

public slots:
	virtual void refresh() override;

protected:
	//QGLIsolineItem *m_lineItem;
	std::unique_ptr<Curve> m_curveMain;
	FreeHorizonAttributRepOnSlice *m_rep;
	QTransform m_mainTransform;
};

#endif
