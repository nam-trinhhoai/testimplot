#ifndef FixedRGBLayersFromDatasetLayerOnSlice_H
#define FixedRGBLayersFromDatasetLayerOnSlice_H

#include "graphiclayer.h"
#include "sliceutils.h"
#include "curve.h"

class QGraphicsItem;
class FixedRGBLayersFromDatasetRepOnSlice;
class QGraphicsScene;
class QGLIsolineItem;
class IGeorefImage;

class FixedRGBLayersFromDatasetLayerOnSlice : public GraphicLayer{
	  Q_OBJECT
public:
	FixedRGBLayersFromDatasetLayerOnSlice(FixedRGBLayersFromDatasetRepOnSlice *rep,SliceDirection dir,
			const IGeorefImage * const transfoProvider,int startValue,QGraphicsScene *scene,
			int defaultZDepth,QGraphicsItem *parent);
	virtual ~FixedRGBLayersFromDatasetLayerOnSlice();

	void setSliceIJPosition(int imageVal);

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;
public slots:
	virtual void refresh() override;

protected:
	//QGLIsolineItem *m_lineItem;
	std::unique_ptr<Curve> m_curveMain;
	FixedRGBLayersFromDatasetRepOnSlice *m_rep;

	const IGeorefImage * const m_transfoProvider;
};

#endif
