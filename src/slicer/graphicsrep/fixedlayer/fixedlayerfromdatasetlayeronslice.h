#ifndef FixedLayerFromDatasetLayerOnSlice_H
#define FixedLayerFromDatasetLayerOnSlice_H

#include "graphiclayer.h"
#include "sliceutils.h"
class QGraphicsItem;
class FixedLayerFromDatasetRepOnSlice;
class QGraphicsScene;
class QGLIsolineItem;
class IGeorefImage;
class Curve;

class FixedLayerFromDatasetLayerOnSlice : public GraphicLayer{
	  Q_OBJECT
public:
	  FixedLayerFromDatasetLayerOnSlice(FixedLayerFromDatasetRepOnSlice *rep,SliceDirection dir,const IGeorefImage * const transfoProvider,int startValue,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent);
	virtual ~FixedLayerFromDatasetLayerOnSlice();

	void setSliceIJPosition(int imageVal);
	void setPenColor(QColor color);
	QColor penColor();

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;
public slots:
	virtual void refresh() override;
	void recomputeCurve();

protected:
	Curve *m_curve;
	FixedLayerFromDatasetRepOnSlice *m_rep;
	SliceDirection m_dir;
	int m_slicePosition;

	const IGeorefImage * const m_transfoProvider;
	bool m_isShown = false;
};

#endif
