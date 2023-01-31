#ifndef FixedLayerFromDatasetLayerOnRandom_H
#define FixedLayerFromDatasetLayerOnRandom_H

#include "graphiclayer.h"
#include "sliceutils.h"
class QGraphicsItem;
class FixedLayerFromDatasetRepOnRandom;
class QGraphicsScene;
class QGLIsolineItem;
class IGeorefImage;
class Curve;

class FixedLayerFromDatasetLayerOnRandom : public GraphicLayer{
	  Q_OBJECT
public:
	FixedLayerFromDatasetLayerOnRandom(FixedLayerFromDatasetRepOnRandom *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent);
	virtual ~FixedLayerFromDatasetLayerOnRandom();

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
	FixedLayerFromDatasetRepOnRandom *m_rep;

	bool m_isShown = false;
};

#endif
