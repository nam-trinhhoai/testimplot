#ifndef RgbDatasetLayerOnRandom_H
#define RgbDatasetLayerOnRandom_H

#include "graphiclayer.h"
class QGraphicsItem;
class QGLFullCUDARgbaImageItem;
class RgbDatasetRepOnRandom;
class QGLColorBar;

class RgbDatasetLayerOnRandom : public GraphicLayer{
	  Q_OBJECT
public:
	RgbDatasetLayerOnRandom(RgbDatasetRepOnRandom *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent);
	virtual ~RgbDatasetLayerOnRandom();

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;

public slots:
	virtual void refresh() override;
	void modeChanged();
	void constantAlphaChanged();
	void radiusAlphaChanged();
protected:
	QGLFullCUDARgbaImageItem *m_item;
	RgbDatasetRepOnRandom *m_rep;

	bool m_isShown = false;
	QGraphicsItem* m_parent;
};

#endif
