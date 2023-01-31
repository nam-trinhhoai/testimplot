#ifndef RgbDatasetLayerOnSlice_H
#define RgbDatasetLayerOnSlice_H

#include "graphiclayer.h"
class QGraphicsItem;
class QGLFullCUDARgbaImageItem;
class RgbDatasetRepOnSlice;
class QGLColorBar;

class RgbDatasetLayerOnSlice : public GraphicLayer{
	  Q_OBJECT
public:
	RgbDatasetLayerOnSlice(RgbDatasetRepOnSlice *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent);
	virtual ~RgbDatasetLayerOnSlice();

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
	RgbDatasetRepOnSlice *m_rep;

	bool m_isShown = false;
	QGraphicsItem* m_parent;
};

#endif
