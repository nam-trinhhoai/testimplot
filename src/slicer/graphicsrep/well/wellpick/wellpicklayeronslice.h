#ifndef WellPickLayerOnSlice_H
#define WellPickLayerOnSlice_H

#include "graphiclayer.h"

class QGraphicsView;
class QGraphicsItem;
class QGraphicsLineItem;
class QGLImageGridItem;
class WellPickRepOnSlice;
class QEvent;

class WellPickLayerOnSlice : public GraphicLayer{
	  Q_OBJECT
public:
	  WellPickLayerOnSlice(WellPickRepOnSlice *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem * parent);
	virtual ~WellPickLayerOnSlice();

	virtual void show() override;
	virtual void hide() override;
	virtual void hide(bool soft);

	virtual QRectF boundingRect() const override;
	virtual void refresh() override;

	void reloadItems();

private:
	WellPickRepOnSlice *m_rep;

	QGraphicsLineItem* m_lineItem;
	QGraphicsView* m_view;
	bool m_isShown = false;
	bool m_isVisibleOnSection = false;
};

#endif
