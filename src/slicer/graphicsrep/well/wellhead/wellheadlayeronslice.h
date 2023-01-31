#ifndef WellHeadLayerOnSlice_H
#define WellHeadLayerOnSlice_H

#include "graphiclayer.h"

class QGraphicsView;
class QGraphicsItem;
class QGraphicsLineItem;
class QGraphicsSvgItem;
class QGLImageGridItem;
class WellHeadRepOnSlice;
class QEvent;

class WellHeadLayerOnSlice : public GraphicLayer{
	  Q_OBJECT
public:
	  WellHeadLayerOnSlice(WellHeadRepOnSlice *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem * parent);
	virtual ~WellHeadLayerOnSlice();

	virtual void show() override;
	virtual void hide() override;
	virtual void hide(bool soft);

	virtual QRectF boundingRect() const override;
	virtual void refresh() override;

	void reloadItems();

protected slots:
	bool eventFilter(QObject* watched, QEvent* ev) override;

private slots:
	void updateFromZoom();

private:
	WellHeadRepOnSlice *m_rep;

	QGraphicsSvgItem* m_item;
	QGraphicsLineItem* m_lineItem;
	QGraphicsView* m_view;
	bool m_isShown = false;
	bool m_isVisibleOnSection = false;
};

#endif
