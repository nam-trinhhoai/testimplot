#ifndef WellHeadLayerOnMap_H
#define WellHeadLayerOnMap_H

#include "graphiclayer.h"
#include <QGraphicsSvgItem>

class QGraphicsView;
class QGraphicsItem;
class QGraphicsSvgItem;
class QGLImageGridItem;
class WellHeadRepOnMap;
class QEvent;

class WellHeadLayerOnMap : public GraphicLayer{
	  Q_OBJECT
public:
	  WellHeadLayerOnMap(WellHeadRepOnMap *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem * parent);
	virtual ~WellHeadLayerOnMap();

	virtual void show() override;
	virtual void hide() override;

	virtual QRectF boundingRect() const override;

	QGraphicsItem* graphicsItem() const
	{
		return 	m_item;
	}

protected slots:
	virtual void refresh() override;
	bool eventFilter(QObject* watched, QEvent* ev) override;

public slots:
	void updateFromZoom();
signals:
	void zoomExecuted();
private:
	WellHeadRepOnMap *m_rep;

	QGraphicsSvgItem* m_item;
	QGraphicsView* m_view;

	double m_cachedScale = 1;
	bool m_scaleInit = false;
};

#endif
