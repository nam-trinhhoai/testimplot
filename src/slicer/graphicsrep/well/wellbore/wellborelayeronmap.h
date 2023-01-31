#ifndef WellBoreLayerOnMap_H
#define WellBoreLayerOnMap_H

#include "graphiclayer.h"
#include <QGraphicsPathItem>

class QGraphicsView;
class QGraphicsItem;
class QGraphicsSimpleTextItem ;
class QGLImageGridItem;
class WellBoreRepOnMap;
class QEvent;

class WellBoreLayerOnMap : public GraphicLayer{
	  Q_OBJECT
public:
	WellBoreLayerOnMap(WellBoreRepOnMap *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem * parent);
	virtual ~WellBoreLayerOnMap();

	virtual void show() override;
	virtual void hide() override;

	virtual QRectF boundingRect() const override;

	QGraphicsItem* graphicsItem() const
	{
		return 	m_item;
	}

	protected slots:
	virtual void refresh() override;

//	bool eventFilter(QObject* watched, QEvent* ev) override;

public slots:
	void updateFromZoom();
	void setColorWellChanged(QColor );
	void setWidth(double val);
private:
	WellBoreRepOnMap *m_rep;

	QGraphicsPathItem* m_item;
	QGraphicsSimpleTextItem* m_textItem;
	QGraphicsView* m_view;

	double m_cachedScale = 1;
	bool m_scaleInit = false;
};

#endif
