#ifndef WellPickLayerOnRandom_H
#define WellPickLayerOnRandom_H

#include "graphiclayer.h"
#include <QList>

class QGraphicsView;
class QGraphicsItem;
class QGraphicsLineItem;
class QGLImageGridItem;
class WellPickRepOnRandom;
class QEvent;

class WellPickLayerOnRandom : public GraphicLayer{
	  Q_OBJECT
public:
	  WellPickLayerOnRandom(WellPickRepOnRandom *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem * parent);
	virtual ~WellPickLayerOnRandom();

	virtual void show() override;
	virtual void hide() override;

	virtual QRectF boundingRect() const override;
	virtual void refresh() override;

private:
	void internalShow();
	void internalHide();

	QGraphicsLineItem* getLineItem() const;

	WellPickRepOnRandom *m_rep;

	QList<QGraphicsLineItem*> m_lineItems;
	QGraphicsItem* m_parent;
	bool m_isShown = false;
	bool m_requestedShown = false;
};

#endif
