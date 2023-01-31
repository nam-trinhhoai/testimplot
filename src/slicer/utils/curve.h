#ifndef Curve_H_
#define Curve_H_

#include <QPolygon>
#include <QPen>
#include <QTransform>
#include <QList>

class QGraphicsPathItem;
class QGraphicsScene;
class QGraphicsItem;
class TmpClass;

/**
 * Side note : the provided scene need to have at least one QGraphicsView to compute pen width during "redraw"
 */
class Curve {
	friend class TmpClass;
public:
	Curve(QGraphicsScene* scene, QGraphicsItem* parent=0);
	virtual ~Curve();

	QPolygon getPolygon();
	void setPolygon(QPolygon poly);
	QList<QPolygon> getPolygons();
	void setPolygons(QList<QPolygon> poly);
	QPen getPen();
	void setPen(QPen pen);
	QBrush getBrush();
	void setBrush(QBrush brush);


	QTransform getTransform();
	void setTransform(QTransform transfo);

	void redraw();

	void show();
	void hide();

	void setZValue(int);
	QRectF boundingRect() const;

	void addToScene();
	void removeFromScene();

private:
	void resetCurve();
	void internalRedraw();

	QList<QPolygon> m_polygons;
	bool m_drawLock = false;
	bool m_recallDraw = false;
	bool m_isInScene = false;

	QGraphicsScene* m_canvas = nullptr;
	QGraphicsPathItem* m_curve = nullptr;
	QPen m_pen;
    QTransform m_transform;
};

#endif
