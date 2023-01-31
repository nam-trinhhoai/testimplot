#include "curve.h"

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPathItem>
#include <QGraphicsItem>
#include <QPainter>
#include <QDebug>
#include <QObject>

#include <cmath>

class TmpClass : public QGraphicsPathItem {
public:
	TmpClass(Curve* parentCurve, QGraphicsItem* parent) :
		QGraphicsPathItem(parent) {
		m_parentCurve = parentCurve;
	}
	TmpClass(Curve* parentCurve) {
		m_parentCurve = parentCurve;
	}
	virtual ~TmpClass() {
		m_parentCurve->resetCurve();
	}
private:
	Curve* m_parentCurve;
};

Curve::Curve(QGraphicsScene* scene, QGraphicsItem* parent) {
	m_canvas = scene;

	m_transform = QTransform( 1, 0, 0, 0, 1, 0, 0, 0, 1);
	if(parent==nullptr) {
		m_curve = new TmpClass(this);
		m_curve->setPen(m_pen);
		m_canvas->addItem(m_curve);
	} else {
		m_curve = new TmpClass(this, parent);
		m_curve->setPen(m_pen);
	}
	m_isInScene = true;
	m_curve->show();
}

Curve::~Curve() {
	if (m_curve!=nullptr && m_isInScene) {
		m_canvas->removeItem(m_curve);
		delete m_curve;
	}
}

QPolygon Curve::getPolygon() {
	if (m_polygons.count()>0) {
		return m_polygons[0];
	} else {
		return QPolygon();
	}
}

void Curve::setPolygon(QPolygon poly) {
	m_polygons.clear();
	m_polygons << poly;
	redraw();
}

QList<QPolygon> Curve::getPolygons() {
	return m_polygons;
}

void Curve::setPolygons(QList<QPolygon> poly) {
	m_polygons = poly;
	redraw();
}

QPen Curve::getPen() {
	return m_pen;
}

void Curve::setPen(QPen pen) {
	m_pen = pen;
	if (m_curve!=nullptr) {
		m_curve->setPen(pen);
		redraw();
	}
}

QBrush Curve::getBrush() {
	return m_pen.brush();
}

void Curve::setBrush(QBrush brush) {
	m_pen.setBrush(brush);
	if (m_curve!=nullptr) {
		m_curve->setPen(m_pen);
		redraw();
	}
}


QTransform Curve::getTransform() {
	return m_transform;
}

void Curve::setTransform(QTransform transfo) {
	m_transform = transfo;
	redraw();
}

void Curve::redraw() {
	if (m_drawLock) {
		m_recallDraw = true;
		return;
	} else {
		m_drawLock = true;

		internalRedraw();

		m_drawLock = false;
		if (m_recallDraw) {
			m_recallDraw = false;
			redraw();
		}
	}
}

void Curve::show() {
	if (m_curve!=nullptr && m_isInScene) {
		m_curve->show();
	}
}

void Curve::hide() {
	if (m_curve!=nullptr && m_isInScene) {
		m_curve->hide();
	}
}

void Curve::addToScene() {
	if (m_curve!=nullptr && !m_isInScene) {
		m_canvas->addItem(m_curve);
		m_isInScene = true;
	}
}

void Curve::removeFromScene() {
	if (m_curve!=nullptr && m_isInScene) {
		m_canvas->removeItem(m_curve);
		m_isInScene = false;
	}
}

void Curve::internalRedraw() {
	if (m_curve==nullptr) {
		return;
	}
	double ratioW = 1.0;
	double ratioH = 1.0;

	// get screen pixel size in view
	QPoint ptOri(0, 0);
	QPoint ptH(0, 1);
	QPoint ptW(1, 0);
	QPointF dW = m_canvas->views()[0]->mapToScene(ptW) - m_canvas->views()[0]->mapToScene(ptOri);
	QPointF dH = m_canvas->views()[0]->mapToScene(ptH) - m_canvas->views()[0]->mapToScene(ptOri);
	ratioW = std::sqrt(std::pow(dW.x(), 2) + std::pow(dW.y(), 2));
	ratioH = std::sqrt(std::pow(dH.x(), 2) + std::pow(dH.y(), 2));

	QPen pen = m_curve->pen();
//	if (ratioW>ratioH) {
//		pen.setWidthF(ratioW*2);
//	} else {
//		pen.setWidthF(ratioH*2);
//	}
	pen.setWidth(2);
	pen.setCosmetic(true);
	m_curve->setPen(pen);

	// Draw
	QPainterPath painter;
	for (QPolygon& polygon : m_polygons) {
		if (polygon.size()>1) {
			painter.addPolygon(polygon);
		} else if (polygon.size()==1) {
			painter.addEllipse(polygon.at(0), 0.5, 0.5);
		}
	}
	//QTransform transform(m_transform);
	painter = m_transform.map(painter);
	m_curve->setPath(painter);
}

void Curve::setZValue(int z) {
	if (m_curve!=nullptr) {
		m_curve->setZValue(z);
	}
}

// This should only be called by TmpClass
void Curve::resetCurve() {
	m_curve = nullptr;
}

QRectF Curve::boundingRect() const {
	QRectF rect;
	// get all bounding rectangles
	for (std::size_t i=0; i<m_polygons.size(); i++) {
		const QPolygon& poly = m_polygons[i];
		QPolygonF polyF;
		for (QPoint pt : poly) {
			polyF << pt;
		}
		QRectF newRect = m_transform.map(polyF).boundingRect();
		if (i==0) {
			rect = newRect;
		} else {
			rect = rect.united(newRect);
		}
	}

	return rect;
}
