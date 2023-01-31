#include "wellborelayeronrandom.h"
#include "wellborereponrandom.h"
#include "wellbore.h"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPathItem>
#include <QGraphicsItem>
#include <QEvent>

#include <cmath>

WellBoreLayerOnRandom::WellBoreLayerOnRandom(WellBoreRepOnRandom *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem * parent) :
GraphicLayer(scene, defaultZDepth) {
	m_rep=rep;
	m_view = nullptr;

	WellBore* wellBore = dynamic_cast<WellBore*>(rep->data());


	QPainterPath path;
//	path.addPolygon(poly);

	m_item = new QGraphicsPathItem(path, parent);
	m_item->setZValue(defaultZDepth);
	QPen pen = m_item->pen();
	pen.setCosmetic(true);
	pen.setColor(Qt::yellow);
	m_item->setPen(pen);
	m_item->setToolTip(wellBore->name());

	QPainterPath pathLog;
//	path.addPolygon(poly);

	m_itemLog = new QGraphicsPathItem(path, parent);
	m_itemLog->setZValue(defaultZDepth);
	pen = m_itemLog->pen();
	pen.setCosmetic(true);
	pen.setColor(wellBore->logColor());
	m_itemLog->setPen(pen);
	m_itemLog->setToolTip(wellBore->name());

	connect(wellBore, &WellBore::logColorChanged, this, &WellBoreLayerOnRandom::setLogColor);

	refresh();
}

WellBoreLayerOnRandom::~WellBoreLayerOnRandom() {
}

void WellBoreLayerOnRandom::show() {
	QList<QGraphicsView*> views = m_scene->views();
	if (views.size()<1) {
		m_view = nullptr;
		return;
	}
	m_view = views[0];

	m_isShown = true;
	m_scene->addItem(m_item);
	if (m_isShownLog) {
		m_scaleInit = false;
		refreshLog();
		m_scene->addItem(m_itemLog);
	}
	emit layerShownChanged(true);

//	m_view->installEventFilter(this);
}

void WellBoreLayerOnRandom::hide() {

	if(m_view != nullptr){
//		m_view->removeEventFilter(this);
		m_isShown = false;
		m_scene->removeItem(m_item);
		if (m_isShownLog) {
			m_scene->removeItem(m_itemLog);
		}

		m_view = nullptr;
	}
	emit layerShownChanged(false);
}

QRectF WellBoreLayerOnRandom::boundingRect() const
{
	QRectF rect = m_rep->boundingBox(); // could use united with textItem boundingRect but need to add its pos
	// ex : textItem->boundingRect() : 0, 0, 100, 20
	// -> need to add textItem pos to QRectF to have right bounding box. (also scale ?)
	return rect;
}

void WellBoreLayerOnRandom::refresh() {
	const std::vector<QPolygonF> & paths =  m_rep->displayTrajectories();
	QPainterPath path;
	for (QPolygonF poly : paths) {
		path.addPolygon(poly);
	}
	m_item->setPath(path);

	m_item->update();

	if (m_isShownLog) {
		m_scaleInit = false;
		refreshLog();
		m_itemLog->update();
	}
}

bool WellBoreLayerOnRandom::isShown() const {
	return m_isShown;
}

void WellBoreLayerOnRandom::setLogColor(QColor color) {
	QPen pen = m_itemLog->pen();
	pen.setColor(color);
	m_itemLog->setPen(pen);
}

void WellBoreLayerOnRandom::toggleLogDisplay(bool showLog) {
	if (m_isShownLog!=showLog) {
		m_isShownLog = showLog;
		if (m_isShownLog) {
			m_scaleInit = false;
			refreshLog();
			if (m_isShown) {
				m_scene->addItem(m_itemLog);
			}
		} else if (m_isShown) {

			m_scene->removeItem(m_itemLog);
		}
	}
}

void WellBoreLayerOnRandom::refreshLog() {
	if (m_view==nullptr) {
		return;
	}

	// get scale
	QGraphicsView* view = m_view;

	QSize viewSize = view->viewport()->size();

	QPoint topLeft(0, 0), topRight(viewSize.width(), 0), bottomLeft(0, viewSize.height());
	QPointF topLeft1 = view->mapToScene(topLeft);
	QPointF topRight1 = view->mapToScene(topRight);
	QPointF bottomLeft1 = view->mapToScene(bottomLeft);

	QPointF dWidthPt = (topRight1 - topLeft1);
	double dWidth = std::sqrt(std::pow(dWidthPt.x(),2) + std::pow(dWidthPt.y(), 2));
	QPointF dHeightPt = (bottomLeft1 - topLeft1);
	double dHeight = std::sqrt(std::pow(dHeightPt.x(),2) + std::pow(dHeightPt.y(), 2));

	double scale = std::min(dWidth/viewSize.width(), dHeight/viewSize.height());

	if (!m_scaleInit || std::fabs(scale-m_cachedScale)>std::min(0.001, scale/100.0)) {
		double displayWidth = m_width;
		double displayOffset = m_origin;
		double logDynamic = m_logMax - m_logMin;
		double logMin = m_logMin;

		// build polygon
		QPainterPath pathLog;
		const std::vector<std::vector<WellBoreRepOnRandom::LogGraphicPoint>>& logParams = m_rep->logDisplayParams();

		if (!qFuzzyIsNull(logDynamic)) {
			for (const std::vector<WellBoreRepOnRandom::LogGraphicPoint>& curveParams : logParams) {
				QPolygonF polygon;

				for (const WellBoreRepOnRandom::LogGraphicPoint& param : curveParams) {
					// compute standard display size
					double logValue = param.logValue;
		//			if (param.logValue<logMin) {
		//				logValue = logMin;
		//			} else if (param.logValue>logMin+logDynamic) {
		//				logValue = logMin+logDynamic;
		//			} else {
		//				logValue = param.logValue;
		//			}
					double val = displayWidth * (logValue - logMin) / logDynamic + displayOffset;
					// apply scaling
					val *= scale;
					polygon << param.refPoint + param.normal.toPointF() * val;
				}

				// apply polygon
				pathLog.addPolygon(polygon);
			}
		}

		m_itemLog->setPath(pathLog);
	}
	m_cachedScale = scale;
	m_scaleInit = true;
}

//bool WellBoreLayerOnRandom::eventFilter(QObject* watched, QEvent* ev) {
//	if (ev->type() == QEvent::Wheel) {
//		refreshLog();
//	}
//	return false;
//}

double WellBoreLayerOnRandom::origin() const {
	return m_origin;
}

void WellBoreLayerOnRandom::setOrigin(double val) {
	if (m_origin!=val) {
		m_origin = val;
		if (m_isShownLog) {
			m_scaleInit = false;
			refreshLog();
		}
	}
}

double WellBoreLayerOnRandom::width() const {
	return m_width;
}

void WellBoreLayerOnRandom::setWidth(double val) {
	if (m_width!=val) {
		m_width = val;
		if (m_isShownLog) {
			m_scaleInit = false;
			refreshLog();
		}
	}
}

double WellBoreLayerOnRandom::logMin() const {
	return m_logMin;
}

void WellBoreLayerOnRandom::setLogMin(double val) {
	if (m_logMin!=val) {
		m_logMin = val;
		if (m_isShownLog) {
			m_scaleInit = false;
			refreshLog();
		}
	}
}

double WellBoreLayerOnRandom::logMax() const {
	return m_logMax;
}

void WellBoreLayerOnRandom::setLogMax(double val) {
	if (m_logMax!=val) {
		m_logMax = val;
		if (m_isShownLog) {
			m_scaleInit = false;
			refreshLog();
		}
	}
}

void WellBoreLayerOnRandom::setPenWidth(double val)
{
	if(m_item != nullptr)
	{
		QPen pen = m_item->pen();
		pen.setWidthF(val);
		m_item->setPen(pen);
	}
	if(m_itemLog != nullptr)
	{
		QPen pen = m_itemLog->pen();
		pen.setWidthF(val);
		m_itemLog->setPen(pen);
	}
}
