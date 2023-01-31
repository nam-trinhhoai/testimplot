#include "wellborelayeronmap.h"
#include "wellborereponmap.h"
#include "qglimagegriditem.h"
#include "wellbore.h"

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsItem>
#include <QGraphicsSimpleTextItem>
#include <QEvent>
#include <QScreen>
#include <cmath>

WellBoreLayerOnMap::WellBoreLayerOnMap(WellBoreRepOnMap *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem * parent) :
GraphicLayer(scene, defaultZDepth) {
	m_rep=rep;
	m_view = nullptr;

	WellBore* wellBore = dynamic_cast<WellBore*>(rep->data());

	connect(wellBore,SIGNAL(wellColorChanged(QColor)),this,SLOT(setColorWellChanged(QColor)));
	QPolygonF poly;
	const Deviations& deviations = wellBore->deviations();
	for (long i=0; i<deviations.xs.size(); i++) {
		double x = deviations.xs[i];
		double y = deviations.ys[i];
		poly << QPointF(x, y);
	}
	QPainterPath path;
	path.addPolygon(poly);

	m_item = new QGraphicsPathItem(path, parent);
	m_item->setZValue(defaultZDepth);
	QPen pen = m_item->pen();
	pen.setCosmetic(true);
	pen.setWidth(2);
	pen.setColor(wellBore->colorWell());// Qt::yellow);
	m_item->setPen(pen);
	m_item->setToolTip(wellBore->name());

//	m_textItem = new QGraphicsSimpleTextItem(wellBore->name(), parent);
//
//	QTransform invertYTransform(1, 0, 0, -1, 0, 0);
//	m_textItem->setTransform(invertYTransform);
//	m_textItem->setZValue(defaultZDepth);
//	QBrush brush = m_textItem->brush();
//	brush.setColor(Qt::green);
//	m_textItem->setBrush(brush);
//	QFont font = m_textItem->font();
//	font.setPixelSize(15);
//	font.setBold(true);
//	m_textItem->setFont(font);
//
//	m_textItem->setPos(poly.last());
	//m_textItem->setTransformOriginPoint(poly.last());
}

WellBoreLayerOnMap::~WellBoreLayerOnMap() {
}

void WellBoreLayerOnMap::setColorWellChanged(QColor c)
{
	if(m_item != nullptr)
	{
		QPen pen = m_item->pen();
		pen.setColor(c);
		m_item->setPen(pen);
	}
}

void WellBoreLayerOnMap::setWidth(double val)
{
	if(m_item != nullptr)
	{
		QPen pen = m_item->pen();
		pen.setWidthF(val);
		m_item->setPen(pen);
	}
}

void WellBoreLayerOnMap::show() {
	m_scene->addItem(m_item);
//	m_scene->addItem(m_textItem);

	QList<QGraphicsView*> views = m_scene->views();
	if (views.size()<1) {
		m_view = nullptr;
		return;
	}
	m_view = views[0];

	m_scaleInit = false;
	updateFromZoom();

	//m_view->installEventFilter(this);
}

void WellBoreLayerOnMap::hide() {
	if (m_view!=nullptr) {
//		m_view->removeEventFilter(this);
		m_view = nullptr;
	}


	m_scaleInit = false;
	m_scene->removeItem(m_item);
//	m_scene->removeItem(m_textItem);
}

QRectF WellBoreLayerOnMap::boundingRect() const
{
	QRectF rect = m_item->boundingRect(); // could use united with textItem boundingRect but need to add its pos
	// ex : textItem->boundingRect() : 0, 0, 100, 20
	// -> need to add textItem pos to QRectF to have right bounding box. (also scale ?)
	return rect;
}


void WellBoreLayerOnMap::refresh() {
	m_item->update();
//	m_textItem->update();
}

//bool WellBoreLayerOnMap::eventFilter(QObject* watched, QEvent* ev) {
//	if (ev->type() == QEvent::Wheel) {
//		updateFromZoom();
//	}
//	return false;
//}

void WellBoreLayerOnMap::updateFromZoom() {
	if (m_view==nullptr) {
		return;
	}
	return;

	QGraphicsView* view = m_view;

	QScreen* screen = view->screen();
	float inchSize = 0.15f;
	QPoint topLeft(0, 0);
	QPoint topRight(screen->logicalDotsPerInchX(), 0);
	QPoint bottomLeft(0, screen->logicalDotsPerInchY());

//	QSize viewSize = view->viewport()->size();

//	QPoint topLeft(0, 0), topRight(viewSize.width(), 0), bottomLeft(0, viewSize.height());
	QPointF topLeft1 = view->mapToScene(view->mapFromGlobal(topLeft));
	QPointF topRight1 = view->mapToScene(view->mapFromGlobal(topRight));
	QPointF bottomLeft1 = view->mapToScene(view->mapFromGlobal(bottomLeft));

	QPointF dWidthPt = (topRight1 - topLeft1);
	double dWidth = std::sqrt(std::pow(dWidthPt.x(),2) + std::pow(dWidthPt.y(), 2));
	QPointF dHeightPt = (bottomLeft1 - topLeft1);
	double dHeight = std::sqrt(std::pow(dHeightPt.x(),2) + std::pow(dHeightPt.y(), 2));

	QRectF rect = m_item->boundingRect();
	double penWidth = std::min(dWidth, dHeight) * inchSize / 5;
//
//	QPen pen = m_item->pen();
//	pen.setWidthF(penWidth);
//	m_item->setPen(pen);

	QRectF rectText = m_textItem->boundingRect();
	double svgScale = dHeight / rectText.height() * inchSize; //std::min(dWidth / rectText.width(), dHeight / rectText.height()) / 5;

	double scale = svgScale;
	if (!m_scaleInit || std::fabs(scale-m_cachedScale)>std::min(0.001, scale/100.0)) {
		m_textItem->setScale(scale);

		QPen pen = m_textItem->pen();
		pen.setWidthF(penWidth/2);
		m_textItem->setPen(pen);
	}
	m_cachedScale = scale;
	m_scaleInit = true;

//	refresh();
//	m_view->update();
}
