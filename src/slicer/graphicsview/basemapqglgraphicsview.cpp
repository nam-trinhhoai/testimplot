#include "basemapqglgraphicsview.h"
#include "globalconfig.h"
#include "GraphicSceneEditor.h"
#include <QMouseEvent>
//#include <QGLWidget>
#include <iostream>
#include <QOpenGLContext>
//#include <QGLWidget>
#include <QOpenGLWidget>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsRectItem>
#include <cmath>
#include <QApplication>
#include <QDebug>

/*
class testItem: public QGraphicsObject
{
public:
	QGraphicsRectItem * m_item;

	testItem(qreal x, qreal y, qreal width, qreal height)
	{
		m_item = new QGraphicsRectItem(x,y,width,height,this);
		QPen pen(Qt::red);
		pen.setWidth(5);
		pen.setCosmetic(true);
		m_item->setPen(pen);
		m_item->setZValue(1000);
		m_item->setFlag(QGraphicsItem::ItemIsMovable, true);

	}

	void wheelEvent(QGraphicsSceneWheelEvent *event)
	{
		QRectF rect = m_item->mapToParent(m_item->boundingRect()).boundingRect();
		QRectF rect2 = m_item->mapToScene(m_item->boundingRect()).boundingRect();
		qDebug()<<" wheel event parent"<<rect;
		qDebug()<<" wheel event scene"<<rect2;
	}

	QRectF boundingRect()const
	{
		return m_item->mapToParent(m_item->boundingRect()).boundingRect(); //m_item->boundingRect();
	}

void paint(QPainter* painter,const QStyleOptionGraphicsItem* option, QWidget *widget)
{

}
};*/

BaseMapQGLGraphicsView::BaseMapQGLGraphicsView(QWidget *parent) :
		BaseQGLGraphicsView(parent) {
	scale(1, -1);


}

void BaseMapQGLGraphicsView::wheelEvent(QWheelEvent *e) {
	if ((!scene()->selectedItems().isEmpty()) &&  (e->modifiers() & Qt::ShiftModifier) )
	{
		QPoint numDegrees = e->angleDelta() / 8;
		if (!numDegrees.isNull())
		{
			if (numDegrees.y() >= 15)
			{
				dynamic_cast<GraphicSceneEditor*>(scene())->rotate(5);
			}
			else if (numDegrees.y() <= -15)
			{
				dynamic_cast<GraphicSceneEditor*>(scene())->rotate(-5);
			}
		}
	}
	else
	{
		if (zoomLocked())
			return;

		if (!scene() || !isInteractive()) {
			QAbstractScrollArea::wheelEvent(e);
			return;
		}
		e->ignore();
		QGraphicsSceneWheelEvent wheelEvent(QEvent::GraphicsSceneWheel);
		wheelEvent.setWidget(viewport());
		wheelEvent.setScenePos(mapToScene(e->position().toPoint()));
		wheelEvent.setScreenPos(e->globalPosition().toPoint());
		wheelEvent.setButtons(e->buttons());
		wheelEvent.setModifiers(e->modifiers());
		const bool horizontal = qAbs(e->angleDelta().x()) > qAbs(e->angleDelta().y());
		wheelEvent.setDelta(horizontal ? e->angleDelta().x() : e->angleDelta().y());
		wheelEvent.setOrientation(horizontal ? Qt::Horizontal : Qt::Vertical);
		wheelEvent.setAccepted(false);
		QCoreApplication::sendEvent(scene(), &wheelEvent);
		e->setAccepted(wheelEvent.isAccepted());


		GlobalConfig& config = GlobalConfig::getConfig();
		if(!e->isAccepted())
		{
			int delta = e->angleDelta().y();//original e->delta()
			double factor = std::pow(config.wheelZoomBaseFactor(),
					config.wheelZoomExponentFactor() * delta);
			setTransformationAnchor(QGraphicsView::AnchorUnderMouse); //scale around the mouse position
			applyScale(factor, factor);
			e->accept();
		}
	}
}
std::pair<float, float> BaseMapQGLGraphicsView::resetZoom(void) {
	return setVisibleRect(scene()->itemsBoundingRect());
}

std::pair<float, float> BaseMapQGLGraphicsView::setVisibleRect(const QRectF &bbox) {
	// Reset the view scale to 1:1.
	resetScale(this);

	//Compute Aspect Ration
	QRectF viewRect = viewport()->rect();
	if (viewRect.isEmpty())
		return std::pair<float, float>(1, 1);
	QRectF sceneRect = transform().mapRect(bbox);
	if (sceneRect.isEmpty())
		return std::pair<float, float>(1, 1);
	qreal xratio = viewRect.width() / sceneRect.width();
	qreal yratio = viewRect.height() / sceneRect.height();

	qreal r = qMin(xratio, yratio);
	scale(r, r);

	centerOn(bbox.center());
	return std::pair<float, float>(r, r);
}
