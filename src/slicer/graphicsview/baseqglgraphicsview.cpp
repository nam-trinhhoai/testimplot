#include "baseqglgraphicsview.h"
#include "ipopmenu.h"

#include <QOpenGLWidget>
//#include <QGLWidget>
#include <QGraphicsItem>
#include <QDebug>

AbstractOverlayPainter::~AbstractOverlayPainter(){
}

BaseQGLGraphicsView::BaseQGLGraphicsView(QWidget *parent) :
		QGraphicsView(parent) {
	QOpenGLWidget* glwidget = new QOpenGLWidget();
	QSurfaceFormat format = glwidget->format();
	format.setStencilBufferSize(8);
	glwidget->setFormat(format);
	setViewport(glwidget);

	setRenderHint(QPainter::Antialiasing, true);
	setOptimizationFlags(DontSavePainterState | DontAdjustForAntialiasing);
	setViewportUpdateMode(SmartViewportUpdate);
	setTransformationAnchor(AnchorUnderMouse);
	setDragMode(QGraphicsView::ScrollHandDrag);

	setBackgroundBrush(QBrush("#19232D"));
	m_zoomLocked=false;
}
void BaseQGLGraphicsView::lockZoom(bool lock)
{
	m_zoomLocked=lock;
}
void BaseQGLGraphicsView::applyScale(double sx, double sy)
{
	scale(sx,sy);
	emit scaleChanged(sx,sy);
}

std::pair<float, float> BaseQGLGraphicsView::resetZoom(void) {
	return setVisibleRect(scene()->itemsBoundingRect());
}

std::pair<float, float> BaseQGLGraphicsView::setVisibleRect(const QRectF &bbox) {
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

	//qreal r = qMin(xratio, yratio);
	//scale(r, r);
	scale(xratio, yratio);
	centerOn(bbox.center());
	//return r;
	return std::pair<float, float>(xratio, yratio);
}


void BaseQGLGraphicsView::mouseMoveEvent(QMouseEvent *e) {
	if(m_zoomLocked)return;

	if (!(e->modifiers() & Qt::ControlModifier))
		QGraphicsView::mouseMoveEvent(e);

	QPointF scenePos = this->mapToScene(e->pos());
	emit mouseMoved(scenePos.x(), scenePos.y(), e->button(), e->modifiers());
}
void BaseQGLGraphicsView::mousePressEvent(QMouseEvent *e) {
	if(m_zoomLocked)return;
//	e->ignore();

	if (!(e->modifiers() & Qt::ControlModifier)) {
		QGraphicsView::mousePressEvent(e);
		if(e->isAccepted()) {
			return;
		}
	}

	QPointF scenePos = this->mapToScene(e->pos());

	emit mousePressed(scenePos.x(), scenePos.y(), e->button(), e->modifiers());

	//e->accept();
}

void BaseQGLGraphicsView::mouseReleaseEvent(QMouseEvent *e) {
	if(m_zoomLocked)return;
	if (!(e->modifiers() & Qt::ControlModifier))
		QGraphicsView::mouseReleaseEvent(e);

	QPointF scenePos = this->mapToScene(e->pos());
	emit mouseRelease(scenePos.x(), scenePos.y(), e->button(), e->modifiers());
}

void BaseQGLGraphicsView::mouseDoubleClickEvent(QMouseEvent *e) {
        if(m_zoomLocked)return;
        if (!(e->modifiers() & Qt::ControlModifier))
                QGraphicsView::mouseDoubleClickEvent(e);

        QPointF scenePos = this->mapToScene(e->pos());
        emit mouseDoubleClick(scenePos.x(), scenePos.y(), e->button(), e->modifiers());
}

void BaseQGLGraphicsView::contextMenuEvent(QContextMenuEvent *event) {
	QMenu mainMenu;
	QPointF scenePos = this->mapToScene(event->pos());

	// search in scene to fill menu
	std::size_t previousColumnCount = 0;

	QGraphicsItem* mouseGrabber = scene()->mouseGrabberItem();
	if (mouseGrabber) {
		IPopMenu* interface = dynamic_cast<IPopMenu*>(mouseGrabber);
		if (interface!=nullptr) {
			interface->fillContextMenu(scenePos, event->reason(), mainMenu);

			// add separator between items menus
			if (previousColumnCount!=mainMenu.actions().count()) {
				mainMenu.addSeparator();
				previousColumnCount = mainMenu.actions().count();
			}
		}
	}

	QList<QGraphicsItem*> items = scene()->items(scenePos);
	for (QGraphicsItem* item : items) {
		if (item==mouseGrabber) {
			// avoid to have the same item do the menu twice
			continue;
		}
		IPopMenu* interface = dynamic_cast<IPopMenu*>(item);
		if (interface!=nullptr) {
			interface->fillContextMenu(scenePos, event->reason(), mainMenu);

			// add separator between items menus
			if (previousColumnCount!=mainMenu.actions().count()) {
				mainMenu.addSeparator();
				previousColumnCount = mainMenu.actions().count();
			}
		}
	}

	// propagate menu
	emit contextMenu(scenePos.x(), scenePos.y(), event->reason(), mainMenu);

	// only show if not empty
	if (mainMenu.actions().count()>0) {
		mainMenu.exec(event->globalPos());
	}

	// Do not call QGraphicsView::contextMenuEvent, because we override the event
}

void BaseQGLGraphicsView::resetScale(QGraphicsView *view)
{
	QRectF unity = view->transform().mapRect(QRectF(0, 0, 1, 1));
	if (unity.isEmpty())
		return;
	view->scale(1 / unity.width(), 1 / unity.height());
}

void BaseQGLGraphicsView::addOverlayPainter(AbstractOverlayPainter* painter){
	m_overlayPainters.removeOne(painter);
	m_overlayPainters.append(painter);
	update();
}

void BaseQGLGraphicsView::removeOverlayPainter(AbstractOverlayPainter* painter){
	m_overlayPainters.removeOne(painter);
	update();
}

void BaseQGLGraphicsView::drawForeground(QPainter* painter, const QRectF& rec) {
	QRectF viewportRect = this->getViewportRect();

	//qDebug() << "Painting overlay extensions";

	painter->save();
	//painter->resetMatrix();
	painter->resetTransform();
	QRect visibleRect = this->viewport()->visibleRegion().boundingRect();
	for(auto op : m_overlayPainters){
		op->paintOverlay(painter,visibleRect);
	}
	painter->restore();
}

QRectF BaseQGLGraphicsView::getViewportRect() const {
	QPointF p1 = this->mapToScene(0, 0);
	QPointF p2 = this->mapToScene(this->width(), this->height());

	double xmin = std::min(p1.x(), p2.x());
	double xmax = std::max(p1.x(), p2.x());
	double ymin = std::min(p1.y(), p2.y());
	double ymax = std::max(p1.y(), p2.y());

	QRectF res = QRectF(QPointF(xmin, ymin), QPointF(xmax, ymax));

	return res;
}
