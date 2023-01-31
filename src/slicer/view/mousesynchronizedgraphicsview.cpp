#include "mousesynchronizedgraphicsview.h"
#include <QDebug>
#include "abstract2Dinnerview.h"
#include "graphicsutil.h"

MouseSynchronizedGraphicsView::MouseSynchronizedGraphicsView(WorkingSetManager *factory,ViewType viewType,
		QString uniqueName, QWidget *parent) :
		MonoTypeGraphicsView(factory,viewType, uniqueName, parent) {
}


MouseSynchronizedGraphicsView::~MouseSynchronizedGraphicsView() {

}

void MouseSynchronizedGraphicsView::viewPortChanged(const QPolygonF &poly) {
	QVector<AbstractInnerView*> views = innerViews();
	if (m_mutexView.tryLock()) {
	for (AbstractInnerView *v : views) {
		if (v == sender())
			continue;
		if (Abstract2DInnerView *view2D = dynamic_cast<Abstract2DInnerView*>(v)) {
			view2D->setViewRect(poly.boundingRect());
		}
	}
	m_mutexView.unlock();
	}
}
void MouseSynchronizedGraphicsView::registerView(AbstractInnerView *newView) {
	MonoTypeGraphicsView::registerView(newView);
	connect(newView, SIGNAL(viewAreaChanged(const QPolygonF & )), this,
			SLOT(viewPortChanged(const QPolygonF &)));
}

void MouseSynchronizedGraphicsView::unregisterView(AbstractInnerView *toBeDeleted)
{
	MonoTypeGraphicsView::unregisterView(toBeDeleted);
	disconnect(toBeDeleted, SIGNAL(viewAreaChanged(const QPolygonF & )), this,
				SLOT(viewPortChanged(const QPolygonF &)));
}

