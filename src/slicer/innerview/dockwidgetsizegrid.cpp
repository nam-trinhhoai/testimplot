#include "dockwidgetsizegrid.h"
#include <QDebug>

DockWidgetSizeGrid::DockWidgetSizeGrid(QWidget *parent) :QSizeGrip(parent){

}

QWidget *DockWidgetSizeGrid::sizegrip_topLevelWidget(QWidget* w)
{
    while (w && !w->isWindow() && w->windowType() != Qt::SubWindow)
        w = w->parentWidget();
    return w;
}

void DockWidgetSizeGrid::mouseMoveEvent(QMouseEvent * event)
{
	QSizeGrip::mouseMoveEvent(event);
	QWidget * parent=sizegrip_topLevelWidget(this);
	emit geometryChanged(parent->geometry());
}
DockWidgetSizeGrid::~DockWidgetSizeGrid() {
	// TODO Auto-generated destructor stub
}

