#include "basemapgraphicsview.h"
#include "basemapview.h"
#include <QDebug>

BaseMapGraphicsView::BaseMapGraphicsView(WorkingSetManager *factory,
		QString uniqueName, QWidget *parent) :
		MouseSynchronizedGraphicsView(factory,ViewType::BasemapView, 
		uniqueName, parent) {
	registerView(generateView(ViewType::BasemapView,true));
}

BaseMapGraphicsView::~BaseMapGraphicsView() {

}
