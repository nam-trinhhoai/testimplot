#include "monotypegraphicsview.h"
#include <QPushButton>
#include <QToolBar>

#include "graphicsutil.h"

MonoTypeGraphicsView::MonoTypeGraphicsView(WorkingSetManager *factory,ViewType viewType, QString uniqueName,
		QWidget *parent) : AbstractGraphicsView(factory, uniqueName, parent) {
	m_viewType=viewType;
	QPushButton *addInternalView = GraphicsUtil::generateToobarButton(
				":/slicer/icons/add.png", "Add view", toolBar());
	connect(addInternalView, SIGNAL(clicked()), this, SLOT(addView()));
	addInternalView->setDefault(false);
	addInternalView->setAutoDefault(false);
	m_toolbar->addWidget(addInternalView);
}

void MonoTypeGraphicsView::addView() {
	registerView(generateView(m_viewType,true));
}


MonoTypeGraphicsView::~MonoTypeGraphicsView() {

}
