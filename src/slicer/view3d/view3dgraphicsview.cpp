#include "view3dgraphicsview.h"
#include "viewqt3d.h"
#include <QPushButton>
#include <QHBoxLayout>
#include <QSpinBox>
#include <QLabel>
#include <QToolBar>
#include "graphicsutil.h"


View3DGraphicsView::View3DGraphicsView(
		WorkingSetManager *factory, QString uniqueName, QWidget *parent) :
		MonoTypeGraphicsView(factory,ViewType::View3D, uniqueName, parent) {
	registerView(generateView(ViewType::View3D,true));
	toolBar()->addWidget(createZScaleWidget());
}

void View3DGraphicsView::zScaleChanged(int val) {
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views) {
		if (ViewQt3D *s = dynamic_cast<ViewQt3D*>(v)) {
			s->zScaleChanged(val);
		}
	}
}

QWidget* View3DGraphicsView::createZScaleWidget() {
	m_zscale = new QSpinBox();
	m_zscale->setMinimum(100);
	m_zscale->setMaximum(10000);
	m_zscale->setSingleStep(1);
	m_zscale->setValue(100);

	m_zscale->setWrapping(false);
	connect(m_zscale, SIGNAL(valueChanged(int)), this,
			SLOT(zScaleChanged(int)));

	QWidget *zscaleWidget = new QWidget(this);
	zscaleWidget->setContentsMargins(QMargins(0, 0, 0, 0));
	QHBoxLayout *layout = new QHBoxLayout(zscaleWidget);
	layout->setContentsMargins(0,0,0,0);
	layout->addWidget(new QLabel("Z Exageration"));
	layout->addWidget(m_zscale);
	layout->addWidget(new QLabel("%"));
	return zscaleWidget;
}

View3DGraphicsView::~View3DGraphicsView() {

}

