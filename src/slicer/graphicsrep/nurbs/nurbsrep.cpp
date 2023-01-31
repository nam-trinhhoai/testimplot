#include "nurbsrep.h"
#include "nurbsdataset.h"
#include "nurbslayer.h"
#include "nurbswidget.h"

#include <QMenu>

NurbsRep::NurbsRep(NurbsDataset *nurbs, AbstractInnerView *parent) :
		AbstractGraphicRep(parent) {
	m_data = nurbs;
	m_name = nurbs->name();
	m_layer3D=nullptr;
}

NurbsRep::~NurbsRep() {
	if (m_layer3D != nullptr)
			delete m_layer3D;
}

QWidget* NurbsRep::propertyPanel() {
	return nullptr;
}


void NurbsRep::buildContextMenu(QMenu *menu) {

	QAction *editerNurbsAction = new QAction("Edit Nurbs", this);
	menu->addAction(editerNurbsAction);

	//QAction *removeNurbsAction = new QAction("Remove Nurbs", this);
	//menu->addAction(removeNurbsAction);

	connect(editerNurbsAction, SIGNAL(triggered()), this, SLOT(editerNurbs()));
	//connect(removeNurbsAction, SIGNAL(triggered()), this, SLOT(removeNurbs()));
}

GraphicLayer * NurbsRep::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) {
	return nullptr;
}

Graphic3DLayer* NurbsRep::layer3D(QWindow *parent, Qt3DCore::QEntity *root,Qt3DRender::QCamera *camera) {
	if (m_layer3D == nullptr) {
		m_layer3D = new NurbsLayer(this, parent, root, camera);
	}
	return m_layer3D;
}

IData* NurbsRep::data() const {
	return m_data;
}

AbstractGraphicRep::TypeRep NurbsRep::getTypeGraphicRep() {
    return AbstractGraphicRep::NotDefined;
}

/*
void NurbsRep::removeNurbs()
{
	NurbsWidget::removeNurbs("",m_name);
}*/

void NurbsRep::editerNurbs()
{
	NurbsWidget::showWidget();
	NurbsWidget::editerNurbs("",m_name);
}


