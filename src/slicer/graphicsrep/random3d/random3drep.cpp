#include "random3drep.h"
#include "randomdataset.h"
#include "random3dlayer.h"

Random3dRep::Random3dRep(RandomDataset *random, AbstractInnerView *parent) :
		AbstractGraphicRep(parent) {
	m_data = random;
	m_name = random->name();
	m_layer3D=nullptr;
}

Random3dRep::~Random3dRep() {
	if (m_layer3D != nullptr)
			delete m_layer3D;
}

QWidget* Random3dRep::propertyPanel() {
	return nullptr;
}

GraphicLayer * Random3dRep::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) {
	return nullptr;
}

Graphic3DLayer* Random3dRep::layer3D(QWindow *parent, Qt3DCore::QEntity *root,Qt3DRender::QCamera *camera) {
	if (m_layer3D == nullptr) {
		m_layer3D = new Random3dLayer(this, parent, root, camera);
	}
	return m_layer3D;
}

IData* Random3dRep::data() const {
	return m_data;
}

AbstractGraphicRep::TypeRep Random3dRep::getTypeGraphicRep() {
    return AbstractGraphicRep::NotDefined;
}
