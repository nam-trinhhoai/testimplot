#include "random3dtexrep.h"
#include "randomtexdataset.h"
#include "random3dtexlayer.h"

Random3dTexRep::Random3dTexRep(RandomTexDataset *random, AbstractInnerView *parent) :
		AbstractGraphicRep(parent) {
	m_data = random;
	m_name = random->name();
	m_layer3D=nullptr;
}

Random3dTexRep::~Random3dTexRep() {
	if (m_layer3D != nullptr)
			delete m_layer3D;
}

QWidget* Random3dTexRep::propertyPanel() {
	return nullptr;
}

GraphicLayer * Random3dTexRep::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) {
	return nullptr;
}

Graphic3DLayer* Random3dTexRep::layer3D(QWindow *parent, Qt3DCore::QEntity *root,Qt3DRender::QCamera *camera) {
	if (m_layer3D == nullptr) {
		m_layer3D = new Random3dTexLayer(this, parent, root, camera);
	}
	return m_layer3D;
}

IData* Random3dTexRep::data() const {
	return m_data;
}

AbstractGraphicRep::TypeRep Random3dTexRep::getTypeGraphicRep() {
    return AbstractGraphicRep::NotDefined;
}
