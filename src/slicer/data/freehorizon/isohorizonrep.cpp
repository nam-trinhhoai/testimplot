
#include <QAction>
#include <QMenu>

#include <workingsetmanager.h>
#include "isohorizonrep.h"
#include "isohorizon.h"

IsoHorizonRep::IsoHorizonRep(IsoHorizon *freehorizon, AbstractInnerView *parent) :
		AbstractGraphicRep(parent) {
	m_data = freehorizon;
	m_name = freehorizon->name();
}

IsoHorizonRep::~IsoHorizonRep() {

}

QWidget* IsoHorizonRep::propertyPanel() {
	return nullptr;
}

GraphicLayer * IsoHorizonRep::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) {
	return nullptr;
}

IData* IsoHorizonRep::data() const {
	return m_data;
}

AbstractGraphicRep::TypeRep IsoHorizonRep::getTypeGraphicRep() {
    return AbstractGraphicRep::NotDefined;
}

void IsoHorizonRep::buildContextMenu(QMenu *menu) {
	QAction *delete_ = new QAction(tr("delete"), this);
	menu->addAction(delete_);
	connect(delete_, SIGNAL(triggered()), this, SLOT(deleteHorizon()));
}

void IsoHorizonRep::deleteHorizon()
{
	// m_parent->hide();
	// emit deletedRep(this);
	WorkingSetManager *manager = m_data->workingSetManager();
	manager->removeIsoHorizons(m_data);
	this->deleteLater();
}
