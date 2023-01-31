#include "wellborereponmap.h"
#include "wellbore.h"
#include "wellborelayeronmap.h"
#include "wellborelayer3d.h"
#include "abstractinnerview.h"
#include "wellpickrep.h"
#include "wellpick.h"
#include "workingsetmanager.h"
#include "abstract2Dinnerview.h"
#include "stackbasemapview.h"
#include "basemapview.h"

#include <QMenu>
#include <QAction>

WellBoreRepOnMap::WellBoreRepOnMap(WellBore *wellBore, AbstractInnerView *parent) :
	AbstractGraphicRep(parent) {
	m_data = wellBore;
	m_layer = nullptr;
	m_sampleUnit = SampleUnit::NONE;

	connect(m_data,&WellBore::deletedMenu,this,&WellBoreRepOnMap::deleteWellBoreRepOnMap); // MZR 18082021
}

WellBoreRepOnMap::~WellBoreRepOnMap() {
	if (m_layer != nullptr) {
		delete m_layer;
		disconnect(dynamic_cast<Abstract2DInnerView*>(m_parent), &Abstract2DInnerView::viewAreaChanged,
								m_layer, &WellBoreLayerOnMap::updateFromZoom);
	}
}

WellBore* WellBoreRepOnMap::wellBore() const {
	return m_data;
}

IData* WellBoreRepOnMap::data() const {
	return m_data;
}

QString WellBoreRepOnMap::name() const {
	return m_data->name();
}

QWidget* WellBoreRepOnMap::propertyPanel() {
	return nullptr;
}

WellBoreLayerOnMap* WellBoreRepOnMap::layer(){
	return m_layer;
}

GraphicLayer * WellBoreRepOnMap::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)
{
	if (m_layer == nullptr) {
		m_layer = new WellBoreLayerOnMap(this,scene,defaultZDepth,parent);

		StackBaseMapView* viewStacked = dynamic_cast<StackBaseMapView*>(m_parent);
		BaseMapView* viewMap = dynamic_cast<BaseMapView*>(m_parent);

		if (viewStacked) {
			connect(viewStacked, &StackBaseMapView::signalWellMapWidth, m_layer, &WellBoreLayerOnMap::setWidth);
			m_layer->setWidth(viewStacked->getWellMapWidth());
		} else if (viewMap) {
			connect(viewMap, &BaseMapView::signalWellMapWidth, m_layer, &WellBoreLayerOnMap::setWidth);
			m_layer->setWidth(viewMap->getWellMapWidth());
		}

		connect(dynamic_cast<Abstract2DInnerView*>(m_parent), &Abstract2DInnerView::viewAreaChanged,
										m_layer, &WellBoreLayerOnMap::updateFromZoom);
	}

	return m_layer;
}

Graphic3DLayer * WellBoreRepOnMap::layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera) {
	return nullptr;
}

SampleUnit WellBoreRepOnMap::sampleUnit() const {
	return m_sampleUnit;
}

bool WellBoreRepOnMap::setSampleUnit(SampleUnit type) {
	if (type==SampleUnit::TIME && !m_data->isWellCompatibleForTime(true)) {
		m_sampleUnit = SampleUnit::NONE;
		return false;
	} else {
		m_sampleUnit = type;
		return true;
	}
}

QList<SampleUnit> WellBoreRepOnMap::getAvailableSampleUnits() const {
	QList<SampleUnit> list;
	if (m_data->isTfpDefined()) {
		list.push_back(SampleUnit::TIME);
	}
	list.push_back(SampleUnit::DEPTH);
	return list;
}

QString WellBoreRepOnMap::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	if (list.contains(sampleUnit)) {
		return "Failure to load supported unit";
	} else{
		return "Unknown unit";
	}
}


// MZR 18082021
void WellBoreRepOnMap::buildContextMenu(QMenu *menu){
	QAction *deleteAction = new QAction(tr("Delete Wells 3"), this);
	menu->addAction(deleteAction);
	connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteWellBoreRepOnMap()));
}

void WellBoreRepOnMap::deleteWellBoreRepOnMap(){
	m_parent->hideRep(this);
	emit deletedRep(this);

	disconnect(m_data, nullptr, this, nullptr);
	m_data->deleteRep();

	if(m_layer != nullptr){
		m_layer->hide();
	}
	WorkingSetManager *manager = m_data->workingSetManager();
	manager->deleteWellHead(m_data->wellHead());

	this->deleteLater();
}

AbstractGraphicRep::TypeRep WellBoreRepOnMap::getTypeGraphicRep() {
    return AbstractGraphicRep::Courbe;
}
