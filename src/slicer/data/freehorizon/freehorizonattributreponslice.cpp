
/*
#include "wellpickreponslice.h"
#include "wellpick.h"
#include "wellbore.h"
#include "marker.h"
#include "wellpicklayeronslice.h"
#include "abstractsectionview.h"
#include "affine2dtransformation.h"
#include "wellborereponslice.h"
#include "workingsetmanager.h"
*/

#include <sliceutils.h>
#include "freehorizonattributreponslice.h"
#include "fixedlayerfromdatasetproppanel.h"
#include <freehorizonattribut.h>
#include <freehorizonproppanel.h>
#include <QMenu>
#include <QAction>

FreeHorizonAttributRepOnSlice::FreeHorizonAttributRepOnSlice(FreeHorizonAttribut* freehorizonattribut, SliceDirection dir, AbstractInnerView *parent) :
		AbstractGraphicRep(parent) {
	m_data = freehorizonattribut;
	m_name = "a";
	m_dir = dir;



	// m_layer = nullptr;

	/*
	AbstractSectionView* sectionView = dynamic_cast<AbstractSectionView*>(parent);
	m_displayDistance = sectionView->displayDistance();

	connect(sectionView, &AbstractSectionView::displayDistanceChanged, this,
			&WellPickRepOnSlice::setDisplayDistance);

	connect(m_data,&WellPick::deletedMenu,this,&WellPickRepOnSlice::deleteWellPickRepOnSlice); // MZR 18082021
	*/
}

FreeHorizonAttributRepOnSlice::~FreeHorizonAttributRepOnSlice() {
	/*
	if (m_layer!=nullptr) {
		delete m_layer;
	}
	*/
}

IData* FreeHorizonAttributRepOnSlice::data() const {
	return m_data;
}

QString FreeHorizonAttributRepOnSlice::name() const {
	return m_data->name();
}

QWidget* FreeHorizonAttributRepOnSlice::propertyPanel() {
	if (m_propPanel == nullptr) {
		m_propPanel = new FreeHorizonPropPanel(this, m_parent->viewType() == ViewType::View3D, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this](){
			m_propPanel = nullptr;
		});
	}
	return m_propPanel;

}

AbstractGraphicRep::TypeRep FreeHorizonAttributRepOnSlice::getTypeGraphicRep() {
	return Courbe;
}

/*
bool FreeHorizonAttributRepOnSlice::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> FreeHorizonAttributRepOnSlice::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_fixedLayer->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString FreeHorizonAttributRepOnSlice::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}
*/


/*
void FreeHorizonAttributRep::setSliceIJPosition(int val) {
	if (m_layer!=nullptr) {
		m_layer->reloadItems();
	}

}
*/


/*
bool WellPickRepOnSlice::isCurrentPointSet() const {
	return m_isPointSet;
}

QVector3D WellPickRepOnSlice::getCurrentPoint(bool* ok) const {
	*ok = isCurrentPointSet();
	return m_point;
}

//AbstractGraphicRep
QWidget* WellPickRepOnSlice::propertyPanel() {
	return nullptr;
}

GraphicLayer * WellPickRepOnSlice::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) {
	if (m_layer==nullptr) {
		m_layer = new WellPickLayerOnSlice(this, scene, defaultZDepth, parent);
	}
	return m_layer;
}


bool WellPickRepOnSlice::setSampleUnit(SampleUnit unit) {
	bool isValid = false;
	if (unit==SampleUnit::TIME || unit==SampleUnit::DEPTH) {
		WellUnit wellUnit = m_data->kindUnit();
		double value = m_data->value();
		double depth, x, y;
		WellBore* wellBore = m_data->wellBore();
		depth = wellBore->getDepthFromWellUnit(value, wellUnit, unit, &isValid);
		if (isValid) {
			x = wellBore->getXFromWellUnit(value, wellUnit, &isValid);
		}
		if (isValid) {
			y = wellBore->getYFromWellUnit(value, wellUnit, &isValid);
		}

		if (isValid) {
			m_sectionType = unit;
			AbstractSectionView* sectionView = dynamic_cast<AbstractSectionView*>(view());
			double imageX, imageY;
			sectionView->inlineXLineToXY()->worldToImage(x, y, imageX, imageY);
			m_point = QVector3D(imageX, imageY, depth);
			m_isPointSet = true;
		} else {
			setSampleUnit(SampleUnit::NONE);
		}
	} else {
		m_sectionType = SampleUnit::NONE;
		m_isPointSet = false;
		isValid = unit==SampleUnit::NONE;
	}
	return isValid;
}

QList<SampleUnit> WellPickRepOnSlice::getAvailableSampleUnits() const {
	QList<SampleUnit> list;
	if (m_data->wellBore()->isTfpDefined()) {
		list.push_back(SampleUnit::TIME);
	}
	list.push_back(SampleUnit::DEPTH);
	return list;
}

QString WellPickRepOnSlice::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	if (list.contains(sampleUnit)) {
		return "Failure to load supported unit";
	} else{
		return "Unknown unit";
	}
}

bool WellPickRepOnSlice::linkedRepShown() const {
	return m_linkedRepShown;
}

bool WellPickRepOnSlice::isLinkedRepValid() const {
	return m_linkedRep!=nullptr;
}

void WellPickRepOnSlice::wellBoreRepDeleted() {
	m_linkedRepShown = false;
	m_linkedRep = nullptr;
	setSliceIJPosition(0);
}

void WellPickRepOnSlice::wellBoreLayerChanged(bool toggle, WellBoreRepOnSlice* originObj) {
	if (originObj==m_linkedRep || m_linkedRep!=nullptr) {
		m_linkedRepShown = toggle;
		setSliceIJPosition(0);
	} else if (m_linkedRep==nullptr && originObj!=nullptr) {
		m_linkedRep = originObj;
		m_linkedRepShown = toggle;
		connect(m_linkedRep, &WellBoreRepOnSlice::destroyed, this, &WellPickRepOnSlice::wellBoreRepDeleted);
		setSliceIJPosition(0);
	}
}

void WellPickRepOnSlice::searchWellBoreRep() {
	// get well bore rep
	const QList<AbstractGraphicRep*>& reps = m_parent->getVisibleReps();
	std::size_t index = 0;
	while (index<reps.size() && m_linkedRep==nullptr) {
		WellBoreRepOnSlice* wellRep = dynamic_cast<WellBoreRepOnSlice*>(reps[index]);
		if (wellRep!=nullptr && wellRep->data()==m_data->wellBore()) {
			m_linkedRep = wellRep;
			connect(m_linkedRep, &WellBoreRepOnSlice::destroyed, this, &WellPickRepOnSlice::wellBoreRepDeleted);
		}
		index++;
	}

	if (m_linkedRep!=nullptr) {
		// is layer of well bore rep shown ?
		m_linkedRepShown = m_linkedRep->isLayerShown();
	}
}

double WellPickRepOnSlice::displayDistance() const {
	return m_displayDistance;
}

void WellPickRepOnSlice::setDisplayDistance(double val) {
	if (m_displayDistance!=val) {
		m_displayDistance = val;
		if (m_layer) {
			m_layer->reloadItems();
		}
	}
}

void WellPickRepOnSlice::buildContextMenu(QMenu *menu) {
	QAction *deleteAction = new QAction(tr("Delete Pick"), this);
	menu->addAction(deleteAction);
	connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteWellPickRepOnSlice()));
}

void WellPickRepOnSlice::deleteWellPickRepOnSlice(){
	m_parent->hideRep(this);
	emit deletedRep(this);

	disconnect(m_data,nullptr,this,nullptr);
	m_data->removeGraphicsRep();

	WorkingSetManager *manager = m_data->wellBore()->workingSetManager();
	manager->deleteMarker(m_data->currentMarker());

	this->deleteLater();
}

AbstractGraphicRep::TypeRep WellPickRepOnSlice::getTypeGraphicRep() {
	return Courbe;
}
*/


void FreeHorizonAttributRepOnSlice::buildContextMenu(QMenu *menu) {
	QAction *attribut = new QAction(tr("Compute attribut"), this);
	menu->addAction(attribut);
	QAction *delete_ = new QAction(tr("unselect"), this);
	menu->addAction(delete_);
	QAction *info = new QAction(tr("info"), this);
	menu->addAction(info);
	QAction *folder = new QAction(tr("folder"), this);
	menu->addAction(folder);

	// connect(attribut, SIGNAL(triggered()), this, SLOT(computeAttribut()));
	// connect(delete_, SIGNAL(triggered()), this, SLOT(deleteHorizon()));
	// connect(info, SIGNAL(triggered()), this, SLOT(infoHorizon()));
	// connect(folder, SIGNAL(triggered()), this, SLOT(folderHorizon()));
}
