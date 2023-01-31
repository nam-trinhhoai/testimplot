#include "wellpickreponrandom.h"
#include "wellpick.h"
#include "wellbore.h"
#include "wellhead.h"
#include "marker.h"
#include "wellpicklayeronrandom.h"
#include "randomlineview.h"
#include "affine2dtransformation.h"
#include "wellborereponrandom.h"
#include "geometry2dtoolbox.h"

#include "workingsetmanager.h"
#include <QMenu>
#include <QAction>

WellPickRepOnRandom::WellPickRepOnRandom(WellPick* wellPick, AbstractInnerView *parent) :
		AbstractGraphicRep(parent) {
	m_data = wellPick;
	m_layer = nullptr;

	RandomLineView* random = dynamic_cast<RandomLineView*>(parent);
	m_displayDistance = random->displayDistance();

	connect(random, &RandomLineView::displayDistanceChanged, this,
			&WellPickRepOnRandom::setDisplayDistance);
	connect(m_data->wellBore(), &WellBore::boreUpdated, this, &WellPickRepOnRandom::reExtractPosition);

	connect(m_data,&WellPick::deletedMenu,this,&WellPickRepOnRandom::deleteWellPickRepOnRandom);// MZR 18082021
}

WellPickRepOnRandom::~WellPickRepOnRandom() {
	if (m_layer!=nullptr) {
		delete m_layer;
	}
}

IData* WellPickRepOnRandom::data() const {
	return m_data;
}

QString WellPickRepOnRandom::name() const {
	return m_data->name();
}

WellPick* WellPickRepOnRandom::wellPick() const {
	return m_data;
}

bool WellPickRepOnRandom::isCurrentPointSet() const {
	return m_isPointSet;
}

const QList<QPointF>& WellPickRepOnRandom::getCurrentPointList(bool* ok) const {
	*ok = isCurrentPointSet();
	return m_points;
}

//AbstractGraphicRep
QWidget* WellPickRepOnRandom::propertyPanel() {
	return nullptr;
}

GraphicLayer * WellPickRepOnRandom::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) {
	if (m_layer==nullptr) {
		m_layer = new WellPickLayerOnRandom(this, scene, defaultZDepth, parent);
	}
	return m_layer;
}

bool WellPickRepOnRandom::setSampleUnit(SampleUnit unit) {
	bool isValid = false;
	m_points.clear();
	m_isPointSet = false;
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
			RandomLineView* randomView = dynamic_cast<RandomLineView*>(view());
			WellHead* wellHead =  m_data->wellBore()->wellHead();
			QPolygonF polyLine = randomView->polyLine();
			double distMin = std::numeric_limits<double>::max();
			QPointF bestPoint;
			for (std::size_t i=0; i<polyLine.size()-1; i++) {
				QPointF worldPos(x, y);
				std::tuple<long, double, bool> projectionDiscreate = randomView->getDiscreatePolyLineIndexFromWorldPos(worldPos);
				std::pair<QPointF, QPointF> segment(polyLine[i], polyLine[i+1]);
				bool ok;
				std::pair<double, QPointF> projectionReal = getPointProjectionOnSegment(worldPos, segment, &ok);

//				randomView->inlineXLineToXY()->worldToImage(x, y, imageX, imageY);
				//qDebug() << projectionReal.first << std::get<0>(projectionDiscreate) << std::get<1>(projectionDiscreate) << std::get<2>(projectionDiscreate);
				if (std::get<2>(projectionDiscreate) && ok && projectionReal.first<m_displayDistance) {
					//m_points.push_back(QPointF(std::get<0>(projectionDiscreate), depth));
					if (distMin>projectionReal.first) {
						bestPoint = QPointF(std::get<0>(projectionDiscreate), depth);
						distMin = projectionReal.first;
					}
				}
			}
			if (distMin!=std::numeric_limits<double>::max()) {
				m_points.push_back(bestPoint);
			}
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

QList<SampleUnit> WellPickRepOnRandom::getAvailableSampleUnits() const {
	QList<SampleUnit> list;
	if (m_data->wellBore()->isTfpDefined()) {
		list.push_back(SampleUnit::TIME);
	}
	list.push_back(SampleUnit::DEPTH);
	return list;
}

QString WellPickRepOnRandom::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	if (list.contains(sampleUnit)) {
		return "Failure to load supported unit";
	} else{
		return "Unknown unit";
	}
}

bool WellPickRepOnRandom::linkedRepShown() const {
	return m_linkedRepShown;
}

bool WellPickRepOnRandom::isLinkedRepValid() const {
	return m_linkedRep!=nullptr;
}

void WellPickRepOnRandom::wellBoreRepDeleted() {
	m_linkedRepShown = false;
	m_linkedRep = nullptr;
	updateLayer();
}

void WellPickRepOnRandom::wellBoreLayerChanged(bool toggle, WellBoreRepOnRandom* originObj) {
	if (originObj==m_linkedRep || m_linkedRep!=nullptr) {
		m_linkedRepShown = toggle;
		updateLayer();
	} else if (m_linkedRep==nullptr) {
		m_linkedRep = originObj;
		m_linkedRepShown = toggle;
		connect(m_linkedRep, &WellBoreRepOnRandom::destroyed, this, &WellPickRepOnRandom::wellBoreRepDeleted);
		updateLayer();
	}
}

void WellPickRepOnRandom::searchWellBoreRep() {
	// get well bore rep
	const QList<AbstractGraphicRep*>& reps = m_parent->getVisibleReps();
	std::size_t index = 0;
	while (index<reps.size() && m_linkedRep==nullptr) {
		WellBoreRepOnRandom* wellRep = dynamic_cast<WellBoreRepOnRandom*>(reps[index]);
		if (wellRep!=nullptr && wellRep->data()==m_data->wellBore()) {
			m_linkedRep = wellRep;
			connect(m_linkedRep, &WellBoreRepOnRandom::destroyed, this, &WellPickRepOnRandom::wellBoreRepDeleted);
		}
		index++;
	}

	if (m_linkedRep!=nullptr) {
		// is layer of well bore rep shown ?
		m_linkedRepShown = m_linkedRep->isLayerShown();
	}
}

void WellPickRepOnRandom::updateLayer() {
	if (m_layer!=nullptr) {
		m_layer->refresh();
	}
}

double WellPickRepOnRandom::displayDistance() const {
	return m_displayDistance;
}

void WellPickRepOnRandom::setDisplayDistance(double val) {
	if (m_displayDistance!=val) {
		m_displayDistance = val;
		setSampleUnit(m_sectionType);
	}
}

void WellPickRepOnRandom::buildContextMenu(QMenu *menu) {
	QAction *deleteAction = new QAction(tr("Delete Pick"), this);
	menu->addAction(deleteAction);
	connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteWellPickRepOnRandom()));
}

void WellPickRepOnRandom::deleteWellPickRepOnRandom(){
	m_parent->hideRep(this);
	emit deletedRep(this);

	disconnect(m_data,nullptr,this,nullptr);
	m_data->removeGraphicsRep();

	WorkingSetManager *manager = m_data->wellBore()->workingSetManager();
	manager->deleteMarker(m_data->currentMarker());

	this->deleteLater();
}

AbstractGraphicRep::TypeRep WellPickRepOnRandom::getTypeGraphicRep() {
	return Courbe;
}

void WellPickRepOnRandom::deleteLayer(){
    if (m_layer!=nullptr) {
        delete m_layer;
        m_layer = nullptr;
    }
}

void WellPickRepOnRandom::reExtractPosition() {
	setSampleUnit(m_sectionType);
	if (m_layer) {
		m_layer->refresh();
	}
}
