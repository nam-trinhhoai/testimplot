#include "marker.h"
#include "markergraphicrepfactory.h"

#include "wellpick.h"
#include "wellbore.h"
#include "seismic3dabstractdataset.h"

Marker::Marker(WorkingSetManager * workingSet,const QString &name, QObject *parent) :
		IData(workingSet, parent), m_name(name) {
	m_uuid = QUuid::createUuid();
	m_repFactory = new MarkerGraphicRepFactory(this);
}

Marker::~Marker() {
	for (long i=m_wellPicks.size()-1; i>=0; i--) {
		WellPick* pick = m_wellPicks[i];
		// remove from well bore
		pick->wellBore()->removePick(pick);
		// remove from this
		removeWellPick(pick);
		// delete data
		delete pick;
	}
}

	//IData
IGraphicRepFactory *Marker::graphicRepFactory() {
	return m_repFactory;
}

QUuid Marker::dataID() const {
	return m_uuid;
}

QColor Marker::color() const {
	return m_color;
}

void Marker::setColor(const QColor& color) {
	m_color = color;
}

QList<RgtSeed> Marker::getProjectedPicksOnDataset(Seismic3DAbstractDataset* dataset, int channel, SampleUnit sampleUnit) {
	QList<RgtSeed> outList;
	for (WellPick* pick : m_wellPicks) {
		std::pair<RgtSeed, bool> projection = pick->getProjectionOnDataset(dataset, channel, sampleUnit);
		if (projection.second) {
			outList.push_back(projection.first);
		}
	}
	return outList;
}

QList<WellPick*> Marker::getWellPickFromWell(WellBore* bore) {
	QList<WellPick*> out;
	for (std::size_t i=0; i<m_wellPicks.size(); i++) {
		if (m_wellPicks[i]->wellBore()==bore) {
			out.push_back(m_wellPicks[i]);
		}
	}
	return out;
}

const QList<WellPick*>& Marker::wellPicks() const {
	return m_wellPicks;
}

void Marker::addWellPick(WellPick* pick) {
	m_wellPicks.push_back(pick);
	emit wellPickAdded(pick);
}

void Marker::removeWellPick(WellPick* pick) {
	m_wellPicks.removeOne(pick);
	emit wellPickRemoved(pick);
}
