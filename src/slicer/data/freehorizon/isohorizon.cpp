
#include <QFileInfo>
#include "icontreewidgetitemdecorator.h"
#include "seismicsurvey.h"
#include <freeHorizonQManager.h>
#include "isohorizongraphicrepfactory.h"
// #include <isohorizonattribut.h>
#include "isohorizon.h"
// #include "markergraphicrepfactory.h"
// #include "wellpick.h"
// #include "wellbore.h"
// #include "seismic3dabstractdataset.h"

#include <fixedlayerimplisohorizonfromdatasetandcube.h>
#include <freeHorizonManager.h>
#include <fixedattributimplfreehorizonfromdirectories.h>
#include "fixedattributimplfromdirectories.h"
#include "stacksynchronizer.h"

IsoHorizon::IsoHorizon(WorkingSetManager * workingSet, SeismicSurvey *survey, const QString &path, const QString &name, QObject *parent) :
		IData(workingSet, parent), m_name(name) {
	m_uuid = QUuid::createUuid();
	m_path = path;
	bool colorOk = false;
	QColor color = FreeHorizonQManager::loadColorFromPath(m_path, &colorOk);
	if ( colorOk )
	{
		m_color = color;
	}
	else
	{
		m_color = Qt::white;
	}
	FreeHorizonManager::PARAM horizonParam = FreeHorizonManager::dataSetGetParam(m_path.toStdString()+"/iso_00000/"+FreeHorizonManager::isoDataName);
	if (horizonParam.axis==inri::Xt::Time)
	{
		m_sampleUnit = SampleUnit::TIME;
	}
	else if (horizonParam.axis==inri::Xt::Depth)
	{
		m_sampleUnit = SampleUnit::DEPTH;
	}
	else
	{
		m_sampleUnit = SampleUnit::NONE;
	}
	m_parent = parent;
	m_workingSet = workingSet;
	m_decorator = nullptr;
	m_repFactory = new IsoHorizonGraphicRepFactory(this);
	m_attribut.clear();
	m_survey = survey;

	m_synchronizer = new StackSynchronizer(StackType::ISO, this);
	horizonAttributCreate();

}

IsoHorizon::~IsoHorizon() {
	if (!m_isoData.isNull()) {
		disconnect(m_isoData.data(), &FixedLayerImplIsoHorizonFromDatasetAndCube::colorChanged,
								this, &IsoHorizon::setColor);
	}
}


void IsoHorizon::horizonAttributCreate()
{
	QString path0 = m_path + "/iso_00000/";
	QString isoPath = path0 + "/" + QString::fromStdString(FreeHorizonManager::isoDataName);
	std::vector<QString> list = FreeHorizonQManager::getAttributData(path0);
	m_attribut.resize(list.size());
	for (int n=0; n<list.size(); n++)
	{
		QFileInfo fi(list[n]);
		QString name = fi.completeBaseName();
		QString type = QString::fromStdString(FreeHorizonManager::typeFromAttributName(name.toStdString()));
		// m_attribut[n] = new FreeHorizonAttribut(m_workingSet, m_survey, m_path, name, m_parent);


		bool isValid = false;
		// QString path0 = m_path + "iso_00000/";
		std::vector<QString> names;
		names.resize(1, "a");

		if ( type == "spectrum")
		{
			FixedRGBLayersFromDatasetAndCube::Grid3DParameter params = FixedRGBLayersFromDatasetAndCube::createGrid3DParameterFromHorizon(isoPath, m_survey, &isValid);
			m_attribut[n].pFixedRGBLayersFromDatasetAndCube = new FixedAttributImplFromDirectories(m_path, name, name, m_workingSet, params, m_parent);
		}
		else
		{
			FixedLayersFromDatasetAndCube::Grid3DParameter params = FixedLayersFromDatasetAndCube::createGrid3DParameterFromHorizon(isoPath, m_survey, &isValid);
			m_attribut[n].pFixedLayerImplIsoHorizonFromDatasetAndCube = new FixedLayerImplIsoHorizonFromDatasetAndCube(m_path, name, name, m_workingSet, params, m_parent);
		}

		if (m_isoData.isNull() && type == "isochrone") {
			m_isoData = m_attribut[n].pFixedLayerImplIsoHorizonFromDatasetAndCube;
			QString toolTip = this->name();
			QStringList parenthesisSplit = toolTip.split("(");
			if (parenthesisSplit.size()>1) {
				toolTip = parenthesisSplit[0];
			}
			m_isoData->setSectionToolTip(toolTip);
			connect(m_isoData.data(), &FixedLayerImplIsoHorizonFromDatasetAndCube::colorChanged,
					this, &IsoHorizon::setColor);
		}

		StackableData* stackInterface = dynamic_cast<StackableData*>(m_attribut[n].data());
		if (stackInterface) {
			m_synchronizer->addData(stackInterface);
		}
	}
}

	//IData
 IGraphicRepFactory *IsoHorizon::graphicRepFactory() {
	return m_repFactory;
}


QUuid IsoHorizon::dataID() const {
	return m_uuid;
}

QColor IsoHorizon::color() const {
	return m_color;
}

void IsoHorizon::setColor(const QColor& color) {
	if (m_color!=color) {
		m_color = color;
		emit colorChanged(m_color);
	}
}

ITreeWidgetItemDecorator* IsoHorizon::getTreeWidgetItemDecorator()
{
	if (m_decorator==nullptr)
	{
		QIcon icon = FreeHorizonQManager::getHorizonIcon(m_color, m_sampleUnit);
		m_decorator = new IconTreeWidgetItemDecorator(icon, this);
		connect(this, &IsoHorizon::iconChanged, m_decorator, &IconTreeWidgetItemDecorator::setIcon);
		connect(this, &IsoHorizon::colorChanged, this, &IsoHorizon::updateIcon);
	}
	return m_decorator;
}

IData* IsoHorizon::getIsochronData() {
	return m_isoData.data();
}

void IsoHorizon::updateIcon(QColor color)
{
	QIcon icon = FreeHorizonQManager::getHorizonIcon(color, m_sampleUnit);
	emit iconChanged(icon);
}

/*
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
*/

IData* IsoHorizon::Attribut::data() {
	IData* ptr = nullptr;
	if (pFixedRGBLayersFromDatasetAndCube!=nullptr) {
		ptr = pFixedRGBLayersFromDatasetAndCube;
	} else if (pFixedLayerImplIsoHorizonFromDatasetAndCube!=nullptr) {
		ptr = pFixedLayerImplIsoHorizonFromDatasetAndCube;
	}
}
