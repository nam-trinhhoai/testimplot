#include "importsismagehorizonwithworkingsetwidget.h"

#include "folderdata.h"
#include "importsismagehorizonwidget.h"
#include "seismic3dabstractdataset.h"
#include "seismicsurvey.h"
#include "workingsetmanager.h"

#include "fixedrgblayersfromdatasetandcube.h"

#include <freehorizon.h>
#include <freeHorizonManager.h>

#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QVBoxLayout>
#include <freeHorizonQManager.h>

ImportSismageHorizonWithWorkingSetWidget::ImportSismageHorizonWithWorkingSetWidget(
		WorkingSetManager* manager, QWidget* parent) : QWidget(parent) {
	setWindowTitle("Horizon Sismage Import");
	setAttribute(Qt::WA_DeleteOnClose);

	m_manager = manager;
	m_currentDataset = nullptr;

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	QHBoxLayout* comboLayout = new QHBoxLayout;
	mainLayout->addLayout(comboLayout);
	comboLayout->addWidget(new QLabel("Dataset"));
	m_datasetComboBox = new QComboBox;
	comboLayout->addWidget(m_datasetComboBox);

	m_importerWidget = new ImportSismageHorizonWidget;
	mainLayout->addWidget(m_importerWidget);

	connect(m_datasetComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ImportSismageHorizonWithWorkingSetWidget::currentDatasetChanged);
	connect(m_manager->folders().seismics, &FolderData::dataAdded, this, &ImportSismageHorizonWithWorkingSetWidget::potentialSurveyAdded);
	connect(m_manager->folders().seismics, &FolderData::dataRemoved, this, &ImportSismageHorizonWithWorkingSetWidget::potentialSurveyRemoved);
	connect(m_importerWidget, &ImportSismageHorizonWidget::horizonExtracted, this, &ImportSismageHorizonWithWorkingSetWidget::relayHorizonExtracted);
	connect(m_importerWidget, &ImportSismageHorizonWidget::importFinished, this, &ImportSismageHorizonWithWorkingSetWidget::relayImportFinished);

	updateComboBox();
}

ImportSismageHorizonWithWorkingSetWidget::~ImportSismageHorizonWithWorkingSetWidget() {
	for (auto it=m_connections.begin(); it!=m_connections.end(); it++) {
		disconnect(it->second.addConnection);
		disconnect(it->second.removeConnection);
	}
}

void ImportSismageHorizonWithWorkingSetWidget::currentDatasetChanged(int index) {
	if (index<0 || index>=m_datasetComboBox->count()) {
		m_currentDataset = nullptr;
		m_importerWidget->setDatasetPath("");
		return;
	}

	QVariant var = m_datasetComboBox->itemData(index);
	if (!var.canConvert<Seismic3DAbstractDataset*>()) {
		m_currentDataset = nullptr;
		m_importerWidget->setDatasetPath("");
	} else {
		Seismic3DAbstractDataset* dataset = qvariant_cast<Seismic3DAbstractDataset*>(var);
		QString datasetPath = dataset->idPath();

		m_currentDataset = dataset;
		m_importerWidget->setDatasetPath(datasetPath);
	}
}

void ImportSismageHorizonWithWorkingSetWidget::datasetAdded(Seismic3DAbstractDataset* dataset) {
	updateComboBox();
}

void ImportSismageHorizonWithWorkingSetWidget::datasetRemoved(Seismic3DAbstractDataset* dataset) {
	updateComboBox();
}

void ImportSismageHorizonWithWorkingSetWidget::potentialSurveyAdded(IData* data) {
	SeismicSurvey* survey = dynamic_cast<SeismicSurvey*>(data);
	if (survey==nullptr) {
		return;
	}

	SurveyConnections conns;
	conns.addConnection = connect(survey, &SeismicSurvey::datasetAdded, this, &ImportSismageHorizonWithWorkingSetWidget::datasetAdded);
	conns.removeConnection = connect(survey, &SeismicSurvey::datasetRemoved, this, &ImportSismageHorizonWithWorkingSetWidget::datasetRemoved);
	m_connections[survey] = conns;

	updateComboBox();
}

void ImportSismageHorizonWithWorkingSetWidget::potentialSurveyRemoved(IData* data) {
	SeismicSurvey* survey = dynamic_cast<SeismicSurvey*>(data);
	if (survey==nullptr) {
		return;
	}

	auto it = m_connections.find(survey);
	if (it!=m_connections.end()) {
		disconnect(it->second.addConnection);
		disconnect(it->second.removeConnection);
	}
}

void ImportSismageHorizonWithWorkingSetWidget::relayHorizonExtracted(QString horizonName, QString datasetPath) {
	emit horizonExtracted(horizonName, datasetPath);
}

void ImportSismageHorizonWithWorkingSetWidget::relayImportFinished() {

	updateHorizonsInTree();
	emit importFinished();
}


void ImportSismageHorizonWithWorkingSetWidget::updateComboBox(bool force) {
	if (!force && !isVisible()) {
		return;
	}


	QList<Seismic3DAbstractDataset*> foundDatasets;

	long numberOfSurveys = 0;
	QList<IData*> datas = m_manager->folders().seismics->data();
	for (long i=0; i<datas.size(); i++) {
		SeismicSurvey* survey = dynamic_cast<SeismicSurvey*>(datas[i]);
		if (survey) {
			QList<Seismic3DAbstractDataset*> datasets = survey->datasets();
			if (datasets.size()>0) {
				numberOfSurveys++;
			}

			for (long j=0; j<datasets.size(); j++) {
				foundDatasets.append(datasets[j]);
			}
		}
	}

	if (m_currentDataset!=nullptr) {
		int index = foundDatasets.indexOf(m_currentDataset);
		if (index<0 || index>=foundDatasets.size()) {
			m_currentDataset = nullptr;
		}
	}

	{
		// avoid unneeded calls to currentDatasetChanged
		QSignalBlocker b(m_datasetComboBox);

		m_datasetComboBox->clear();
		long currentDatasetIndex = -1;
		for (long i=0; i<foundDatasets.size(); i++) {
			Seismic3DAbstractDataset* dataset = foundDatasets[i];
			QString name = dataset->name();
			if (numberOfSurveys>1) {
				name = dataset->survey()->name() + " : " + name;
			}
			QVariant datasetVar;
			datasetVar.setValue(dataset);
			m_datasetComboBox->addItem(name, datasetVar);

			if (dataset==m_currentDataset) {
				currentDatasetIndex = i;
			}
		}
		if (currentDatasetIndex>=0 && currentDatasetIndex<m_datasetComboBox->count()) {
			m_datasetComboBox->setCurrentIndex(currentDatasetIndex);
		}
	}
	if (m_currentDataset==nullptr) {
		// call even if foundDatasets.size()==0
		currentDatasetChanged(0);
	}
}

void ImportSismageHorizonWithWorkingSetWidget::showEvent(QShowEvent* event) {
	updateComboBox(true);

	QWidget::showEvent(event);
}

//
void ImportSismageHorizonWithWorkingSetWidget::updateHorizonsInTree()
{
	std::vector<QString> horizonPath = m_importerWidget->getImportedHorizonPath();
	for (QString str:horizonPath)
		qDebug() << "imported   " << str;

	if ( horizonPath.empty() || m_currentDataset == nullptr ) return;
	SeismicSurvey* survey = m_currentDataset->survey();

	for (int i=0; i<horizonPath.size(); i++)
	{
		// check
		QString horizonName = QString::fromStdString(FreeHorizonManager::getHorizonNameFromPath(horizonPath[i].toStdString()));

		QList<IData*> datas = m_manager->folders().horizonsFree->data();
		bool horizonNotHere = true;
		int loadedIdx = 0;
		while (horizonNotHere && loadedIdx<datas.size())
		{
			FreeHorizon* horizon = dynamic_cast<FreeHorizon*>(datas[loadedIdx]);
			if (horizon!=nullptr)
			{
				horizonNotHere = horizonPath[i].compare(horizon->path())!=0;
			}
			if (horizonNotHere)
			{
				loadedIdx++;
			}
		}

		if (!horizonNotHere && loadedIdx<datas.size())
		{
			FreeHorizon* horizon = dynamic_cast<FreeHorizon*>(datas[loadedIdx]);
			if (horizon)
			{
				m_manager->removeFreeHorizons(horizon);
			}
		}
		FreeHorizon *freeHorizon = new FreeHorizon(m_manager, survey, horizonPath[i], horizonName);
		m_manager->addFreeHorizons(freeHorizon);
		IData* isochronData = freeHorizon->getIsochronData();
		if (isochronData) {
			isochronData->setDisplayPreferences({ViewType::InlineView, ViewType::XLineView, ViewType::RandomView}, true);
		}
	}
}

