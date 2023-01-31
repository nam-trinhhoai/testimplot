#include "datamanager.h"
#include "ProjectManager.h"
#include "SurveyManager.h"
#include "ProjectManagerWidget.h"
#include "fileinformationtablewidget.h"
#include "filedeletiontablewidget.h"
#include "trashtablewidget.h"
#include "deleteableleaf.h"
#include "globalconfig.h"
#include "culturals.h"
#include "layerings.h"
#include "sismagedbmanager.h"
#include "leafcontainer.h"
#include "filestoragecontroler.h"
#include "leafcontaineraggregator.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QStackedWidget>
#include <QListWidget>
#include <QPushButton>
#include <QFileDialog>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QSettings>
#include <QDebug>

DataManager::DataManager(QWidget *parent, Qt::WindowFlags f) : QWidget(parent, f) {
	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	QPushButton* loadSessionButton = new QPushButton("Load Session");
	mainLayout->addWidget(loadSessionButton);

	QHBoxLayout* managerLayout = new QHBoxLayout;
	mainLayout->addLayout(managerLayout);

	m_projectManager = new ProjectManager;
	managerLayout->addWidget(m_projectManager);

	m_surveyManager = new SurveyManager;
	m_projectManager->setSurveyManager(m_surveyManager);
	managerLayout->addWidget(m_surveyManager);

	QHBoxLayout* tabLayout = new QHBoxLayout;
	mainLayout->addLayout(tabLayout);

	m_tabList = new QListWidget;
	tabLayout->addWidget(m_tabList);

	m_stackWidget = new QStackedWidget;
	tabLayout->addWidget(m_stackWidget);

	m_controler = new FileStorageControler(this);

	LeafContainerAggregator* trashAggregator = new LeafContainerAggregator;

	LeafContainer* nextVisionHorizonContainer = new LeafContainer;
	m_nextVisionHorizonTrash = new LeafContainer(trashAggregator);

	FileStorageControler::ContainerDuo nextVisionHorizonDuo;
	nextVisionHorizonDuo.main = nextVisionHorizonContainer;
	nextVisionHorizonDuo.trash = m_nextVisionHorizonTrash;

	std::size_t nextVisionHorizonId = m_controler->addContainerDuo(nextVisionHorizonDuo);

	LeafContainer* cubeRgt2RgbContainer = new LeafContainer;
	m_cubeRgt2RgbTrash = new LeafContainer(trashAggregator);

	FileStorageControler::ContainerDuo cubeRgt2RgbDuo;
	cubeRgt2RgbDuo.main = cubeRgt2RgbContainer;
	cubeRgt2RgbDuo.trash = m_cubeRgt2RgbTrash;

	std::size_t cubeRgt2rgbId = m_controler->addContainerDuo(cubeRgt2RgbDuo);

	trashAggregator->addContainer(nextVisionHorizonDuo.trash);
	trashAggregator->addContainer(cubeRgt2RgbDuo.trash);

	m_seismicTable = new FileInformationTableWidget;
	m_culturalTable = new FileInformationTableWidget;
	m_layerTable = new FileInformationTableWidget;
	m_sismageHorizonTable = new FileInformationTableWidget;
	m_nextvisionHorizonTable = new FileDeletionTableWidget(nextVisionHorizonContainer);
	m_cubergt2rgbTable = new FileDeletionTableWidget(cubeRgt2RgbContainer);

	m_trashTable = new TrashTableWidget(trashAggregator);

	nextVisionHorizonContainer->setParent(m_nextvisionHorizonTable);
	cubeRgt2RgbContainer->setParent(m_cubergt2rgbTable);
	trashAggregator->setParent(m_trashTable);

	addTab(m_seismicTable, "Seismic");
	addTab(m_culturalTable, "Cultural");
	addTab(m_layerTable, "Layer");
	addTab(m_sismageHorizonTable, "Sismage Horizon");
	addTab(m_nextvisionHorizonTable, "NextVision Horizon");
	addTab(m_cubergt2rgbTable,"RGT to RGB" );
	addTab(m_trashTable, "Trash");

	connect(loadSessionButton, &QPushButton::clicked, this, &DataManager::loadSessionGui);
	connect(m_projectManager, &ProjectManager::projectChanged, this, &DataManager::projectChanged);
	connect(m_surveyManager, &SurveyManager::surveyChanged, this, &DataManager::surveyChanged);

	connect(m_tabList, &QListWidget::currentItemChanged, this, &DataManager::tabListItemChanged);

	connect(m_nextvisionHorizonTable, &FileDeletionTableWidget::requestDataDeletion, [this, nextVisionHorizonId](std::size_t dataKey) {
		this->m_controler->removeLeafFromMainContainer(dataKey, nextVisionHorizonId);
	});

	connect(m_cubergt2rgbTable, &FileDeletionTableWidget::requestDataDeletion, [this, cubeRgt2rgbId](std::size_t dataKey) {
		this->m_controler->removeLeafFromMainContainer(dataKey, cubeRgt2rgbId);
	});

	connect(m_trashTable, &TrashTableWidget::requestDataRestoration, [this, trashAggregator](LeafContainerAggregator::AggregatorKey leafKey) {
		LeafContainer* container = trashAggregator->container(leafKey.containerId);
		this->m_controler->restoreLeafFromTrashContainer(leafKey.leafId, container);
	});

	connect(m_trashTable, &TrashTableWidget::requestDataDeletion, [this, trashAggregator](LeafContainerAggregator::AggregatorKey leafKey) {
		LeafContainer* container = trashAggregator->container(leafKey.containerId);
		this->m_controler->deleteLeafFromTrashContainer(leafKey.leafId, container);
	});

	// auto load last session
	loadLastSession();
}

DataManager::~DataManager() {

}

void DataManager::addTab(QWidget* widget, const QString& tabName) {
	int index = m_stackWidget->addWidget(widget);
	QListWidgetItem* item = new QListWidgetItem(tabName);
	item->setData(Qt::UserRole, index);
	m_tabList->addItem(item);
}

void DataManager::loadSessionGui() {
    QSettings settings;
    GlobalConfig& config = GlobalConfig::getConfig();
    const QString lastPath = settings.value(LAST_SESSION_PATH_IN_SETTINGS, config.sessionPath()).toString();

    const QString filePath = QFileDialog::getOpenFileName(this,
                                                          tr("Load Session"),
                                                          lastPath,
                                                          QLatin1String("*.json"));
    if (filePath.isEmpty())
        return;

    QFileInfo fi(filePath);
    settings.setValue(LAST_SESSION_PATH_IN_SETTINGS, fi.absoluteFilePath());

    loadSession(filePath);
}

void DataManager::loadLastSession() {
	QSettings settings;
	const QString lastPath = settings.value(LAST_SESSION_PATH_IN_SETTINGS, "").toString();
	if (lastPath.isEmpty() || !QFileInfo(lastPath).exists())
		return;

	loadSession(lastPath);
}

void DataManager::loadSession(const QString& sessionPath) {
	QFile file(sessionPath);
	if (!file.open(QIODevice::ReadOnly))
	{
		qDebug() << "DataManager : cannot load session, file not readable";
		return;
	}

	QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
	if (!doc.isObject())
	{
		qDebug() << "DataManager : cannot load session, root is not a json object";
		return;
	}

	bool isProjectValid = false;
	bool isSurveyValid = false;

	QJsonObject rootObj = doc.object();
	QString project = "";
	QString survey = "";

	if (rootObj.contains(projectTypeKey0))
	{
		QString projectPath = "";
		QString project_type = rootObj.value(projectTypeKey0).toString("None");
		if ( project_type.compare("USER") == 0 && rootObj.contains(projectPathKey0) )
		{
			projectPath = rootObj.value(projectPathKey0).toString(".");
			m_projectManager->setUserProjectPath(projectPath);
		}
		bool isProjectTypeValid = m_projectManager->setProjectType(project_type);
		if (isProjectTypeValid && rootObj.contains(projectKey0))
		{ // load project
			QString project = rootObj.value(projectKey0).toString("");

			if (m_projectManager->setProjectName(project))
			{
				isProjectValid = true;
			}
		}
		if (isProjectValid && rootObj.contains(surveyKey0))
		{
			QString survey = rootObj.value(surveyKey0).toString("");
			if (m_surveyManager->setForceName(survey))
			{
				isSurveyValid = true;
			}
		}
	}
}

void DataManager::projectChanged() {
	qDebug() << m_projectManager->getName();
	if (m_cacheProjectPath.compare(m_projectManager->getPath())==0) {
		return;
	}
	QString projectName = m_projectManager->getName();
	if (projectName.isNull() || projectName.isEmpty()) {
		m_cacheProjectPath = "";
		clearProject();
	} else {
		m_cacheProjectPath = m_projectManager->getPath();
		m_culturals.reset(new Culturals(SismageDBManager::projectPath2CulturalPath(m_cacheProjectPath.toStdString())));
		QList<MonoFileBasedData> culturalFiles;
		for (const Cultural* cultural : m_culturals->getCulturals()) {
			culturalFiles.append(MonoFileBasedData(QString::fromStdString(cultural->getName()), QString::fromStdString(cultural->getGriFilePath())));
		}
		m_culturalTable->addData(culturalFiles);

		m_controler->setProjectPath(m_cacheProjectPath);
	}
}

void DataManager::surveyChanged() {
	qDebug() << m_surveyManager->getName();
	if (m_cacheSurveyPath.compare(m_surveyManager->getPath())==0) {
		return;
	}
	QString surveyName = m_surveyManager->getName();
	if (surveyName.isNull() || surveyName.isEmpty()) {
		m_cacheSurveyPath = "";
		clearSurvey();
	} else {
		m_cacheSurveyPath = m_surveyManager->getPath();

		QList<MonoFileBasedData> seismicFiles;
		std::vector<QString> seismicList = SeismicManager::getSeismicList(QString::fromStdString(SismageDBManager::surveyPath2DatasetPath(m_cacheSurveyPath.toStdString()))); // only use to get names
		for (const QString& path : seismicList) {
			QFileInfo pathFileInfo(path);
			QString ext = pathFileInfo.suffix(); // multiple "." in file name are expected
			QStringList separatedName = pathFileInfo.completeBaseName().split("_"); // _ is the separator used by geotime to generate rgt and dip files
			long index = separatedName.size()-1;
			while (index>0 && !separatedName[index].startsWith("rgt") && !separatedName[index].startsWith("dipxy") &&
					!separatedName[index].startsWith("dipxz")) { // because expect seismic3d.myVolume_rgt.ext or seismic3d.myVolume_uniqueuuid_rgt.ext
				index--;
			}
			bool attributeMatch = index>0;

			if (ext.compare("cwt")==0 || attributeMatch) {
				QString name = SeismicManager::seismicFullFilenameToTinyName(path);
				seismicFiles.append(MonoFileBasedData(name, path));
			}
		}
		m_seismicTable->addData(seismicFiles);

		QList<MonoFileBasedData> sismageHorizonFiles;
		QDir sismageHorizonDir(QString::fromStdString(SismageDBManager::surveyPath2HorizonsPath(m_cacheSurveyPath.toStdString())));
		QStringList filters;
		filters << "*.iso"; // horizon extension is sismage
		QFileInfoList sismageHorizonList = sismageHorizonDir.entryInfoList(filters, QDir::Files, QDir::Name);
		for (const QFileInfo& horizonInfo : sismageHorizonList) {
			QString name = horizonInfo.baseName();
			if (name.startsWith("Nv")) { // prefix used to save horizon, see slicer/dialog/exportmultilayerblocdialog.cpp
				sismageHorizonFiles.append(MonoFileBasedData(name, horizonInfo.absoluteFilePath()));
			}
		}
		m_sismageHorizonTable->addData(sismageHorizonFiles);

		QList<MonoFileBasedData> layerFiles;
		Layerings layerings(SismageDBManager::surveyPath2LayerPath(m_cacheSurveyPath.toStdString()), "NextVision"); // see stackbasemap.cpp for layering object initialization
		for (const std::string& layeringName : layerings.getNames()) {
			Layering* layering = layerings.getLayering(layeringName);
			layerFiles.append(MonoFileBasedData(QString::fromStdString(layering->getName()), QString::fromStdString(layering->getDirName())));
		}

		m_layerTable->addData(layerFiles);

		QDir importExportDatasetsDir(m_cacheSurveyPath + "/ImportExport/IJK/");
		QStringList importExportDatasetsList = importExportDatasetsDir.entryList(QStringList(), QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
		for (const QString& importExportDatasetDir : importExportDatasetsList) {
			QList<DeletableLeaf> nextVisionHorizonLeaves = DeletableLeaf::findLeavesFromHorizonDirectory(importExportDatasetsDir.absoluteFilePath(importExportDatasetDir + "/HORIZON_GRIDS/"));
			m_nextvisionHorizonTable->container()->addLeafs(nextVisionHorizonLeaves);

			QList<DeletableLeaf> rgt2rgbLeaves = DeletableLeaf::findLeavesFromRGT2RGBDirectory(importExportDatasetsDir.absoluteFilePath(importExportDatasetDir + "/cubeRgt2RGB/"));
			m_cubergt2rgbTable->container()->addLeafs(rgt2rgbLeaves);
		}

		// fill trash
		QDir importExportTrashDatasetsDir(m_cacheSurveyPath + "/ImportExport/IJK_Trash/IJK/");
		QStringList importExportTrashDatasetsList = importExportTrashDatasetsDir.entryList(QStringList(), QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
		for (const QString& importExportTrashDatasetDir : importExportTrashDatasetsList) {
			QList<DeletableLeaf> nextVisionHorizonLeaves = DeletableLeaf::findLeavesFromHorizonDirectory(importExportTrashDatasetsDir.absoluteFilePath(importExportTrashDatasetDir + "/HORIZON_GRIDS/"));
			m_nextVisionHorizonTrash->addLeafs(nextVisionHorizonLeaves);

			QList<DeletableLeaf> rgt2rgbLeaves = DeletableLeaf::findLeavesFromRGT2RGBDirectory(importExportTrashDatasetsDir.absoluteFilePath(importExportTrashDatasetDir + "/cubeRgt2RGB/"));
			m_cubeRgt2RgbTrash->addLeafs(rgt2rgbLeaves);
		}
	}
}

void DataManager::clearProject() {
	// clear survey
	clearSurvey();

	// clear ui
	m_culturalTable->clear();

	// clear data
	m_culturals.reset(nullptr);
	m_controler->clearProjectPath();
}

void DataManager::clearSurvey() {
	// clear ui
	m_seismicTable->clear();
	m_sismageHorizonTable->clear();
	m_layerTable->clear();
	m_nextvisionHorizonTable->container()->clear();
	m_cubergt2rgbTable->container()->clear();
	m_trashTable->container()->clearContainersContent();
}

void DataManager::tabListItemChanged(QListWidgetItem* current, QListWidgetItem* previous) {
	if (current) {
		bool ok;
		int widgetIndex = current->data(Qt::UserRole).toInt(&ok);
		if (ok) {
			m_stackWidget->setCurrentIndex(widgetIndex);
		}
	}
}
