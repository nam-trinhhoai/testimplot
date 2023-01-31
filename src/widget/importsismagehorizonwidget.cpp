#include "importsismagehorizonwidget.h"

#include "freeHorizonManager.h"
#include "freeHorizonQManager.h"
#include "gettextwithoverwritedialog.h"
#include "globalconfig.h"
#include "grid2d.h"
#include "nextvisiondbmanager.h"
#include "sismagedbmanager.h"
#include "util_filesystem.h"

#include <QCheckBox>
#include <QDir>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QMutexLocker>
#include <QPushButton>
#include <QVBoxLayout>

#include <cstdlib>

ImportSismageHorizonWidget::ImportSismageHorizonWidget(QWidget* parent) : QWidget(parent) {
	m_badDatasetToolTip = "Bad dataset provided";

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	QHBoxLayout* selectionLayout = new QHBoxLayout;
	mainLayout->addLayout(selectionLayout);
	QPushButton* selectButton = new QPushButton("Select all");
	selectionLayout->addWidget(selectButton);
	QPushButton* unselectButton = new QPushButton("Unselect all");
	selectionLayout->addWidget(unselectButton);

	QHBoxLayout* filterLayout = new QHBoxLayout;
	mainLayout->addLayout(filterLayout);
	m_filterLineEdit = new QLineEdit;
	filterLayout->addWidget(m_filterLineEdit);
	QPushButton* clearFilterButton = new QPushButton("X");
	clearFilterButton->setStyleSheet("QPushButton {min-width: 40px}");
	filterLayout->addWidget(clearFilterButton);

	m_listWidget = new QListWidget;
	m_listWidget->setSelectionMode(QAbstractItemView::MultiSelection);
	m_listWidget->setSortingEnabled(true);
	m_listWidget->setContextMenuPolicy(Qt::CustomContextMenu);
	mainLayout->addWidget(m_listWidget);

	QCheckBox* brickedCheckBox = new QCheckBox("Is data bricked ? ");
	brickedCheckBox->setToolTip("Change if horizon data is not valid");
	brickedCheckBox->setCheckState(brickedCheckBox ? Qt::Checked : Qt::Unchecked);
	mainLayout->addWidget(brickedCheckBox);

	m_importButton = new QPushButton("Import");
	mainLayout->addWidget(m_importButton);

	connect(selectButton, &QPushButton::clicked, this, &ImportSismageHorizonWidget::selectAllItems);
	connect(unselectButton, &QPushButton::clicked, this, &ImportSismageHorizonWidget::unselectAllItems);
	connect(m_filterLineEdit, &QLineEdit::textChanged, this, &ImportSismageHorizonWidget::changeFilter);
	connect(clearFilterButton, &QPushButton::clicked, this, &ImportSismageHorizonWidget::clearFilter);
	connect(m_listWidget, &QWidget::customContextMenuRequested, this, &ImportSismageHorizonWidget::openItemMenu);
	connect(brickedCheckBox, &QCheckBox::stateChanged, this, &ImportSismageHorizonWidget::toggleBricked);
	connect(m_importButton, &QPushButton::clicked, this, &ImportSismageHorizonWidget::importHorizons);

	setDatasetPathPrivate("");

	m_preferedDatasetWorker = new FindPreferedDatasetsWorker;
	m_preferedDatasetWorker->moveToThread(&m_workerThread);
	connect(&m_workerThread, &QThread::finished, m_preferedDatasetWorker, &QObject::deleteLater);
	connect(this, &ImportSismageHorizonWidget::searchPreferedDatasetSignal, m_preferedDatasetWorker, &FindPreferedDatasetsWorker::doWork);
	connect(m_preferedDatasetWorker, &FindPreferedDatasetsWorker::preferedDatasetFound, this, &ImportSismageHorizonWidget::preferedDatasetFound);
	m_workerThread.start();
}

ImportSismageHorizonWidget::~ImportSismageHorizonWidget() {
	for (auto it=m_apiConnections.begin(); it!=m_apiConnections.end(); it++) {
		disconnect(it->second.mainConnection);
		disconnect(it->second.errorConnection);
		disconnect(it->second.destroyedApiConnection);
	}

	m_preferedDatasetWorker->stop();
	m_workerThread.quit();
	m_workerThread.wait();
}

QString ImportSismageHorizonWidget::projectPath() const {
	return m_projectPath;
}

QString ImportSismageHorizonWidget::projectName() const {
	return QFileInfo(m_projectPath).fileName();
}

void ImportSismageHorizonWidget::setProjectPath(const QString& path) {
	QFileInfo projectFileInfo(path);
	QString projectPath = projectFileInfo.absoluteFilePath();
	bool changePath = projectPath.compare(m_projectPath)!=0;

	if (changePath) {
		setDatasetPathPrivate("");
		m_surveyPath = "";
		m_currentSurveyId = "";

		m_preferedDatasetWorker->stop();
		m_listWidget->clear();
		m_horizons.clear();
		m_cacheSurveys.clear();
		m_cacheGrids.clear();

		std::string oldDirProject = SismageDBManager::dirProjectPathFromProjectPath(m_projectPath.toStdString());
		std::string newDirProject = SismageDBManager::dirProjectPathFromProjectPath(projectPath.toStdString());
		bool changeDirProject = oldDirProject!=newDirProject;

		if (changeDirProject && m_serversHolder) {
			m_serversHolder->setDirProject(QString::fromStdString(newDirProject));
		}

		m_projectPath = projectPath;

		loadHorizons();
	}
}

QString ImportSismageHorizonWidget::surveyPath() const {
	return m_surveyPath;
}

QString ImportSismageHorizonWidget::surveyName() const {
	QString name;
	if (!m_surveyPath.isNull() && !m_surveyPath.isEmpty()) {
		name = QString::fromStdString(SismageDBManager::surveyNameFromSurveyPath(m_surveyPath.toStdString()));
	}
	return name;
}

void ImportSismageHorizonWidget::setSurveyPath(const QString& path) {
	QFileInfo surveyFileInfo(path);
	QString surveyPath = surveyFileInfo.absoluteFilePath();
	bool changePath = surveyPath.compare(m_surveyPath)!=0;

	if (changePath) {
		setDatasetPathPrivate("");

		m_preferedDatasetWorker->stop();
		m_listWidget->clear();
		m_cacheGrids.clear();

		std::string oldProject = SismageDBManager::projectPathFromSurveyPath(m_surveyPath.toStdString());
		std::string newProject = SismageDBManager::projectPathFromSurveyPath(surveyPath.toStdString());
		bool changeProject = oldProject!=newProject;

		if (changeProject) {
			setProjectPath(QString::fromStdString(newProject));
		}

		m_surveyPath = surveyPath;
		m_currentSurveyId = "";
		// reset prefered datasets
		for (auto it=m_horizons.begin(); it!=m_horizons.end(); it++) {
			it->second->preferedDatasetPath = "";
		}
		updateList();
	}
}

QString ImportSismageHorizonWidget::datasetPath() const {
	return m_datasetPath;
}

QString ImportSismageHorizonWidget::datasetName() const {
	QString name;
	if (!m_datasetPath.isNull() && !m_datasetPath.isEmpty()) {
		name = QString::fromStdString(SismageDBManager::datasetNameFromDatasetPath(m_datasetPath.toStdString()));
	}
	return name;
}

void ImportSismageHorizonWidget::setDatasetPath(const QString& path) {
	QFileInfo datasetFileInfo(path);
	QString datasetPath = datasetFileInfo.absoluteFilePath();
	bool changePath = datasetPath.compare(m_datasetPath)!=0;

	if (changePath) {
		std::string oldSurvey = SismageDBManager::survey3DPathFromDatasetPath(m_datasetPath.toStdString());
		std::string newSurvey = SismageDBManager::survey3DPathFromDatasetPath(datasetPath.toStdString());
		bool changeSurvey = oldSurvey!=newSurvey;

		if (changeSurvey) {
			setSurveyPath(QString::fromStdString(newSurvey));
		}

		setDatasetPathPrivate(datasetPath);

		// reset prefered datasets
		for (auto it=m_horizons.begin(); it!=m_horizons.end(); it++) {
			it->second->preferedDatasetPath = m_datasetPath;
		}
		updateListColor();
	}
}

void ImportSismageHorizonWidget::setDatasetPathPrivate(const QString& path) {
	bool validDataset = !path.isNull() && !path.isEmpty();
	if (validDataset) {
		Grid2D grid2DDataset = getGrid2DFromDatasetPath(path);

		validDataset = grid2DDataset.isGridValid();
	}

	if (validDataset) {
		m_importButton->setStyleSheet("");
		m_importButton->setToolTip("");
	} else {
		m_importButton->setStyleSheet(QString("QPushButton{ background: %1; }").arg(QColor(Qt::red).name()));
		m_importButton->setToolTip(m_badDatasetToolTip);
	}

	m_datasetPath = path;
}

void ImportSismageHorizonWidget::setBadDatasetToolTip(const QString& msg) {
	m_badDatasetToolTip = msg;
	setDatasetPathPrivate(m_datasetPath);
}

const std::map<QString, std::shared_ptr<ImportSismageHorizonWidget::Horizon>>& ImportSismageHorizonWidget::horizons() const {
	return m_horizons;
}

void ImportSismageHorizonWidget::extractHorizon(Task task) {
	QMutexLocker locker(&m_tasksMutex);

	const OpenAPI::Interpretation::OAIInterpretationHorizon& horizon = task.horizon->horizon;

	// check that the task is not already running
	auto itOnGoing = m_onGoingTasks.find(horizon.getId());
	auto itDone = m_doneTasks.find(horizon.getId());
	auto itError = m_errorTasks.find(horizon.getId());

	if (itDone!=m_doneTasks.end() || itOnGoing!=m_onGoingTasks.end() || itError!=m_errorTasks.end()) {
		return;
	}

	initServers();

	if (m_serversHolder==nullptr || !m_serversHolder->isValid() || m_serversHolder->interpretationApi()==nullptr) {
		return;
	}

	openConnectionForApi(m_serversHolder->interpretationApi());

	OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker* worker = m_serversHolder->interpretationApi()->getIsochronValues(
			task.isochron.getId(), 0, task.isochron.getBrickSizeDescriptor().getBrickCount()-1);
	task.worker = worker;

	m_onGoingTasks[horizon.getId()] = task;
}

void ImportSismageHorizonWidget::changeFilter(QString text) {
	m_listFilter = text;

	updateList();
}

void ImportSismageHorizonWidget::clearFilter() {
	m_filterLineEdit->clear();
}

void ImportSismageHorizonWidget::closeConnectionForApi(OpenAPI::Interpretation::OAIInterpretationDefaultApi* api) {
	auto it = m_apiConnections.find(api);
	if (it!=m_apiConnections.end()) {
		disconnect(it->second.mainConnection);
		disconnect(it->second.errorConnection);
		disconnect(it->second.destroyedApiConnection);
	}
}

QString ImportSismageHorizonWidget::getCurrentSurveyId() {
	if (!m_currentSurveyId.isNull() && !m_currentSurveyId.isEmpty()) {
		return m_currentSurveyId;
	}

	if (m_cacheSurveys.size()==0) {
		loadSurveys();
	}

	auto it = m_cacheSurveys.cbegin();
	QString surveyName = this->surveyName();
	while ((m_currentSurveyId.isNull() || m_currentSurveyId.isEmpty()) && it!=m_cacheSurveys.cend()) {
		if (surveyName.compare(it->second.getName())==0) {
			m_currentSurveyId = it->first;
		}
		it++;
	}

	return m_currentSurveyId;
}

Grid2D ImportSismageHorizonWidget::getGrid2DFromDatasetPath(const QString& datasetPath) {
	bool gotLock = m_cacheGridsMutex.tryLockForRead(1000);
	Grid2D grid;
	if (gotLock) {
		auto it = m_cacheGrids.find(datasetPath);
		if (it==m_cacheGrids.end()) {
			m_cacheGridsMutex.unlock();
			grid = Grid2D::getMapGridFromDatasetPath(datasetPath.toStdString());
			if (m_cacheGridsMutex.tryLockForWrite(1000)) {
				m_cacheGrids[datasetPath] = grid;
				m_cacheGridsMutex.unlock();
			}
		} else {
			grid = it->second;
			m_cacheGridsMutex.unlock();
		}
	} else {
		grid = Grid2D::getMapGridFromDatasetPath(datasetPath.toStdString());
	}
	return grid;
}

void ImportSismageHorizonWidget::loadHorizons() {
	initServers();
	if (m_serversHolder==nullptr || !m_serversHolder->isValid() || m_serversHolder->interpretationApi()==nullptr) {
		return;
	}

	m_preferedDatasetWorker->stop();
	m_listWidget->clear();
	m_horizons.clear();

	QList<OpenAPI::Interpretation::OAIInterpretationHorizon> horizons = SeaUtils::getHorizons(*(m_serversHolder->interpretationApi()), projectName());
	for (long i=0; i<horizons.size(); i++) {
		OpenAPI::Interpretation::OAIInterpretationHorizon& horizon = horizons[i];
		if (horizon.is_id_Set()) {
			std::shared_ptr<Horizon> horizonData;
			horizonData.reset(new Horizon);
			horizonData->horizon = horizon;
			horizonData->preferedDatasetPath = m_datasetPath;

			m_horizons[horizon.getId()] = horizonData;
		}
	}
	updateList();
}

void ImportSismageHorizonWidget::loadIsochrons(Horizon& horizonStruct) {
	initServers();
	if (m_serversHolder==nullptr || !m_serversHolder->isValid() || m_serversHolder->interpretationApi()==nullptr) {
		return;
	}

	QList<OpenAPI::Interpretation::OAIInterpretationHorizonProperty> isochrons = SeaUtils::getIsochrons(*(m_serversHolder->interpretationApi()), horizonStruct.horizon.getId());

	for (long j=0; j<isochrons.size(); j++) {
		if (isochrons[j].is_survey_id_Set()) {
			horizonStruct.isochrons[isochrons[j].getSurveyId()] = isochrons[j];
		}
	}
	horizonStruct.areIsochronsSet = true;
}

void ImportSismageHorizonWidget::loadSurveys() {
	initServers();
	if (m_serversHolder==nullptr || !m_serversHolder->isValid() || m_serversHolder->seismicApi()==nullptr) {
		return;
	}

	QMutexLocker locker(&m_cacheSurveysMutex);

	m_cacheSurveys.clear();
	QList<OpenAPI::Seismic::OAISeismicSurvey3D> surveys = SeaUtils::getSurveys(*(m_serversHolder->seismicApi()), projectName());

	for (long i=0; i<surveys.size(); i++) {
		OpenAPI::Seismic::OAISeismicSurvey3D& survey = surveys[i];
		if (survey.is_name_Set() && survey.is_key_Set()) {
			m_cacheSurveys[survey.getKey()] = survey;
		}
	}
}

QString ImportSismageHorizonWidget::askSavePath(const QString& oriSavePath,
		const OpenAPI::Interpretation::OAIInterpretationHorizon& horizon, bool& skipOutput, bool& overwriteOutput,
		bool& skipAll, bool& overwriteAll) {
	QString savePath = oriSavePath;
	QString overwrite = "overwrite";
	QString writeNextTo = "save with another name";
	QString skipHorizon = "skip horizon";
	QString skipAllOpt = "skip all";
	QString overwriteAllOpt = "overwrite all";
	QString horizonName = horizon.getRealName();
	QString version = horizon.getVersion();
	if (!version.isNull() && !version.isEmpty()) {
		horizonName += "." + version;
	}
	QStringList options;
	options << overwrite << writeNextTo << skipHorizon << skipAllOpt << overwriteAllOpt;
	bool okItem;
	QString txt = QInputDialog::getItem(this, tr("File exists"),
			tr("Horizon ")+horizonName+" exists, do you want to : ", options, 1, false, &okItem);

	overwriteOutput = overwriteAll || txt.compare(overwrite)==0;
	bool skip = true;
	if (okItem && txt.compare(writeNextTo)==0) {
		QString saveDir = QFileInfo(savePath).dir().absolutePath();
		QString saveName = QFileInfo(savePath).fileName();
		QString suffix;
		QStringList splitName = saveName.split("(");
		QString editableSaveName = saveName;
		if (splitName.size()>1) {
			suffix = "_("+splitName.last();
			editableSaveName = saveName.chopped(suffix.size());
		}

		bool chooseName = true;
		while (chooseName && !overwriteOutput && QFileInfo(saveDir+"/"+editableSaveName+suffix).exists()) {
			long addInt = 2;
			bool notFoundInt = true;
			while (notFoundInt && addInt<1000) { // 1000 is a safety
				notFoundInt = QFileInfo(saveDir+"/"+editableSaveName+"_"+QString::number(addInt)+suffix).exists();
				if (notFoundInt) {
					addInt++;
				}
			}
			QString newSaveName = editableSaveName+"_"+QString::number(addInt);
			GetTextWithOverWriteDialog dialog("Save name", this);
			dialog.setWindowTitle("Set horizon saving name");
			dialog.setText(newSaveName);
			int dialogRes = dialog.exec();
			newSaveName = dialog.text();
			chooseName = dialogRes==QDialog::Accepted && !newSaveName.isNull()  && !newSaveName.isEmpty();
			if (chooseName) {
				editableSaveName = newSaveName;
				overwriteOutput = dialog.isOverwritten();
			}
		}

		if (chooseName) {
			savePath = saveDir + "/" + editableSaveName+suffix;
		}

		skip = !chooseName;
	} else if (okItem && (txt.compare(overwrite)==0 || txt.compare(overwriteAllOpt)==0)) {
		skip = false;
	} else {
		skip = true;
	}

	skipOutput = skip;
	skipAll = txt.compare(skipAllOpt)==0;
	overwriteAll = txt.compare(overwriteAllOpt)==0;
	return savePath;
}

bool ImportSismageHorizonWidget::validateOverwrite(const QString& horizonPath, bool& skipAll, bool& overwriteAll) {
	bool overwrite = true;
	bool hasAttributes = FreeHorizonManager::hasAttributs(horizonPath.toStdString());
	if (hasAttributes && !skipAll && !overwriteAll) {
		QString horizonName = QDir(horizonPath).dirName();

		QString overwriteOpt = "overwrite horizon";
		QString skipHorizon = "skip horizon";
		QString skipAllOpt = "skip all";
		QString overwriteAllOpt = "overwrite all";
		QStringList options;
		options << overwriteOpt << skipHorizon << skipAllOpt << overwriteAllOpt;
		bool okItem;
		QString output = QInputDialog::getItem(this, tr("Overwrite attributes"), tr("Horizon ")+horizonName+" has attributes, do you want to : ",
				options, 0, false, &okItem);

		if (okItem) {
			skipAll = output.compare(skipAllOpt)==0;
			overwriteAll = output.compare(overwriteAllOpt)==0;
			overwrite = overwriteAll || output.compare(overwriteOpt)==0;
		} else {
			overwrite = false;
		}
	} else if (hasAttributes && skipAll) {
		overwrite = false;
	}
	return overwrite;
}

void ImportSismageHorizonWidget::importHorizons() {


	m_importedHorizonPath.clear();


	if (m_datasetPath.isNull() || m_datasetPath.isEmpty()) {
		return;
	}

	QList<QListWidgetItem*> selectedItems = m_listWidget->selectedItems();

	QString currentSurveyId = getCurrentSurveyId();
	QString errMsg;
	int errCount = 0;
	bool skipAll = false;
	bool overwriteAll = false;
	bool attributesSkipAll = false;
	bool attributesOverwriteAll = false;
	{
		QMutexLocker locker(&m_importMutex);

		for (long i=0; i<selectedItems.size(); i++) {
			QString horizonId = selectedItems[i]->data(Qt::UserRole).toString();
			auto it = m_horizons.find(horizonId);
			if (it!=m_horizons.end()) {
				std::pair<bool, QString> importedResult = isHorizonImported(*(it->second));
				bool skip = false;
				bool overwrite = false;
				QString horizonName = it->second->horizon.getRealName();
				QString version = it->second->horizon.getVersion();
				QString savePath = getSavePath(m_surveyPath, it->second->preferedDatasetPath, horizonName, version);
				if (importedResult.first && !overwriteAll && !skipAll) {
					savePath = askSavePath(savePath, it->second->horizon, skip, overwrite, skipAll, overwriteAll);
				} else if (importedResult.first && overwriteAll) {
					skip = false;
				} else if (importedResult.first && skipAll) {
					skip = true;
				}
				if (overwrite || overwriteAll) {
					overwrite = validateOverwrite(savePath, attributesSkipAll, attributesOverwriteAll);
					skip = !overwrite;
				}

				if (!skip && !it->second->areIsochronsSet) {
					loadIsochrons(*(it->second));
				}

				auto isoIt = it->second->isochrons.find(currentSurveyId);
				if (!skip && isoIt!=it->second->isochrons.end()) {
					if (overwrite) {
						bool eraseRes = FreeHorizonManager::erase(savePath.toStdString());
						if (!eraseRes) {
							QString horizonFullName = horizonName;
							QString version = it->second->horizon.getVersion();
							if (!version.isNull() && !version.isEmpty()) {
								horizonFullName += "." + version;
							}
							QMessageBox::information(this, tr("Horizon not removable"), tr("Failed to erase :")+horizonFullName);
						}
					}

					Task task;
					task.isochron = isoIt->second;
					task.horizon = it->second;
					task.datasetPath = it->second->preferedDatasetPath;
					task.surveyPath = m_surveyPath;
					task.isBricked = m_isBricked;
					task.savePath = savePath;

					extractHorizon(task);
				} else if (!skip) {
					if (errCount>0) {
						errMsg += "\n";
					}
					errMsg += "No isochron on survey for " + it->second->horizon.getRealName();
					QString version = it->second->horizon.getVersion();
					if (!version.isNull() && !version.isEmpty()) {
						errMsg += "." + version;
					}
					errCount++;
				}
			}
		}
	}

	unselectAllItems();

	if (errCount>0) {
		QMessageBox::information(this, "Horizons without isochron", errMsg);
	}

	postProcessTasks();
}

void ImportSismageHorizonWidget::initServers() {
	QFileInfo projectFileInfo(m_projectPath);
	bool surveyLoadable = !m_projectPath.isNull() && !m_projectPath.isEmpty() && projectFileInfo.exists() &&
			projectFileInfo.isDir();
	if (!surveyLoadable) {
		return;
	}

	QMutexLocker locker(&m_initServersMutex);

	QString dirProject = QString::fromStdString(SismageDBManager::dirProjectPathFromProjectPath(m_projectPath.toStdString()));
	if (m_serversHolder!=nullptr && m_serversHolder->isValid() && m_serversHolder->dirProject().compare(dirProject)==0) {
		return;
	}

	m_serversHolder.reset(new SeaUtils::ServersHolder(dirProject));
}

std::pair<bool, QString> ImportSismageHorizonWidget::isHorizonImported(const Horizon& horizon) {
	if (horizon.preferedDatasetPath.isNull() || horizon.preferedDatasetPath.isEmpty()) {
		return std::pair<bool, QString>(false, horizon.preferedDatasetPath);
	}

	std::string preferedDatasetName = SismageDBManager::datasetNameFromDatasetPath(horizon.preferedDatasetPath.toStdString());
	std::string surveyPath = SismageDBManager::survey3DPathFromDatasetPath(horizon.preferedDatasetPath.toStdString());
	QString horizonPath = QString::fromStdString(NextVisionDBManager::surveyPath2HorizonDir(surveyPath));
	QString horizonNvName = QString::fromStdString(NextVisionDBManager::getNextVisionNameFromSismageHorizonName(horizon.horizon.getRealName().toStdString(),
			horizon.horizon.getVersion().toStdString(), preferedDatasetName));
	QString horizonDirPath = horizonPath + "/" + horizonNvName;

	QFileInfo horizonFileInfo(horizonDirPath);
	bool valid = horizonFileInfo.exists() && horizonFileInfo.isDir();

	QString selectedDatasetPath;
	if (!valid) {
		Grid2D grid2DDataset = getGrid2DFromDatasetPath(horizon.preferedDatasetPath);
		valid = grid2DDataset.isGridValid();

		std::list<std::string> potentialHorizons;
		if (valid) {
			potentialHorizons = NextVisionDBManager::searchNextVisionHorizonForSismageHorizon(horizon.horizon.getRealName().toStdString(),
				horizon.horizon.getVersion().toStdString(), horizonPath.toStdString());
			valid = potentialHorizons.size()>0;
		}

		if (valid) {
			valid = false;
			auto horizonIt = potentialHorizons.cbegin();
			while (!valid && horizonIt!=potentialHorizons.end()) {
				std::string currentDatasetName = FreeHorizonManager::dataSetNameGet(*horizonIt);
				std::string currentDatasetPath = SismageDBManager::datasetPathFromDatasetFileNameAndSurveyPath(currentDatasetName, surveyPath);

				Grid2D currentGrid2DDataset = getGrid2DFromDatasetPath(QString::fromStdString(currentDatasetPath));
				valid = currentGrid2DDataset.isGridValid() && currentGrid2DDataset.isSameGrid(grid2DDataset) &&
						currentGrid2DDataset.depthAxis() && grid2DDataset.depthAxis();

				if (!valid) {
					horizonIt++;
				} else {
					selectedDatasetPath = QString::fromStdString(currentDatasetPath);
				}
			}
		}
	} else {
		selectedDatasetPath = horizon.preferedDatasetPath;
	}

	return std::pair<bool, QString>(valid, selectedDatasetPath);
}

void ImportSismageHorizonWidget::openConnectionForApi(OpenAPI::Interpretation::OAIInterpretationDefaultApi* api) {
	if (api==nullptr) {
		return;
	}

	auto it = m_apiConnections.find(api);
	if (it==m_apiConnections.end()) {
		ApiConnections conn;
		conn.mainConnection = connect(api, &OpenAPI::Interpretation::OAIInterpretationDefaultApi::getIsochronValuesSignalFull, this,
				&ImportSismageHorizonWidget::valuesLoaded);
		conn.errorConnection = connect(api, &OpenAPI::Interpretation::OAIInterpretationDefaultApi::getIsochronValuesSignalEFull, this,
				&ImportSismageHorizonWidget::valuesLoadedWithError);
		conn.destroyedApiConnection = connect(api, &QObject::destroyed, [this, api]() {
			closeConnectionForApi(api);
		});

		m_apiConnections[api] = conn;
	}
}

void ImportSismageHorizonWidget::openItemMenu(const QPoint& pos) {
	QListWidgetItem* item = m_listWidget->itemAt(pos);
	if (item==nullptr) {
		return;
	}

	QMenu menu;
	menu.addAction("Folder", [this, item]() {
		GlobalConfig& config = GlobalConfig::getConfig();
		QString horizonDir = QString::fromStdString(SismageDBManager::surveyPath2HorizonsPath(m_surveyPath.toStdString()));
		QString cmd = config.fileExplorerProgram() +" " + horizonDir;
		cmd.replace("(", "\\(");
		cmd.replace(")", "\\)");
		std::system(cmd.toStdString().c_str());
	});

	QPoint globalPos = m_listWidget->mapToGlobal(pos);
	menu.exec(globalPos);
}

void ImportSismageHorizonWidget::postProcessTasks() {
	QMutexLocker locker(&m_tasksMutex);

	if (m_onGoingTasks.size()>0 || (m_doneTasks.size()==0 && m_errorTasks.size()==0)) {
		return;
	}
	// check that import is finished
	if (!m_importMutex.tryLock()) {
		return;
	}// else mutex must be unlock to avoid dead lock
	m_importMutex.unlock();

	QString doneHorizonStr = " horizon";
	if (m_doneTasks.size()>1) {
		doneHorizonStr = " horizons";
	}

	QString errorMsg;
	if (m_errorTasks.size()>0) {
		QString errorStr = " error";
		if (m_errorTasks.size()>1) {
			errorStr = " errors";
		}
		errorMsg = ", but got "+QString::number(m_errorTasks.size())+errorStr;
	}

	QMessageBox::information(this, "Horizon import", "Imported "+QString::number(m_doneTasks.size())+doneHorizonStr+errorMsg);
	m_doneTasks.clear();
	m_errorTasks.clear();

	updateListColor();

	emit importFinished();
}

void ImportSismageHorizonWidget::preferedDatasetFound(QString horizonId, QString preferedDataset) {
	QListWidgetItem* horizonItem = nullptr;
	int i=0;
	while (horizonItem==nullptr && i<m_listWidget->count()) {
		QListWidgetItem* item = m_listWidget->item(i);
		QString itemHorizonId = item->data(Qt::UserRole).toString();
		if (horizonId.compare(itemHorizonId)==0) {
			horizonItem = item;
		}

		i++;
	}

	if (horizonItem) {
		horizonItem->setForeground(Qt::green);
		auto it = m_horizons.find(horizonId);
		if (it!=m_horizons.end()) {
			it->second->preferedDatasetPath = preferedDataset;
		}
	}
}

void ImportSismageHorizonWidget::selectAllItems() {
	m_listWidget->selectAll();
}

void ImportSismageHorizonWidget::toggleBricked(int toggle) {
	m_isBricked = toggle==Qt::Checked;
}

void ImportSismageHorizonWidget::unselectAllItems() {
	m_listWidget->clearSelection();
}

void ImportSismageHorizonWidget::updateList() {
	m_preferedDatasetWorker->stop();
	m_listWidget->clear();
	QString currentSurveyId = getCurrentSurveyId();
	for (auto it = m_horizons.begin(); it!=m_horizons.end(); it++) {
		auto isochronIt = it->second->isochrons.find(currentSurveyId);

		if (m_surveyPath.isNull() || m_surveyPath.isEmpty() || !it->second->areIsochronsSet || isochronIt!=it->second->isochrons.end()) {
			QString version = it->second->horizon.getVersion();
			QString itemText;
			if (version.isNull() || version.isEmpty()) {
				itemText = it->second->horizon.getRealName();
			} else {
				itemText = it->second->horizon.getRealName() + "." + version;
			}

			if (itemText.contains(m_listFilter, Qt::CaseInsensitive)) {
				QListWidgetItem* item = new QListWidgetItem;
				item->setText(itemText);
				item->setToolTip(itemText);
				item->setData(Qt::UserRole, it->first);
				item->setForeground(Qt::white);

				m_listWidget->addItem(item);
			}
		}
	}

	updateListColor();
}

void ImportSismageHorizonWidget::updateListColor() {
	m_preferedDatasetWorker->stop();

	for (int i=0; i<m_listWidget->count(); i++) {
		QListWidgetItem* item = m_listWidget->item(i);
		item->setForeground(Qt::white);
	}

	emit searchPreferedDatasetSignal(this);
}

QString ImportSismageHorizonWidget::getSavePath(const QString& surveyPath, const QString& datasetPath,
		const QString& horizonName, const QString& horizonVersion) const {
	QString horizonPath = QString::fromStdString(NextVisionDBManager::surveyPath2HorizonDir(surveyPath.toStdString()));
	std::string taskDatasetName = SismageDBManager::datasetNameFromDatasetPath(datasetPath.toStdString());
	QString horizonNvName = QString::fromStdString(NextVisionDBManager::getNextVisionNameFromSismageHorizonName(horizonName.toStdString(),
			horizonVersion.toStdString(), taskDatasetName));
	QString horizonDirPath = QDir(horizonPath).absoluteFilePath(horizonNvName);
	return horizonDirPath;
}

void ImportSismageHorizonWidget::valuesLoaded(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker *worker, QList<float> summary) {
	QMutexLocker locker(&m_tasksMutex);

	auto it = std::find_if(m_onGoingTasks.begin(), m_onGoingTasks.end(), [worker](auto pair) {
		return pair.second.worker==worker;
	});

	if (it==m_onGoingTasks.end() || worker->response.size()%4!=0) {
		return;
	}

	float* tab = static_cast<float*>(static_cast<void*>(worker->response.data()));
	float* endTab = tab + worker->response.size() / 4;
	QList<float> isochronValues(tab, endTab);

	QString horizonDirPath;
	if (!it->second.savePath.isNull() && !it->second.savePath.isEmpty()) {
		horizonDirPath = it->second.savePath;
	} else {
		horizonDirPath = getSavePath(it->second.surveyPath, it->second.datasetPath,
				it->second.horizon->horizon.getRealName(), it->second.horizon->horizon.getVersion());
	}
	QPair<bool, QStringList> createdDirs = mkpath(horizonDirPath);
	bool success = createdDirs.first && SeaUtils::writeIsochronResponseToFile(isochronValues, it->second.isochron,
				it->second.datasetPath, horizonDirPath, it->second.isBricked);

	if (success && it->second.horizon->horizon.is_color_Set()) {
		m_importedHorizonPath.push_back(horizonDirPath);

		QString colorStr = it->second.horizon->horizon.getColor();
		if (QColor::isValidColor(colorStr)) {
			QColor color(colorStr);
			FreeHorizonQManager::saveColorToPath(horizonDirPath, color);
		}
	}

	if (success) {
		QString horizonNvName = QDir(horizonDirPath).dirName();
		emit horizonExtracted(horizonNvName, it->second.datasetPath);

		m_doneTasks[it->first] = it->second;
		m_onGoingTasks.erase(it);
	} else {
		rmpath(createdDirs.second);
		m_errorTasks[it->first] = it->second;
		m_onGoingTasks.erase(it);
	}

	postProcessTasks();
}

void ImportSismageHorizonWidget::valuesLoadedWithError(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker *worker,
		QNetworkReply::NetworkError error_type, QString error_str) {
	QMutexLocker locker(&m_tasksMutex);

	auto it = std::find_if(m_onGoingTasks.begin(), m_onGoingTasks.end(), [worker](auto pair) {
		return pair.second.worker==worker;
	});

	if (it!=m_onGoingTasks.end()) {
		m_errorTasks[it->first] = it->second;
		m_onGoingTasks.erase(it);
	}

	postProcessTasks();
}

FindPreferedDatasetsWorker::~FindPreferedDatasetsWorker() {

}

void FindPreferedDatasetsWorker::doWork(ImportSismageHorizonWidget* gridProvider) {
	m_stop = false;

	const std::map<QString, std::shared_ptr<ImportSismageHorizonWidget::Horizon>> horizons = gridProvider->horizons();

	auto it = horizons.cbegin();
	while (!m_stop && it!=horizons.cend()) {
		std::pair<bool, QString> importedResult = gridProvider->isHorizonImported(*(it->second));
		if (!m_stop && importedResult.first) {
			emit preferedDatasetFound(it->first, importedResult.second);
		}

		it++;
	}
	if (m_stop) {
		emit workInterrupted();
	} else {
		emit workFinished();
	}
}

void FindPreferedDatasetsWorker::stop() {
	m_stop = true;
}
