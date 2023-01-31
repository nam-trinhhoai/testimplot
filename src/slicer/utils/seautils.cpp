#include "seautils.h"
#include "ioutil.h"
#include "servermanager.h"
#include "freeHorizonManager.h"
#include "freeHorizonQManager.h"
#include "ProjectManagerNames.h"
#include "sismagedbmanager.h"
#include "smsurvey3D.h"
#include "util_filesystem.h"
#include <SeismicManager.h>

#include <QDebug>
#include <QFileInfo>
#include <QMutex>
#include <QMutexLocker>
#include <QNetworkAccessManager>

#include <fstream>
#include <limits>
#include <iomanip>

SeaUtils::ServersHolder::ServersHolder(const QString& dirProject, QObject* parent) : QObject(parent) {
	m_seismicApi = nullptr;
	m_interpretationApi = nullptr;
	m_serversKey = ServerManager::INVALID_SERVERS_KEY;
	m_dirProject = dirProject;
	startServer();
}

SeaUtils::ServersHolder::~ServersHolder() {
	stopServer();
}

bool SeaUtils::ServersHolder::isValid() const {
	return m_serversKey!=ServerManager::INVALID_SERVERS_KEY;
}

QString SeaUtils::ServersHolder::dirProject() const {
	return m_dirProject;
}

void SeaUtils::ServersHolder::setDirProject(const QString& path) {
	bool changeDirProject = m_dirProject.compare(path)!=0;
	if (isValid() && changeDirProject) {
		stopServer();
	}
	if (changeDirProject) {
		m_dirProject = path;
		startServer();
	}
}

OpenAPI::Interpretation::OAIInterpretationDefaultApi* SeaUtils::ServersHolder::interpretationApi() {
	if (m_interpretationApi==nullptr && isValid()) {
		ServerManager& manager = ServerManager::getServerManager();
		SeaModules* server = manager.server(m_dirProject);
		m_interpretationApi = new OpenAPI::Interpretation::OAIInterpretationDefaultApi();
		m_interpretationApi->setParent(this);
		QNetworkAccessManager* networkManager = new QNetworkAccessManager(m_interpretationApi);
		networkManager->connectToHostEncrypted(server->interpretationAddress(), server->interpretationPort());
		m_interpretationApi->setNetworkAccessManager(networkManager);
		m_interpretationApi->setNewServerForAllOperations(server->interpretationAddressWithPort());
	}
	return m_interpretationApi;
}

OpenAPI::Seismic::OAISeismicDefaultApi* SeaUtils::ServersHolder::seismicApi() {
	if (m_seismicApi==nullptr && isValid()) {
		ServerManager& manager = ServerManager::getServerManager();
		SeaModules* server = manager.server(m_dirProject);
		m_seismicApi = new OpenAPI::Seismic::OAISeismicDefaultApi();
		m_seismicApi->setParent(this);
		QNetworkAccessManager* networkManager = new QNetworkAccessManager(m_seismicApi);
		networkManager->connectToHostEncrypted(server->seismicAddress(), server->seismicPort());
		m_seismicApi->setNetworkAccessManager(networkManager);
		m_seismicApi->setNewServerForAllOperations(server->seismicAddressWithPort());
	}
	return m_seismicApi;
}

void SeaUtils::ServersHolder::startServer() {
	if (!isValid()) {
		QFileInfo fileInfo(m_dirProject);
		if (!fileInfo.exists()) {
			qDebug() << "ServersHolder::startServer dir project does not exists : " << m_dirProject;
		} else if (!fileInfo.isDir()) {
			qDebug() << "ServersHolder::startServer dir project is not a directory : " << m_dirProject;
		} else {
			ServerManager& manager = ServerManager::getServerManager();
			m_serversKey = manager.startServer(m_dirProject);
			if (!isValid()) {
				qDebug() << "ServersHolder::startServer failed to start server for dir project : " << m_dirProject;
			}
		}
	}
}

void SeaUtils::ServersHolder::stopServer() {
	if (isValid()) {
		if (m_seismicApi) {
			m_seismicApi->deleteLater();
			m_seismicApi = nullptr;
		}
		if (m_interpretationApi) {
			m_interpretationApi->deleteLater();
			m_interpretationApi = nullptr;
		}

		ServerManager& manager = ServerManager::getServerManager();
		bool success = manager.stopServer(m_dirProject, m_serversKey);
		if (!success) {
			qDebug() << "ServersHolder::stopServer failed to stop server";
		}
		m_serversKey = ServerManager::INVALID_SERVERS_KEY;
	}
}


/**
 * createSurveyTopoFile
 * 
 * This function use SEA to get information from the survey and create the file to parameter survey transforms
 *
 * WARNING : Currently the TOPO points are return is the survey geodesy, this is not the expect behavior
 * This code is awaiting an update in SEA to retrieve points in the project geodesy
 **/
bool SeaUtils::createSurveyTopoFile(const QString& _surveyPath) {
	// to remove "/" at the end
	QString surveyPath = QFileInfo(_surveyPath).absoluteFilePath();
	bool cutEnd = true;
	while (cutEnd) {
		int pos = surveyPath.lastIndexOf("/");
		cutEnd = pos>0 && pos==surveyPath.size()-1;
		if (cutEnd) {
			surveyPath = surveyPath.first(pos);
		}
	}

	ServerManager& manager = ServerManager::getServerManager();

	QString projectPath = QString::fromStdString(SismageDBManager::projectPathFromSurveyPath(surveyPath.toStdString()));
	QString dirProject = QString::fromStdString(SismageDBManager::dirProjectPathFromProjectPath(projectPath.toStdString()));

	long serversKey = manager.startServer(dirProject);
	bool startedServer = serversKey!=ServerManager::INVALID_SERVERS_KEY;
	bool valid = startedServer;
	if (!valid) {
		qDebug() << "SeaUtils::createSurveyTopoFile failed to start server";
	}


	// search survey desc file
	QString surveyDescFile;
	if (valid) {
		QDir dir(surveyPath);
		QFileInfoList list = dir.entryInfoList(QStringList() << "*.desc", QDir::Files);
		if (list.size()>0) {
			surveyDescFile = list[0].absoluteFilePath();
		}
	}

	// do request
	if (valid) {
		QString projectName = QFileInfo(projectPath).fileName();
		QString surveyName = ProjectManagerNames::getKeyFromFilename(surveyDescFile, "name=");
		if (surveyName.isNull() || surveyName.isEmpty()) {
			surveyName = QFileInfo(surveyPath).fileName();
		}

		SeaModules* server = manager.server(dirProject);
		valid = server!=nullptr;

		// example : https://github.com/OpenAPITools/openapi-generator/issues/6682
		if (valid) {
			OpenAPI::Seismic::OAISeismicDefaultApi api;
			QNetworkAccessManager* networkManager = new QNetworkAccessManager(&api);
			networkManager->connectToHostEncrypted(server->seismicAddress(), server->seismicPort());
			api.setNetworkAccessManager(networkManager);
			api.setNewServerForAllOperations(server->seismicAddressWithPort());

			Survey3DSWaiter survey3DSWaiter;
			QObject::connect(&api, &OpenAPI::Seismic::OAISeismicDefaultApi::getSurvey3DSSignalEFull, &survey3DSWaiter,
					&Survey3DSWaiter::getSurvey3DSSignalEFull);
			QObject::connect(&api, &OpenAPI::Seismic::OAISeismicDefaultApi::getSurvey3DSSignalFull, &survey3DSWaiter,
					&Survey3DSWaiter::getSurvey3DSSignalFull);
			survey3DSWaiter.setExpectedWorker(api.getSurvey3DS(projectName));
			survey3DSWaiter.waitForFinished();
			QObject::disconnect(&api, &OpenAPI::Seismic::OAISeismicDefaultApi::getSurvey3DSSignalEFull, &survey3DSWaiter,
								&Survey3DSWaiter::getSurvey3DSSignalEFull);
			QObject::disconnect(&api, &OpenAPI::Seismic::OAISeismicDefaultApi::getSurvey3DSSignalFull, &survey3DSWaiter,
					&Survey3DSWaiter::getSurvey3DSSignalFull);

			const QList<OpenAPI::Seismic::OAISeismicSurvey3D>& surveys = survey3DSWaiter.surveys();
			valid = false;
			int idxSurvey = 0;
			while (!valid && idxSurvey<surveys.count()) {
				valid = surveyName.compare(surveys[idxSurvey].getName())==0;
				if (!valid) {
					idxSurvey++;
				}
			}

			if (valid && (!surveys[idxSurvey].is_key_Set() || !surveys[idxSurvey].is_key_Valid())) {
				qDebug() << "SeaUtils::createSurveyTopoFile matching survey does not have a valid key";
				valid = false;
			}

			valid = valid && surveys[idxSurvey].getDimensions().isSet() &&
					surveys[idxSurvey].getDimensions().is_number_of_inlines_Set() &&
					surveys[idxSurvey].getDimensions().is_number_of_inlines_Valid() &&
					surveys[idxSurvey].getDimensions().is_number_of_crosslines_Set() &&
					surveys[idxSurvey].getDimensions().is_number_of_crosslines_Valid();
			if (valid) {
				OpenAPI::Seismic::OAISeismicSurvey3D survey = surveys[idxSurvey];
				OpenAPI::Seismic::OAISeismicSurvey3DDimensions dims = survey.getDimensions();

				double numTraces = dims.getNumberOfCrosslines();
				double numProfils = dims.getNumberOfInlines();

				QList<OpenAPI::Seismic::OAISeismicPoint> pointsConvert = initPoints(numTraces, numProfils);
				OpenAPI::Seismic::OAISeismicConversionInput input;
				input.setPoints(pointsConvert);
				OpenAPI::Seismic::OAISeismicPointType inType;
				inType.setValue(OpenAPI::Seismic::OAISeismicPointType::eOAISeismicPointType::BLOC);
				OpenAPI::Seismic::OAISeismicPointType outType;
				outType.setValue(OpenAPI::Seismic::OAISeismicPointType::eOAISeismicPointType::EARTH);

				input.setTypeIn(inType);
				input.setTypeOut(outType);

				ConvertWaiter convertWaiter;
				QObject::connect(&api, &OpenAPI::Seismic::OAISeismicDefaultApi::convertPointInSurvey3DSignalEFull, &convertWaiter,
						&ConvertWaiter::survey3dsSurvey3dIdConvertPutSignalEFull);
				QObject::connect(&api, &OpenAPI::Seismic::OAISeismicDefaultApi::convertPointInSurvey3DSignalFull, &convertWaiter,
						&ConvertWaiter::survey3dsSurvey3dIdConvertPutSignalFull);
				convertWaiter.setExpectedWorker(api.convertPointInSurvey3D(survey.getKey(), input));
				convertWaiter.waitForFinished();
				const QList<OpenAPI::Seismic::OAISeismicPoint> resultEarthPoints = convertWaiter.points();

				outType.setValue(OpenAPI::Seismic::OAISeismicPointType::eOAISeismicPointType::TOPO);
				input.setTypeOut(outType);

				convertWaiter.setExpectedWorker(api.convertPointInSurvey3D(survey.getKey(), input));
				convertWaiter.waitForFinished();
				const QList<OpenAPI::Seismic::OAISeismicPoint> resultTopoPoints = convertWaiter.points();
				QObject::disconnect(&api, &OpenAPI::Seismic::OAISeismicDefaultApi::convertPointInSurvey3DSignalEFull, &convertWaiter,
						&ConvertWaiter::survey3dsSurvey3dIdConvertPutSignalEFull);
				QObject::disconnect(&api, &OpenAPI::Seismic::OAISeismicDefaultApi::convertPointInSurvey3DSignalFull, &convertWaiter,
						&ConvertWaiter::survey3dsSurvey3dIdConvertPutSignalFull);

				QString ijkPath = QString::fromStdString(SismageDBManager::topoExchangePathFromSurveyPath(surveyPath.toStdString()));
				writeFile(ijkPath, pointsConvert, resultTopoPoints, resultEarthPoints);
			}

		} else {
			qDebug() << "SeaUtils::createSurveyTopoFile Invalid server but it should exists";
		}
	}

	// stop server if started by function
	if (startedServer) {
		bool stopStatus = manager.stopServer(dirProject, serversKey);
		if (!stopStatus) {
			qDebug() << "SeaUtils::createSurveyTopoFile failed to stop started server";
		}
	}

	return valid;
}

QList<OpenAPI::Seismic::OAISeismicPoint> SeaUtils::initPoints(int numTraces, int numProfils) {
	QList<OpenAPI::Seismic::OAISeismicPoint> out;

	// "xline inline" or "traces profils"
	OpenAPI::Seismic::OAISeismicPoint pt;
	pt.setX(0);
	pt.setY(0);
	pt.setZ(0);
	out << pt;
	pt.setX(0);
	pt.setY(numTraces-1);
	pt.setZ(0);
	out << pt;
	pt.setX(numProfils-1);
	pt.setY(0);
	pt.setZ(0);
	out << pt;
	pt.setX(numProfils-1);
	pt.setY(numTraces-1);
	pt.setZ(0);
	out << pt;

	return out;
}

bool SeaUtils::writeFile(const QString& ijkFilePath, QList<OpenAPI::Seismic::OAISeismicPoint> blocPoints,
		QList<OpenAPI::Seismic::OAISeismicPoint> topoPoints, QList<OpenAPI::Seismic::OAISeismicPoint> earthPoints) {
	bool valid = blocPoints.count()==topoPoints.count() && blocPoints.count()==earthPoints.count();

	if (valid) {
		QDir fileParentDir = QFileInfo(ijkFilePath).dir();
		QPair<bool, QStringList> res = mkpath(fileParentDir.absolutePath());
		valid = res.first;
	}

	if (valid) {
		std::ofstream file(ijkFilePath.toStdString());
		valid = file.is_open();

		if (valid) {
			file << "# NextVision Survey Corners V1.0" << std::endl;
			file << "# InlineIndex CrosslineIndex InlineNumber CrosslineNumber X Y" << std::endl;

			// save with maximum precision, did not found how to
			file << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
			for (int i=0; i<blocPoints.count(); i++) {
				// use 0 as default because server does not send value if the value is 0
				double blocX = blocPoints[i].getX();
				if (!blocPoints[i].is_x_Set()) {
					blocX = 0;
				}
				double blocY = blocPoints[i].getY();
				if (!blocPoints[i].is_y_Set()) {
					blocY = 0;
				}
				double earthX = earthPoints[i].getX();
				if (!earthPoints[i].is_x_Set()) {
					earthX = 0;
				}
				double earthY = earthPoints[i].getY();
				if (!earthPoints[i].is_y_Set()) {
					earthY = 0;
				}
				double topoX = topoPoints[i].getX();
				if (!topoPoints[i].is_x_Set()) {
					topoX = 0;
				}
				double topoY = topoPoints[i].getY();
				if (!topoPoints[i].is_y_Set()) {
					topoY = 0;
				}
				file << blocX << " " << blocY << " " <<
						earthX << " " << earthY << " " <<
						topoX << " " << topoY << std::endl;
			}
		}
	}

	return valid;
}

bool SeaUtils::writeHorizonToFile(const QString& datasetPath, const QString& horizonName, const QString& isochronName,
		const QString& outputPath) {
	QString _surveyPath = QString::fromStdString(SismageDBManager::survey3DPathFromDatasetPath(datasetPath.toStdString()));

	// to remove "/" at the end
	QString surveyPath = QFileInfo(_surveyPath).absoluteFilePath();
	bool cutEnd = true;
	while (cutEnd) {
		int pos = surveyPath.lastIndexOf("/");
		cutEnd = pos>0 && pos==surveyPath.size()-1;
		if (cutEnd) {
			surveyPath = surveyPath.first(pos);
		}
	}

	ServerManager& manager = ServerManager::getServerManager();

	QString projectPath = QString::fromStdString(SismageDBManager::projectPathFromSurveyPath(surveyPath.toStdString()));
	QString dirProject = QString::fromStdString(SismageDBManager::dirProjectPathFromProjectPath(projectPath.toStdString()));

	long serversKey = manager.startServer(dirProject);
	ServersHolder holder(dirProject);
	bool startedServer = serversKey!=ServerManager::INVALID_SERVERS_KEY;
	bool valid = startedServer;
	if (!valid) {
		qDebug() << "SeaUtils::createSurveyTopoFile failed to start server";
	}

	QString surveyDescFile;
	if (valid) {
		QDir dir(surveyPath);
		QFileInfoList list = dir.entryInfoList(QStringList() << "*.desc", QDir::Files);
		if (list.size()>0) {
			surveyDescFile = list[0].absoluteFilePath();
		}
	}

	// do request
	if (valid) {
		SeaModules* server = manager.server(dirProject);
		QString projectName = QFileInfo(projectPath).fileName();
		QString surveyName = ProjectManagerNames::getKeyFromFilename(surveyDescFile, "name=");
		if (surveyName.isNull() || surveyName.isEmpty()) {
			surveyName = QFileInfo(surveyPath).fileName();
		}

		OpenAPI::Interpretation::OAIInterpretationDefaultApi api;
		QNetworkAccessManager* networkManager = new QNetworkAccessManager(&api);
		networkManager->connectToHostEncrypted(server->interpretationAddress(), server->interpretationPort());
		api.setNetworkAccessManager(networkManager);
		api.setNewServerForAllOperations(server->interpretationAddressWithPort());

		const QList<OpenAPI::Interpretation::OAIInterpretationHorizon>& horizons = SeaUtils::getHorizons(api, projectName);
		valid = false;
		int idxHorizon = 0;
		while (!valid && idxHorizon<horizons.count()) {
			valid = horizonName.compare(horizons[idxHorizon].getName())==0;
			if (!valid) {
				idxHorizon++;
			}
		}

		if (valid && (!horizons[idxHorizon].is_id_Set() || !horizons[idxHorizon].is_id_Valid())) {
			qDebug() << "SeaUtils::writeHorizontoFile matching horizon does not have a valid id";
			valid = false;
		}

		int idxIsochron = 0;
		QList<OpenAPI::Interpretation::OAIInterpretationHorizonProperty> isochrons;
		if (valid) {
			isochrons = SeaUtils::getIsochrons(api, horizons[idxHorizon].getId());
			valid = false;
			while (!valid && idxIsochron<isochrons.count()) {
				valid = isochronName.compare(isochrons[idxIsochron].getName())==0;
				if (!valid) {
					idxIsochron++;
				}
			}

			if (valid && (!isochrons[idxIsochron].is_id_Set() || !isochrons[idxIsochron].is_id_Valid() ||
					!isochrons[idxIsochron].is_number_of_crosslines_Valid() || !isochrons[idxIsochron].is_number_of_inlines_Valid() ||
					!isochrons[idxIsochron].is_brick_size_inline_Valid() || !isochrons[idxIsochron].is_brick_size_crossline_Valid() ||
					!isochrons[idxIsochron].is_survey_id_Valid())) {
				qDebug() << "SeaUtils::writeHorizontoFile matching isochron does not have a valid id";
				valid = false;
			}
		}

		if (valid) {
			OpenAPI::Seismic::OAISeismicDefaultApi apiSeismic;
			QNetworkAccessManager* networkManagerSeismic = new QNetworkAccessManager(&apiSeismic);
			networkManagerSeismic->connectToHostEncrypted(server->seismicAddress(), server->seismicPort());
			apiSeismic.setNetworkAccessManager(networkManagerSeismic);
			apiSeismic.setNewServerForAllOperations(server->seismicAddressWithPort());

			OpenAPI::Seismic::OAISeismicSurvey3D surveyObj = SeaUtils::getSurveyFromId(apiSeismic, isochrons[idxIsochron].getSurveyId());

			valid = surveyObj.is_name_Valid() && surveyName.compare(surveyObj.getName())==0;
		}

		if (valid) {
			QList<float> isochronValues = SeaUtils::getIsochronValues(api, isochrons[idxIsochron].getId());

			valid = writeIsochronResponseToFile(isochronValues, isochrons[idxIsochron], datasetPath, outputPath);
		}
	}

	// stop server if started by function
	if (startedServer) {
		bool stopStatus = manager.stopServer(dirProject, serversKey);
		if (!stopStatus) {
			qDebug() << "SeaUtils::writeHorizontoFile failed to stop started server";
		}
	}

	return valid;
}
bool SeaUtils::writeIsochronResponseToFile(QList<float> isochronValues, OpenAPI::Interpretation::OAIInterpretationHorizonProperty isochron,
		const QString& datasetPath, const QString& outputPath, bool isBricked) {
	long xLinesCount = isochron.getNumberOfCrosslines();
	long inlinesCount = isochron.getNumberOfInlines();
	long xLineBlockSize = isochron.getBrickSizeCrossline();
	long inlineBlockSize = isochron.getBrickSizeInline();
	long N = xLinesCount * inlinesCount;

	bool valid = isochronValues.size()==N;
	if (!valid) {
		return valid;
	}

	std::vector<float> buf;
	buf.resize(N);

	if (isBricked) {
		// bricked
		valid = brick2DToContinuousData(isochronValues.cbegin(), isochronValues.cend(), buf.begin(), buf.end(),
				xLineBlockSize, inlineBlockSize, xLinesCount, inlinesCount);
		if (!valid) {
			return valid;
		}
	} else {
		// continuous
		for (long i=0; i<N; i++) {
			buf[i] = isochronValues[i];
		}
	}

	switch_list_endianness_inplace(buf.data(), sizeof(float), buf.size());


	// change null value to -9999.0
	float newNonVal = -9999.0;
	if (isochron.is_null_value_Set() && isochron.getNullValue()!=newNonVal) {
		float oldNonVal = isochron.getNullValue();
		for (long i=0; i<N; i++) {
			if (buf[i]==oldNonVal) {
				buf[i] = newNonVal;
			}
		}
	}

	QString _surveyPath = QString::fromStdString(SismageDBManager::survey3DPathFromDatasetPath(datasetPath.toStdString()));
	QString surveyPath = QFileInfo(_surveyPath).absoluteFilePath();
	Grid2D grid2DHorizon = SeaUtils::getGridFromIsochronAndSurveyPath(isochron, surveyPath);
	Grid2D grid2DDataset = Grid2D::getMapGridFromDatasetPath(datasetPath.toStdString());

	std::vector<float> datasetBuffer;
	valid = FreeHorizonQManager::getCroppedBufferFromGrids(grid2DHorizon, grid2DDataset, buf.begin(),
				buf.end(), datasetBuffer);

	if (valid) {
		QString datasetName = SeismicManager::seismicFullFilenameToTinyName(datasetPath);
		std::string output = FreeHorizonManager::write(datasetPath.toStdString(), datasetName.toStdString(), outputPath.toStdString(), datasetBuffer.data());
		valid = output=="ok";
	}

	return valid;
}

Grid2D SeaUtils::getGridFromIsochronAndSurveyPath(const OpenAPI::Interpretation::OAIInterpretationHorizonProperty& isochron, const QString& surveyPath) {
	bool valid = isochron.isSet();
	if (!valid) {
		return Grid2D();
	}

	double startInline = 0;
	if (isochron.is_first_inline_Set()) {
		startInline = isochron.getFirstInline();
	}

	double startXLine = 0;
	if (isochron.is_first_crossline_Set()) {
		startXLine = isochron.getFirstCrossline();
	}

	long countInline = 0;
	if (isochron.is_number_of_inlines_Set()) {
		countInline = isochron.getNumberOfInlines();
	}

	long countXLine = 0;
	if (isochron.is_number_of_crosslines_Set()) {
		countXLine = isochron.getNumberOfCrosslines();
	}

	double stepInline = 1;
	if (isochron.is_last_inline_Set() && countInline>1) {
		stepInline = (isochron.getLastInline() - startInline) / (countInline - 1);
	}

	double stepXLine = 1;
	if (isochron.is_last_crossline_Set() && countXLine>1) {
		stepXLine = (isochron.getLastCrossline() - startXLine) / (countXLine - 1);
	}

	SampleUnit depthAxis = SampleUnit::NONE;
	if (isochron.is_kind_Set()) {
		if (isochron.getKind().compare("Time")==0) {
			depthAxis = SampleUnit::TIME;
		} else if (isochron.getKind().compare("Depth")==0) {
			depthAxis = SampleUnit::DEPTH;
		}
	}

	// get topo
	SmSurvey3D survey3D(surveyPath.toStdString());

	return Grid2D(startInline, startXLine, stepInline, stepXLine, countInline, countXLine,
			depthAxis, survey3D.inlineXlineToXYTransfo());
}

QList<OpenAPI::Interpretation::OAIInterpretationHorizon> SeaUtils::getHorizons(OpenAPI::Interpretation::OAIInterpretationDefaultApi& api,
		const QString& projectName) {
	HorizonsWaiter horizonsWaiter;
	QObject::connect(&api, &OpenAPI::Interpretation::OAIInterpretationDefaultApi::getHorizonsSignalEFull, &horizonsWaiter,
			&HorizonsWaiter::getHorizonsSignalEFull);
	QObject::connect(&api, &OpenAPI::Interpretation::OAIInterpretationDefaultApi::getHorizonsSignalFull, &horizonsWaiter,
			&HorizonsWaiter::getHorizonsSignalFull);
	horizonsWaiter.setExpectedWorker(api.getHorizons(projectName));
	horizonsWaiter.waitForFinished();
	QObject::disconnect(&api, &OpenAPI::Interpretation::OAIInterpretationDefaultApi::getHorizonsSignalEFull, &horizonsWaiter,
			&HorizonsWaiter::getHorizonsSignalEFull);
	QObject::disconnect(&api, &OpenAPI::Interpretation::OAIInterpretationDefaultApi::getHorizonsSignalFull, &horizonsWaiter,
			&HorizonsWaiter::getHorizonsSignalFull);

	return horizonsWaiter.horizons();
}

QList<OpenAPI::Interpretation::OAIInterpretationHorizonProperty> SeaUtils::getIsochrons(OpenAPI::Interpretation::OAIInterpretationDefaultApi& api,
		const QString& horizonId) {
	IsochronsWaiter isochronsWaiter;
	QObject::connect(&api, &OpenAPI::Interpretation::OAIInterpretationDefaultApi::getIsochronsSignalEFull, &isochronsWaiter,
			&IsochronsWaiter::getIsochronsSignalEFull);
	QObject::connect(&api, &OpenAPI::Interpretation::OAIInterpretationDefaultApi::getIsochronsSignalFull, &isochronsWaiter,
			&IsochronsWaiter::getIsochronsSignalFull);
	isochronsWaiter.setExpectedWorker(api.getIsochrons(horizonId));
	isochronsWaiter.waitForFinished();
	QObject::disconnect(&api, &OpenAPI::Interpretation::OAIInterpretationDefaultApi::getIsochronsSignalEFull, &isochronsWaiter,
			&IsochronsWaiter::getIsochronsSignalEFull);
	QObject::disconnect(&api, &OpenAPI::Interpretation::OAIInterpretationDefaultApi::getIsochronsSignalFull, &isochronsWaiter,
			&IsochronsWaiter::getIsochronsSignalFull);

	return isochronsWaiter.isochrons();
}

QList<float> SeaUtils::getIsochronValues(OpenAPI::Interpretation::OAIInterpretationDefaultApi& api, const QString& isochronId) {
	IsochronValuesWaiter isochronValuesWaiter;
	QObject::connect(&api, &OpenAPI::Interpretation::OAIInterpretationDefaultApi::getIsochronValuesSignalEFull, &isochronValuesWaiter,
			&IsochronValuesWaiter::getIsochronValuesSignalEFull);
	QObject::connect(&api, &OpenAPI::Interpretation::OAIInterpretationDefaultApi::getIsochronValuesSignalFull, &isochronValuesWaiter,
			&IsochronValuesWaiter::getIsochronValuesSignalFull);
	isochronValuesWaiter.setExpectedWorker(api.getIsochronValues(isochronId));
	isochronValuesWaiter.waitForFinished();
	QObject::disconnect(&api, &OpenAPI::Interpretation::OAIInterpretationDefaultApi::getIsochronValuesSignalEFull, &isochronValuesWaiter,
			&IsochronValuesWaiter::getIsochronValuesSignalEFull);
	QObject::disconnect(&api, &OpenAPI::Interpretation::OAIInterpretationDefaultApi::getIsochronValuesSignalFull, &isochronValuesWaiter,
			&IsochronValuesWaiter::getIsochronValuesSignalFull);

	return isochronValuesWaiter.values();
}

OpenAPI::Seismic::OAISeismicSurvey3D SeaUtils::getSurveyFromId(OpenAPI::Seismic::OAISeismicDefaultApi& api, const QString& surveyId) {
	Survey3DWaiter surveyWaiter;
	QObject::connect(&api, &OpenAPI::Seismic::OAISeismicDefaultApi::getSurvey3DSignalEFull, &surveyWaiter,
			&Survey3DWaiter::getSurvey3DSignalEFull);
	QObject::connect(&api, &OpenAPI::Seismic::OAISeismicDefaultApi::getSurvey3DSignalFull, &surveyWaiter,
			&Survey3DWaiter::getSurvey3DSignalFull);
	surveyWaiter.setExpectedWorker(api.getSurvey3D(surveyId));
	surveyWaiter.waitForFinished();
	QObject::disconnect(&api, &OpenAPI::Seismic::OAISeismicDefaultApi::getSurvey3DSignalEFull, &surveyWaiter,
			&Survey3DWaiter::getSurvey3DSignalEFull);
	QObject::disconnect(&api, &OpenAPI::Seismic::OAISeismicDefaultApi::getSurvey3DSignalFull, &surveyWaiter,
			&Survey3DWaiter::getSurvey3DSignalFull);

	return surveyWaiter.survey();
}

QList<OpenAPI::Seismic::OAISeismicSurvey3D> SeaUtils::getSurveys(OpenAPI::Seismic::OAISeismicDefaultApi& api,
		const QString& projectName) {
	Survey3DSWaiter surveysWaiter;
	QObject::connect(&api, &OpenAPI::Seismic::OAISeismicDefaultApi::getSurvey3DSSignalEFull, &surveysWaiter,
			&Survey3DSWaiter::getSurvey3DSSignalEFull);
	QObject::connect(&api, &OpenAPI::Seismic::OAISeismicDefaultApi::getSurvey3DSSignalFull, &surveysWaiter,
			&Survey3DSWaiter::getSurvey3DSSignalFull);
	surveysWaiter.setExpectedWorker(api.getSurvey3DS(projectName));
	surveysWaiter.waitForFinished();
	QObject::disconnect(&api, &OpenAPI::Seismic::OAISeismicDefaultApi::getSurvey3DSSignalEFull, &surveysWaiter,
			&Survey3DSWaiter::getSurvey3DSSignalEFull);
	QObject::disconnect(&api, &OpenAPI::Seismic::OAISeismicDefaultApi::getSurvey3DSSignalFull, &surveysWaiter,
			&Survey3DSWaiter::getSurvey3DSSignalFull);

	return surveysWaiter.surveys();
}

void Survey3DSWaiter::getSurvey3DSSignalFull(OpenAPI::Seismic::OAISeismicHttpRequestWorker *worker,
			QList<OpenAPI::Seismic::OAISeismicSurvey3D> summary) {
	if (m_expectedWorker!=worker) {
		return;
	}
	m_summary = summary;
	stop();
}

void Survey3DSWaiter::getSurvey3DSSignalEFull(OpenAPI::Seismic::OAISeismicHttpRequestWorker *worker,
			QNetworkReply::NetworkError error_type, QString error_str) {
	if (m_expectedWorker!=worker) {
		return;
	}
	stop();
}

void Survey3DWaiter::getSurvey3DSignalFull(OpenAPI::Seismic::OAISeismicHttpRequestWorker *worker,
		OpenAPI::Seismic::OAISeismicSurvey3D summary) {
	if (m_expectedWorker!=worker) {
		return;
	}
	m_summary = summary;
	stop();
}

void Survey3DWaiter::getSurvey3DSignalEFull(OpenAPI::Seismic::OAISeismicHttpRequestWorker *worker,
			QNetworkReply::NetworkError error_type, QString error_str) {
	if (m_expectedWorker!=worker) {
		return;
	}
	stop();
}

void HorizonsWaiter::getHorizonsSignalFull(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker *worker,
			QList<OpenAPI::Interpretation::OAIInterpretationHorizon> summary) {
	if (m_expectedWorker!=worker) {
		return;
	}
	m_summary = summary;
	stop();
}

void HorizonsWaiter::getHorizonsSignalEFull(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker *worker, QNetworkReply::NetworkError error_type,
			QString error_str) {
	if (m_expectedWorker!=worker) {
		return;
	}
	stop();
}

void IsochronsWaiter::getIsochronsSignalFull(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker *worker,
			QList<OpenAPI::Interpretation::OAIInterpretationHorizonProperty> summary) {
	if (m_expectedWorker!=worker) {
		return;
	}
	m_summary = summary;
	stop();
}

void IsochronsWaiter::getIsochronsSignalEFull(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker *worker, QNetworkReply::NetworkError error_type,
			QString error_str) {
	if (m_expectedWorker!=worker) {
		return;
	}
	stop();
}

void IsochronValuesWaiter::getIsochronValuesSignalFull(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker *worker, QList<float> summary) {
	if (m_expectedWorker!=worker) {
		return;
	}

	if (summary.size()>0 || worker->response.size()%4!=0) {
		m_summary = summary;
	} else {
		float* tab = static_cast<float*>(static_cast<void*>(worker->response.data()));
		float* endTab = tab + worker->response.size() / 4;
		m_summary = QList<float>(tab, endTab);
	}
	stop();
}

void IsochronValuesWaiter::getIsochronValuesSignalEFull(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker *worker, QNetworkReply::NetworkError error_type,
			QString error_str) {
	if (m_expectedWorker!=worker) {
		return;
	}

	//m_summary = summary;
	float* tab = static_cast<float*>(static_cast<void*>(worker->response.data()));
	float* endTab = tab + worker->response.size() / 4;
	m_summary = QList<float>(tab, endTab);
	stop();
}
