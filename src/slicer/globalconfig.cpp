#include "globalconfig.h"

#include <QGuiApplication>
#include <QDir>
#include <QFileInfo>
#include <QProcess>

#include <boost/filesystem/path.hpp>

GlobalConfig::GlobalConfig() {
	setApplicationPath(QGuiApplication::applicationFilePath());

	m_sessionPathPrefix = "/data/PLI/NextVision/sessions/";
	m_sessionPath = "";
	m_databasePath = "/data/PLI/NextVision/projectmanagerDB2/";

	updateDeleteLogsPathFromDataBasePath();
	m_isDeleteLogsPathSet = false;

	boost::filesystem::path full_path(QGuiApplication::applicationFilePath().toStdString());
	QString rootScriptDir = QString::fromStdString(full_path.parent_path().parent_path().string() + "/scripts");
	m_bnniProgramLocation = rootScriptDir + "/BNNI/";
	m_bnniInterfaceProgramLocation = rootScriptDir+"/BNNI_Interface/";

	m_customKnownHostsFiles.push_back("/data/PLI/NextVision/BNNI/PAU/known_hosts");
	m_customKnownHostsFiles.push_back("/data/PLI/NextVision/BNNI/PARIS/known_hosts");

	m_geneticPlotDirPath = "/data/PLI/armand/plots/figures";
	m_geneticShiftDirPath = "/data/PLI/armand/plots/figures";

	m_tempDirPath = "/data/PLI/TMP";

	m_fileExplorerProgram = "caja";
}

GlobalConfig::~GlobalConfig() {}

double GlobalConfig::wheelZoomBaseFactor() const {
	return m_wheelZoomBaseFactor;
}

double GlobalConfig::wheelZoomExponentFactor() const {
	return m_wheelZoomExponentFactor;
}

std::unique_ptr<GlobalConfig> GlobalConfig::config;

GlobalConfig& GlobalConfig::getConfig() {
	if (GlobalConfig::config==nullptr) {
		GlobalConfig::config.reset(new GlobalConfig);
	}
	return *(GlobalConfig::config.get());
}

QString GlobalConfig::logoPath() const {
	return m_logoPath;
}

void GlobalConfig::setApplicationPath(QString appPath) {
	QFileInfo appFileInfo(appPath);
	QDir binDir = appFileInfo.dir();
	m_logoPath = binDir.absoluteFilePath(m_logoSuffix);
}

QString GlobalConfig::sessionPath() {
	if (m_sessionPath.isNull() || m_sessionPath.isEmpty()) {
		QProcess process;
		process.start("whoami");
		process.waitForFinished();

		QString defaultSessionPath;
		if (process.exitCode()==QProcess::NormalExit) {
			QString outputs(process.readAllStandardOutput());
			outputs = outputs.split("\n").first();
			defaultSessionPath = QString(m_sessionPathPrefix) + QString(outputs);
		} else {
			defaultSessionPath = m_sessionPathPrefix;
		}
		m_sessionPath = defaultSessionPath;
	}
	return m_sessionPath;
}

QString GlobalConfig::sessionPathPrefix() const {
	return m_sessionPathPrefix;
}

void GlobalConfig::setSessionPath(QString newPath) {
	m_sessionPath = newPath;
}

void GlobalConfig::setSessionPathPrefix(QString newPathPrefix) {
	m_sessionPathPrefix = newPathPrefix;
}

const std::vector<std::pair<QString, QString>>& GlobalConfig::dirProjects() {
	if (m_dirProjects.size()==0) {
		std::pair<QString, QString> rd("R&D", "/data/sismage/IMA3G/DIR_PROJET/");
		std::pair<QString, QString> prod("PROD", "/data/IMA3G/DIR_PROJET/");
		std::pair<QString, QString> prod_cph("PROD-CPH", "/data/IMA3G/DIR_PROJET_CPH/");
		std::pair<QString, QString> prod_paris("PROD-PARIS", "/data/IMA3G/DIR_PROJET_PARIS/");
		std::pair<QString, QString> prod_sg("PROD-SG", "/data/IMA3G/DIR_PROJET_SG/");
		std::pair<QString, QString> prod_uk("PROD-UK", "/data/IMA3G/DIR_PROJET_UK/");
		std::pair<QString, QString> pli("PLI", "/data/PLI/DIR_PROJET/");
		std::pair<QString, QString> other("Other", "/data/PLI/jacques/");
		m_dirProjects.push_back(rd);
		m_dirProjects.push_back(prod);
		m_dirProjects.push_back(prod_cph);
		m_dirProjects.push_back(prod_paris);
		m_dirProjects.push_back(prod_sg);
		m_dirProjects.push_back(prod_uk);
		m_dirProjects.push_back(pli);
		m_dirProjects.push_back(other);
	}
	return m_dirProjects;
}

void GlobalConfig::setDirProjects(const std::vector<std::pair<QString, QString>>& newDirProject) {
	m_dirProjects = newDirProject;
}

QString GlobalConfig::databasePath() const {
	return m_databasePath;
}

void GlobalConfig::setDatabasePath(QString newPath) {
	m_databasePath = newPath;
	if (!m_isDeleteLogsPathSet) {
		updateDeleteLogsPathFromDataBasePath();
	}
}

QString GlobalConfig::deleteLogsPath() const {
	return m_deleteLogsPath;
}

void GlobalConfig::setDeleteLogsPath(QString newPath) {
	m_deleteLogsPath = newPath;
	m_isDeleteLogsPathSet = true;
}

void GlobalConfig::updateDeleteLogsPathFromDataBasePath() {
	m_deleteLogsPath = m_databasePath;
	m_isDeleteLogsPathSet = false;
}

QString GlobalConfig::getDeleteLogPathFromProjectPath(QString projectPath) const {
	projectPath = QDir::cleanPath(projectPath);
	QDir dir(m_deleteLogsPath);
	QString fileName = "log_deletion_" + projectPath.split("/").join("_@_") + ".log";
	return dir.absoluteFilePath(fileName);
}

QString GlobalConfig::bnniProgramLocation() const {
	return m_bnniProgramLocation;
}

void GlobalConfig::setBnniProgramLocation(const QString& location) {
	m_bnniProgramLocation = location;
}

QString GlobalConfig::bnniInterfaceProgramLocation() const {
	return m_bnniInterfaceProgramLocation;
}

void GlobalConfig::setBnniInterfaceProgramLocation(const QString& location) {
	m_bnniInterfaceProgramLocation = location;
}

QStringList GlobalConfig::customKnownHostsFiles() const {
	return m_customKnownHostsFiles;
}

void GlobalConfig::setCustomKnownHostsFiles(const QStringList& knownHostsFiles) {
	m_customKnownHostsFiles = knownHostsFiles;
}

QString GlobalConfig::geneticPlotDirPath() const {
	return m_geneticPlotDirPath;
}

void GlobalConfig::setGeneticPlotDirPath(const QString& geneticPlotDirPath) {
	m_geneticPlotDirPath = geneticPlotDirPath;
}

QString GlobalConfig::geneticShiftDirPath() const {
	return m_geneticShiftDirPath;
}

void GlobalConfig::setGeneticShiftDirPath(const QString& geneticShiftDirPath) {
	m_geneticShiftDirPath = geneticShiftDirPath;
}

QString GlobalConfig::tempDirPath() const {
	return m_tempDirPath;
}

void GlobalConfig::setTempDirPath(const QString& tempDirPath) {
	m_tempDirPath = tempDirPath;
}

QString GlobalConfig::fileExplorerProgram() const {
	return m_fileExplorerProgram;
}

void GlobalConfig::setFileExplorerProgram(const QString& file) {
	m_fileExplorerProgram = file;
}
