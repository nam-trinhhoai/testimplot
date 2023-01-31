#ifndef SLICER_CONFIG_H_
#define SLICER_CONFIG_H_

#include <memory>

#include <QString>
#include <QStringList>
#include <vector>
#include <utility>

class GlobalConfig {
public:
	static GlobalConfig& getConfig();

	~GlobalConfig();

	double wheelZoomBaseFactor() const;
	double wheelZoomExponentFactor() const;

	QString sessionPath(); // get session path or build it with prefix if session path is empty
	QString sessionPathPrefix() const;
	void setSessionPath(QString newPath);
	void setSessionPathPrefix(QString newPathPrefix); // set prefix to build session path if session path empty. <!> does not build session path

	const std::vector<std::pair<QString, QString>>& dirProjects();
	void setDirProjects(const std::vector<std::pair<QString, QString>>& newDirProject);
	QString databasePath() const;
	void setDatabasePath(QString newPath);

	QString deleteLogsPath() const;
	void setDeleteLogsPath(QString newPath);
	void updateDeleteLogsPathFromDataBasePath();
	QString getDeleteLogPathFromProjectPath(QString projectPath) const;

	QString logoPath() const;

	QString bnniProgramLocation() const;
	void setBnniProgramLocation(const QString& location);
	QString bnniInterfaceProgramLocation() const;
	void setBnniInterfaceProgramLocation(const QString& location);

	QStringList customKnownHostsFiles() const;
	void setCustomKnownHostsFiles(const QStringList& knownHostsFiles);

	QString geneticPlotDirPath() const;
	void setGeneticPlotDirPath(const QString& geneticPlotDirPath);
	QString geneticShiftDirPath() const;
	void setGeneticShiftDirPath(const QString& geneticShiftDirPath);

	QString tempDirPath() const;
	void setTempDirPath(const QString& tempDirPath);

	QString fileExplorerProgram() const;
	void setFileExplorerProgram(const QString& file);

private:
	GlobalConfig();

	void setApplicationPath(QString appPath);

	double m_wheelZoomBaseFactor = 0.9;
	double m_wheelZoomExponentFactor = 0.0025;
	QString m_logoSuffix = "logoIcon.png";
	QString m_logoPath;

	QString m_sessionPath;
	QString m_sessionPathPrefix;

	std::vector<std::pair<QString, QString>> m_dirProjects;
	QString m_databasePath;
	QString m_deleteLogsPath;
	bool m_isDeleteLogsPathSet;

	QString m_bnniProgramLocation;
	QString m_bnniInterfaceProgramLocation;
	QStringList m_customKnownHostsFiles;

	QString m_geneticPlotDirPath;
	QString m_geneticShiftDirPath;

	QString m_tempDirPath;
	QString m_fileExplorerProgram;

	static std::unique_ptr<GlobalConfig> config;
};

const QLatin1String LAST_SESSION_PATH_IN_SETTINGS("DataSelectorDialog/lastSessionPath");

#endif
