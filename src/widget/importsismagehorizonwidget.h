#ifndef SRC_WIDGET_IMPORTSISMAGEHORIZONWIDGET_H
#define SRC_WIDGET_IMPORTSISMAGEHORIZONWIDGET_H

#include "seautils.h"

#include <QMutex>
#include <QRecursiveMutex>
#include <QReadWriteLock>
#include <QThread>
#include <QWidget>

#include <SEA_Interpretation/OAIInterpretationHorizon.h>
#include <SEA_Interpretation/OAIInterpretationHorizonProperty.h>
#include <SEA_Seismic/OAISeismicSurvey3D.h>

#include <memory>

class QLineEdit;
class QListWidget;
class QPushButton;

class FindPreferedDatasetsWorker;

class ImportSismageHorizonWidget : public QWidget {
	Q_OBJECT
public:
	struct Horizon {
		OpenAPI::Interpretation::OAIInterpretationHorizon horizon;
		// key is survey id from SEA
		bool areIsochronsSet = false;
		std::map<QString, OpenAPI::Interpretation::OAIInterpretationHorizonProperty> isochrons;

		QString preferedDatasetPath;
	};

	struct Task {
		std::shared_ptr<Horizon> horizon;
		OpenAPI::Interpretation::OAIInterpretationHorizonProperty isochron;
		QString surveyPath;
		QString datasetPath;
		QString savePath;
		OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker* worker = nullptr;
		bool isBricked;
	};

	struct ApiConnections {
		//OpenAPI::Interpretation::OAIInterpretationDefaultApi* api;
		QMetaObject::Connection mainConnection;
		QMetaObject::Connection errorConnection;
		QMetaObject::Connection destroyedApiConnection;
	};

	ImportSismageHorizonWidget(QWidget* parent=nullptr);
	~ImportSismageHorizonWidget();

	QString projectPath() const;
	QString projectName() const;
	void setProjectPath(const QString& path);
	QString surveyPath() const;
	QString surveyName() const;
	void setSurveyPath(const QString& path); // will update both project and survey
	QString datasetPath() const;
	QString datasetName() const;
	void setDatasetPath(const QString& path); // will update project, survey and dataset

	std::pair<bool, QString> isHorizonImported(const Horizon& horizon);
	void setBadDatasetToolTip(const QString& msg);

	const std::map<QString, std::shared_ptr<Horizon>>& horizons() const;

public slots:
	void openItemMenu(const QPoint& pos);
	void extractHorizon(Task task);
	void importHorizons();
	void selectAllItems();
	void unselectAllItems();


	std::vector<QString> getImportedHorizonPath() { return m_importedHorizonPath; }

signals:
	void horizonExtracted(QString horizonName, QString datasetPath);
	void importFinished();
	void searchPreferedDatasetSignal(ImportSismageHorizonWidget* gridProvider);

private slots:
	// preferedDatasetFound function must be called in the widget thread
	void preferedDatasetFound(QString horizonId, QString preferedDataset);
	void valuesLoaded(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker *worker, QList<float> summary);
	void valuesLoadedWithError(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker *worker,
			QNetworkReply::NetworkError error_type, QString error_str);

private:
	QString askSavePath(const QString& oriSavePath,
			const OpenAPI::Interpretation::OAIInterpretationHorizon& horizon, bool& skip, bool& overwrite,
			bool& skipAll, bool& overwriteAll);
	void changeFilter(QString text);
	void clearFilter();
	void closeConnectionForApi(OpenAPI::Interpretation::OAIInterpretationDefaultApi* api);
	QString getSavePath(const QString& surveyPath, const QString& datasetPath,
			const QString& horizonName, const QString& horizonVersion) const;
	QString getCurrentSurveyId();
	Grid2D getGrid2DFromDatasetPath(const QString& datasetPath);
	void initServers();
	void loadHorizons();
	void loadIsochrons(Horizon& horizon);
	void loadSurveys();
	void postProcessTasks();
	void openConnectionForApi(OpenAPI::Interpretation::OAIInterpretationDefaultApi* api);
	void toggleBricked(int state);
	void setDatasetPathPrivate(const QString& path);
	void updateList();
	void updateListColor();
	bool validateOverwrite(const QString& horizonPath, bool& skipAll, bool& overwriteAll);

	// key is horizon id from SEA
	std::map<QString, std::shared_ptr<Horizon>> m_horizons;

	// key is survey id from SEA
	std::map<QString, OpenAPI::Seismic::OAISeismicSurvey3D> m_cacheSurveys;
	std::map<QString, Grid2D> m_cacheGrids;

	QReadWriteLock m_cacheGridsMutex;
	QMutex m_cacheSurveysMutex;
	QMutex m_initServersMutex;

	QListWidget* m_listWidget;
	QLineEdit* m_filterLineEdit;
	QPushButton* m_importButton;

	QString m_badDatasetToolTip;
	QString m_dirProjectPath;
	QString m_projectPath;
	QString m_surveyPath;
	QString m_datasetPath;
	QString m_currentSurveyId; // need to be updated to match surveyName
	QString m_listFilter;
	bool m_isBricked = true;

	QMutex m_importMutex;
	QRecursiveMutex m_tasksMutex;

	std::unique_ptr<SeaUtils::ServersHolder> m_serversHolder;

	// key is horizon id
	std::map<QString, Task> m_onGoingTasks;
	std::map<QString, Task> m_doneTasks;
	std::map<QString, Task> m_errorTasks;
	std::map<OpenAPI::Interpretation::OAIInterpretationDefaultApi*, ApiConnections> m_apiConnections;

	std::vector<QString> m_importedHorizonPath;

	QThread m_workerThread;
	FindPreferedDatasetsWorker* m_preferedDatasetWorker;
};

class FindPreferedDatasetsWorker : public QObject {
	Q_OBJECT
public:
	virtual ~FindPreferedDatasetsWorker();
public slots:
	void doWork(ImportSismageHorizonWidget* gridProvider);
	void stop();

signals:
	void preferedDatasetFound(QString horizonId, QString preferedDataset);
	void workFinished();
	void workInterrupted();

private:
	bool m_stop = false;
};

#endif // SRC_WIDGET_IMPORTSISMAGEHORIZONWIDGET_H
