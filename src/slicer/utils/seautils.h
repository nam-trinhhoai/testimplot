#ifndef SRC_SLICER_UTILS_SEAUTILS_H
#define SRC_SLICER_UTILS_SEAUTILS_H

#include <QString>
#include <QList>
#include <QMutex>
#include <QMutexLocker>
#include <QEventLoop>

#include <SEA_Interpretation/OAIInterpretationDefaultApi.h>
#include <SEA_Interpretation/OAIInterpretationHttpRequest.h>
#include <SEA_Seismic/OAISeismicDefaultApi.h>

#include "grid2d.h"

class SeaUtils {
public:
	class ServersHolder : public QObject {
	public:
		ServersHolder(const QString& dirProject, QObject* parent=0);
		~ServersHolder();
		bool isValid() const;
		QString dirProject() const;
		void setDirProject(const QString& path);
		OpenAPI::Interpretation::OAIInterpretationDefaultApi* interpretationApi();
		OpenAPI::Seismic::OAISeismicDefaultApi* seismicApi();
	private:
		void startServer();
		void stopServer();

		long m_serversKey;
		QString m_dirProject;

		OpenAPI::Interpretation::OAIInterpretationDefaultApi* m_interpretationApi;
		OpenAPI::Seismic::OAISeismicDefaultApi* m_seismicApi;
	};

	/**
	 * Create TOPO3D.ijk file with SEA server
	 *
	 * Input is survey path, it will allow to deduce survey, project and dir_projet
	 */
	static bool createSurveyTopoFile(const QString& surveyPath);
	static QList<OpenAPI::Seismic::OAISeismicPoint> initPoints(int numTraces, int numProfils);
	static bool writeFile(const QString& ijkFilePath, QList<OpenAPI::Seismic::OAISeismicPoint> blocPoints,
			QList<OpenAPI::Seismic::OAISeismicPoint> topoPoints, QList<OpenAPI::Seismic::OAISeismicPoint> earthPoints);

	static bool writeHorizonToFile(const QString& datasetPath, const QString& horizonName, const QString& isochronName,
			const QString& outputPath);

	/**
	 * isochron should have the same survey that datasetPath
	 * isochronValues should be the full response (all bricks) from isochron values
	 */
	static bool writeIsochronResponseToFile(QList<float> isochronValues, OpenAPI::Interpretation::OAIInterpretationHorizonProperty isochron,
			const QString& datasetPath, const QString& outputPath, bool isBricked=true);

	static QList<OpenAPI::Interpretation::OAIInterpretationHorizon> getHorizons(OpenAPI::Interpretation::OAIInterpretationDefaultApi& api,
			const QString& projectName);
	static QList<OpenAPI::Interpretation::OAIInterpretationHorizonProperty> getIsochrons(OpenAPI::Interpretation::OAIInterpretationDefaultApi& api,
			const QString& horizonId);
	static QList<float> getIsochronValues(OpenAPI::Interpretation::OAIInterpretationDefaultApi& api, const QString& isochronId);
	static OpenAPI::Seismic::OAISeismicSurvey3D getSurveyFromId(OpenAPI::Seismic::OAISeismicDefaultApi& api,
			const QString& surveyId);
	static QList<OpenAPI::Seismic::OAISeismicSurvey3D> getSurveys(OpenAPI::Seismic::OAISeismicDefaultApi& api,
			const QString& projectName);
	/**
	 * Create a Grid2D object from isochron and use the provided survey path to get topo
	 *
	 * The survey path need to match the survey id in isochron
	 *
	 * For Extension : If the survey path cannot be provided, please use the SEA server to converts points using the survey defined in isochron.
	 * Then use these points to define a inlineXlintToXY transformation.
	 * To do that, use createSurveyTopoFile as an example and avoid code duplicates by creating base functions
	 */
	static Grid2D getGridFromIsochronAndSurveyPath(const OpenAPI::Interpretation::OAIInterpretationHorizonProperty& isochron, const QString& surveyPath);
};

class SignalWaiter {
public:
	SignalWaiter() {

	}
	virtual ~SignalWaiter() {
		QMutexLocker b1(&m_mutex); // to wait for end of waitForFinished
	}
	bool waitForFinished() {
		QMutexLocker b1(&m_mutex);
		m_eventLoop.exec();
		return true;
	}
protected:
	void stop() {
		m_eventLoop.quit();
	}

	QMutex m_mutex;
	QEventLoop m_eventLoop;
};

class Survey3DSWaiter : public QObject, public SignalWaiter {
	Q_OBJECT
public:
	Survey3DSWaiter(QObject* parent = nullptr) : SignalWaiter(), QObject(parent) {
		m_expectedWorker = nullptr;
	}
	void setExpectedWorker(OpenAPI::Seismic::OAISeismicHttpRequestWorker* worker) {
		m_expectedWorker = worker;
	}
	const QList<OpenAPI::Seismic::OAISeismicSurvey3D>& surveys() const {
		return m_summary;
	}

public slots:
	void getSurvey3DSSignalFull(OpenAPI::Seismic::OAISeismicHttpRequestWorker* worker,
			QList<OpenAPI::Seismic::OAISeismicSurvey3D> summary);
	void getSurvey3DSSignalEFull(OpenAPI::Seismic::OAISeismicHttpRequestWorker* worker, QNetworkReply::NetworkError error_type,
		QString error_str);
private:
	QList<OpenAPI::Seismic::OAISeismicSurvey3D> m_summary;
	OpenAPI::Seismic::OAISeismicHttpRequestWorker* m_expectedWorker;
};

class Survey3DWaiter : public QObject, public SignalWaiter {
	Q_OBJECT
public:
	Survey3DWaiter(QObject* parent = nullptr) : SignalWaiter(), QObject(parent) {
		m_expectedWorker = nullptr;
	}
	void setExpectedWorker(OpenAPI::Seismic::OAISeismicHttpRequestWorker* worker) {
		m_expectedWorker = worker;
	}
	const OpenAPI::Seismic::OAISeismicSurvey3D& survey() const {
		return m_summary;
	}

public slots:
	void getSurvey3DSignalFull(OpenAPI::Seismic::OAISeismicHttpRequestWorker* worker,
			OpenAPI::Seismic::OAISeismicSurvey3D summary);
	void getSurvey3DSignalEFull(OpenAPI::Seismic::OAISeismicHttpRequestWorker* worker, QNetworkReply::NetworkError error_type,
		QString error_str);
private:
	OpenAPI::Seismic::OAISeismicSurvey3D m_summary;
	OpenAPI::Seismic::OAISeismicHttpRequestWorker* m_expectedWorker;
};

class ConvertWaiter : public QObject, public SignalWaiter {
	Q_OBJECT
public:
	ConvertWaiter(QObject* parent = nullptr) : SignalWaiter(), QObject(parent) {
		m_expectedWorker = nullptr;
	}
	void setExpectedWorker(OpenAPI::Seismic::OAISeismicHttpRequestWorker* worker) {
		m_expectedWorker = worker;
	}
	const QList<OpenAPI::Seismic::OAISeismicPoint>& points() const {
		return m_summary;
	}

public slots:
	void survey3dsSurvey3dIdConvertPutSignalFull(OpenAPI::Seismic::OAISeismicHttpRequestWorker* worker,
			QList<OpenAPI::Seismic::OAISeismicPoint> summary) {
		if (m_expectedWorker!=worker) {
			return;
		}
		m_summary = summary;
		stop();
	}
	void survey3dsSurvey3dIdConvertPutSignalEFull(OpenAPI::Seismic::OAISeismicHttpRequestWorker* worker, QNetworkReply::NetworkError error_type,
			QString error_str) {
		if (m_expectedWorker!=worker) {
			return;
		}
		stop();
	}
private:
	QList<OpenAPI::Seismic::OAISeismicPoint> m_summary;
	OpenAPI::Seismic::OAISeismicHttpRequestWorker* m_expectedWorker;
};

class HorizonsWaiter : public QObject, public SignalWaiter {
	Q_OBJECT
public:
	HorizonsWaiter(QObject* parent = nullptr) : SignalWaiter(), QObject(parent) {
		m_expectedWorker = nullptr;
	}
	void setExpectedWorker(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker* worker) {
		m_expectedWorker = worker;
	}
	const QList<OpenAPI::Interpretation::OAIInterpretationHorizon>& horizons() const {
		return m_summary;
	}

public slots:
	void getHorizonsSignalFull(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker* worker,
			QList<OpenAPI::Interpretation::OAIInterpretationHorizon> summary);
	void getHorizonsSignalEFull(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker* worker, QNetworkReply::NetworkError error_type,
		QString error_str);
private:
	QList<OpenAPI::Interpretation::OAIInterpretationHorizon> m_summary;
	OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker* m_expectedWorker;
};

class IsochronsWaiter : public QObject, public SignalWaiter {
	Q_OBJECT
public:
	IsochronsWaiter(QObject* parent = nullptr) : SignalWaiter(), QObject(parent) {
		m_expectedWorker = nullptr;
	}
	void setExpectedWorker(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker* worker) {
		m_expectedWorker = worker;
	}
	const QList<OpenAPI::Interpretation::OAIInterpretationHorizonProperty>& isochrons() const {
		return m_summary;
	}

public slots:
	void getIsochronsSignalFull(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker *worker,
			QList<OpenAPI::Interpretation::OAIInterpretationHorizonProperty> summary);
	void getIsochronsSignalEFull(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker *worker, QNetworkReply::NetworkError error_type,
		QString error_str);
private:
	QList<OpenAPI::Interpretation::OAIInterpretationHorizonProperty> m_summary;
	OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker* m_expectedWorker;
};

class IsochronValuesWaiter : public QObject, public SignalWaiter {
	Q_OBJECT
public:
	IsochronValuesWaiter(QObject* parent = nullptr) : SignalWaiter(), QObject(parent) {
		m_expectedWorker = nullptr;
	}

	void setExpectedWorker(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker* worker) {
		m_expectedWorker = worker;
	}

	const QList<float>& values() const {
		return m_summary;
	}

public slots:
	void getIsochronValuesSignalFull(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker *worker, QList<float> summary);
	void getIsochronValuesSignalEFull(OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker *worker, QNetworkReply::NetworkError error_type,
		QString error_str);
private:
	QList<float> m_summary;
	OpenAPI::Interpretation::OAIInterpretationHttpRequestWorker* m_expectedWorker;
};

#endif
