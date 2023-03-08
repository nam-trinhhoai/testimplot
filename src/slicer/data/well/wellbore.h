#ifndef WellBore_H
#define WellBore_H

#include <QObject>
#include <QVector2D>
#include <QVector3D>
#include <QColor>
#include <QMutex>
#include <QThread>
#include "idata.h"
#include "ifilebaseddata.h"
#include "viewutils.h" // for SampleUnit

#include <memory>

#include <gsl/gsl_spline.h>

class WellHead;
class WellBoreGraphicRepFactory;
class WellPick;
class MtLengthUnit;

typedef struct Deviation {
	double tvd;
	double md;
	double x;
	double y;
} Deviation;

typedef struct Deviations {
	std::vector<double> tvds; // tvdss
	std::vector<double> mds;
	std::vector<double> xs;
	std::vector<double> ys;
	double appliedDatum = 0.0; // ! warning tvd already take into account datum, to modify datum you need to update tvd with delta of datum
} Deviations;

typedef struct TFPs {
	bool isTvd;
	std::vector<double> tvds;
	std::vector<double> mds;
	std::vector<double> twts;
} TFPs;

enum WellUnit {
	TVD, TWT, MD, UNDEFINED_UNIT
};

typedef struct Logs {
	WellUnit unit;
	QString sUnit;
	std::vector<double> keys;
	std::vector<double> attributes;
	double nullValue;
	std::vector<std::pair<long, long>> nonNullIntervals;
} Logs;

class FilteringOperator {
public:
	typedef struct DefinitionSet {
		std::vector<double> arrayY;
		double firstX;
		double stepX;
	} DefinitionSet;

	FilteringOperator(const std::vector<DefinitionSet>& intervals, double freq);
	~FilteringOperator();

	double getFilteredY(double x, bool* ok) const;
	bool isDefined(double x) const;
private:
	std::vector<std::pair<double, double>> m_limits;

	gsl_interp_accel* m_acc = nullptr;
	gsl_spline* m_spline = nullptr;
};

enum class ReflectivityError {
	NoError,
	NoTfp,
	TfpNotReversible,
	AttributeLogNotValid,
	VelocityLogNotValid,
	NoLogIntervalIntersection,
	OnlyInvalidIntervals,
	FailToWriteLog
};

/**
 * Specs :
 * Datum are not implemented
 * MD is 0 at well head
 * TVD <-> depth for depth type seismic
 * TWT <-> time for time type seismic
 * only expected units are TVD, TWT, MD
 */
class WellBore : public IData, public IFileBasedData {
	Q_OBJECT
public:
	WellBore(WorkingSetManager* manager, QString descFile, QString deviationPath,
		std::vector<QString> tfpPaths, std::vector<QString> tfpNames, std::vector<QString> logPaths,
		std::vector<QString> logNames, WellHead* wellHead, QObject* parent = 0);
	~WellBore();

	//IData
	virtual IGraphicRepFactory* graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override { return m_name; }

	WellHead* wellHead() const { return m_wellHead; }
	const Deviations& deviations() const;

	const std::vector<QString>& logsNames() const;
	const std::vector<QString>& logsFiles() const;
	const std::vector<QString>& tfpsNames() const;// MZR 05082021
	const std::vector<QString>& tfpsPaths() const;

	static std::pair<QString, double> getNameFromDescFile(QString descFile);
	static QString getTfpFileFromDescFile(QString descFile);
	static std::pair<bool, TFPs> getTFPFromFile(QString tfpFile);
	static QString getKindFromLogFile(QString logFile);


	void GetInfosDescFile(QString descFile);

	QString getInfosFromDescFile(QString descFile);

	void computeMinMax();
	void cacheLogs();
	bool selectTFP(std::size_t index);
	bool selectLog(std::size_t index);
	long currentLogIndex() const;
	const Logs& currentLog() const;
	bool isLogDefined() const;
	void deactivateFiltering();
	void activateFiltering(double freq); // high cut bandpass filter, only valid if a log is set

	//static WellBore* getWellBoreFromDesc(QString descFile, QString deviationPath, QString tfpPath, WellHead* wellHead, WorkingSetManager* manager, QObject* parent=0);
	void addPick(WellPick* pick);
	void removePick(WellPick* pick);
	QList<WellPick*> picks();

	// MZR 04082021
	void SetTfpsPath(const std::vector<QString>& tfpPaths);
	void SetTfpName(const std::vector<QString>& tfpNames);
	void SetlogPath(const std::vector<QString>& logPaths);
	void SetlogName(const std::vector<QString>& logNames);


	// Care about infinite loops in below functions :
	// md is the reference for access to tvd, x, and y (deviation)
	// twt usage should rely on TFPs.isTvd to choose the right unit
	// log usage should rely on Logs.unit to choose the correct transform
	double getTvdFromMd(double mdVal, bool* ok);
	double getXFromMd(double mdVal, bool* ok);
	double getYFromMd(double mdVal, bool* ok);
	double getMdFromTvd(double tvdVal, bool* ok);
	double getXFromTvd(double tvdVal, bool* ok);
	double getYFromTvd(double tvdVal, bool* ok);
	double getTwtFromMd(double mdVal, bool* ok); // md -> tvd -> twt or md -> twt depend of TFP if defined
	double getTwtFromTvd(double tvdVal, bool* ok); //depend of TFP if defined correctly
	double getTvdFromTwt(double twtVal, bool* ok); //depend of TFP if defined correctly
	double getMdFromTwt(double twtVal, bool* ok);  // twt -> tvd -> md or twt -> md   depend of TFP if defined and deviation
	double getXFromTwt(double twt, bool* ok); //depend of TFP if defined correctly
	double getYFromTwt(double twt, bool* ok); //depend of TFP if defined correctly
	double getLogFromMd(double mdVal, bool* ok);
	double getLogFromTvd(double tvdVal, bool* ok);
	double getLogFromTwt(double twtVal, bool* ok);

	// generic functions, to make it easier than above functions
	double getMdFromWellUnit(double idxVal, WellUnit wellUnit, bool* ok);
	double getXFromWellUnit(double idxVal, WellUnit wellUnit, bool* ok);
	double getYFromWellUnit(double idxVal, WellUnit wellUnit, bool* ok);
	double getDepthFromWellUnit(double idxVal, WellUnit wellUnit, SampleUnit depthUnit, bool* ok); // return twt for TIME and tvd for DEPTH
	double getLogFromWellUnit(double idxVal, WellUnit wellUnit, bool* ok);

	// reversed generic function
	double getWellUnitFromTwt(double twtVal, WellUnit wellUnit, bool* ok);

	QString getTfpName() const;
	QString getTfpFilePath() const;
	bool isTfpDefined() const;
	bool isWellCompatibleForTime(bool verbose = false);

	QColor logColor() const;
	void setLogColor(QColor color);
	void deleteRep(); // MZR 19082021

	static double qstringToDouble(const QString& str, bool* ok);
	static QString doubleToQString(const double& val, bool useExponential = false);

	QString getDirName() const;
	QString getDescPath() const;
	QString getLogFileName(int logIndex) const;

	QString getStatus() {
		return m_stat;
	}
	QString getElev() {
		return m_elev;
	}
	QString getDatum() {
		return m_datum;
	}
	// this function could be done by the View3D instead of WellBore data
	QString getConvertedDatum(const MtLengthUnit* newDepthLengthUnit);

	QString getUwi() {
		return m_uwi;
	}

	QString getDomain() {
		return m_domain;
	}
	QString getVelocity() {
		return m_velocity;
	}
	QString getIhs() {
		return m_ihs;
	}

	QColor colorWell()
	{
		return m_wellColor;
	}

	double mini() { return m_mini; }
	double maxi() { return m_maxi; }


	void setWellColor(QColor c)
	{
		m_wellColor = c;
		emit wellColorChanged(c);
	}

	//return vector(X, twt/ depth, Y)
	QVector3D getDirectionFromMd(double value, SampleUnit unit, bool* ok);

	std::vector<QString> extractLogsKinds() const;

	// can be made more generic to support an interval of unknown type, here we only tackle the case of a twt interval
	// instead of creating another one it is better to improve this function
	// this function work well if the delta needed is very small (close to double precision) else it will be extremely slow
	// this can be improved by using dichotomy
	std::vector<std::pair<double, double>> adjustBoundsForTwt(const std::vector<std::pair<double, double>>& intervals,
		WellUnit convertedUnit);

	static bool isLogKeyIncreasing(const Logs& log);
	std::vector<std::pair<double, double>> getTwtNonNullInterval(const Logs& log); // use the current tfp
	static std::vector<std::pair<double, double>> intervalsIntersection(const std::vector<std::pair<double, double>>& intervalA,
		const std::vector<std::pair<double, double>>& intervalB);

	static gsl_spline* getGslObjectsFromLog(const Logs& log); // caller take ownership of the output
	ReflectivityError computeReflectivity(const QString& rhobPath, const QString& velocityPath, double pasech, double freq,
		bool useRicker, const QString& reflectivityName, const QString& reflectivityKind, const QString& reflectivityPath);

	static bool writeLog(const QString& reflectivityName, const QString& reflectivityKind, const QString& reflectivityPath,
		const Logs& log);

	static void computeNonNullInterval(Logs& log);

	static void getAffineFromList(const double* in, const double* out, long n, double& a, double& b);

signals:
	void pickAdded(WellPick* pick);
	void pickRemoved(WellPick* pick);
	void logColorChanged(QColor color);
	void logChanged();
	void boreUpdated();
	void deletedMenu(); // MZR 19082021

	void wellColorChanged(QColor);

private:
	std::pair<bool, Deviations> getDeviationsFromFile(QString deviationFile);
	std::pair<bool, Logs> getLogsFromFile(QString logFile);

	QString m_name;
	QUuid m_uuid;
	QString m_descFile;

	QString m_stat;
	QString m_elev;
	QString m_datum;
	QString m_uwi;
	QString m_domain;
	QString m_velocity;
	QString m_ihs;

	WellHead* m_wellHead;

	Deviations m_deviations;
	TFPs m_currentTfps;
	long m_currentTfpIndex = -1;
	std::vector<QString> m_tfpsFiles;
	std::vector<QString> m_tfpsNames;

	Logs m_currentLogs;
	QMap<QString, std::pair<bool, Logs>> cache_logs;
	long m_currentLogIndex = -1;
	std::vector<QString> m_logsFiles;
	std::vector<QString> m_logsNames;
	bool m_useFiltering = false;
	double m_highcutFrequency = 100.0;

	gsl_interp_accel* m_acc_deviation = nullptr;
	gsl_spline* m_deviation_tvd_spline_steffen = nullptr; // md to tvd
	gsl_spline* m_deviation_x_spline_steffen = nullptr; // md to x
	gsl_spline* m_deviation_y_spline_steffen = nullptr; // md to y

	gsl_interp_accel* m_acc_deviation_tvd = nullptr;
	gsl_spline* m_deviation_tvd2md_spline_steffen = nullptr; // tvd to md, only valid if m_deviationTvd2MdActive == true
	double m_mdFromDeviationBoundMin, m_mdFromDeviationBoundMax;
	bool m_deviationSplineActive;
	bool m_deviationAffineActive;
	bool m_deviationTvd2MdActive = false;
	double m_tvdFromDeviationBoundMin, m_tvdFromDeviationBoundMax; // only valid if m_deviationTvd2MdActive == true
	// affine function : out = in * a + b; for tvd, x, y
	double m_deviation_x_a, m_deviation_x_b, m_deviation_y_a, m_deviation_y_b, m_deviation_tvd_a, m_deviation_tvd_b;
	double m_deviation_tvd2md_a, m_deviation_tvd2md_b; // only valid if m_deviationTvd2MdActive == true

	gsl_spline* m_tfp_spline_steffen = nullptr; // tfpindex to tfp
	gsl_interp_accel* m_acc_tfp = nullptr;
	gsl_spline* m_tfp_index_spline_steffen = nullptr; // tfp to tfpindex
	gsl_interp_accel* m_acc_tfp_index = nullptr;

	double m_fromTfpBoundMin, m_fromTfpBoundMax, m_twtFromTfpBoundMin, m_twtFromTfpBoundMax;
	bool m_tfpTwt2IndexActive = false;
	double m_tfp_twt_a, m_tfp_twt_b, m_tfp_twt_index_a, m_tfp_twt_index_b;
	bool m_tfpSplineActive;
	bool m_tfpAffineActive;

	gsl_interp_accel* m_acc_log = nullptr;
	gsl_spline* m_log_val_spline_steffen = nullptr;
	double m_log_a = 0, m_log_b = 0;
	bool m_logSplineActive = false;
	bool m_logAffineActive = false;

	//std::vector<std::pair<double, double>> m_fromLogBounds; // if needed to check md values

	std::unique_ptr<FilteringOperator> m_logFilter;

	WellBoreGraphicRepFactory* m_repFactory;

	QList<WellPick*> m_picks;
	QColor m_logColor;
	QColor m_wellColor = Qt::yellow;
	bool m_ProcessDelete = false;

	double m_mini = 0.0;
	double m_maxi = 0.0;
};
class WellBoreWorker : public QObject {
	Q_OBJECT
public:
	WellBoreWorker(QObject* parent = nullptr);
	~WellBoreWorker();
public slots:
	void process(WellBore* wellbore);
signals:
	void finished();
	void error(QString err);
private:
};
#endif
