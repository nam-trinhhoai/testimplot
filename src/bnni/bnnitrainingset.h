#ifndef SRC_BNNI_BNNITRAININGSET_H_
#define SRC_BNNI_BNNITRAININGSET_H_

#include <QObject>
#include <QString>

#include <utility>
#include <vector>
#include <map>

#include "viewutils.h"

class BnniTrainingSet : public QObject {
	Q_OBJECT
public:
	enum WellFilter {
		WellName, WellKind
	};

	typedef struct BnniWellHeader {
		WellFilter filterType = WellFilter::WellKind;
		QString filterStr = "";
		float min = 0;
		float max = 1;
		float weight = 1.0;
	} BnniWellHeader;

	typedef struct WellParameter {
		QString deviationFile = "";
		QString boreDescFile = "";
		QString headDescFile = "";
		QString wellName = "";
		QString tfpFile = "";
		QString tfpName = "";
		int cacheHeadIdx = -1;
		int cacheBoreIdx = -1;
		std::map<long, std::pair<QString, QString>> logsPathAndName; // path, nv name
	} WellParameter;

	typedef struct SeismicParameter {
		QString name = "";
		QString path = "";
		float min = 0;
		float max = 1;
		float weight = 1.0;
		SampleUnit unit = SampleUnit::NONE;
	} SeismicParameter;

	typedef struct HorizonParameter {
		QString name = "";
		QString path = "";
		float delta = 0.0f;
	} HorizonParameter;

	typedef struct HorizonIntervalParameter {
		HorizonParameter topHorizon;
		HorizonParameter bottomHorizon;
	} HorizonIntervalParameter;

	BnniTrainingSet(QObject* parent=nullptr);
	~BnniTrainingSet();

	// seismic params
	const std::map<long , SeismicParameter>& seismics() const;
	long createNewSeismic(); // return index
	void deleteSeismic(long id);
	bool changeSeismic(long id, SeismicParameter param);
	float targetSampleRate() const;
	void setTargetSampleRate(float val);
	int halfWindow() const;
	void setHalfWindow(int val);

	// well params
	const std::map<long, WellParameter>& wellBores() const;
	long createNewWell(); // return index
	void deleteWell(long wellId);
	bool changeWellBore(long id, WellParameter wellParam);
	const std::map<long, BnniWellHeader>& wellHeaders() const;
	long createNewKind(); // return index
	void deleteKind(long kindId);
	bool changeKind(long id, BnniWellHeader header);
	const QString& tfpFilter() const;
	void setTfpFilter(const QString& newTfpFilter);


	bool useBandPassHighFrequency() const;
	float bandPassHighFrequency() const;
	void activateBandPassHighFrequency(float freq);
	void deactivateBandPassHighFrequency();
	float mdSamplingRate() const;
	void setMdSamplingRate(float val);
	bool useAugmentation() const;
	void setUseAugmentation(bool val);
	int augmentationDistance() const;
	void setAugmentationDistance(int dist);
	float gaussianNoiseStd() const;
	void setGaussianNoiseStd(float val);
	bool useCnxAugmentation() const;
	void toggleCnxAugmentation(bool val);

	// horizon params
	const std::map<long, HorizonIntervalParameter>& intervals() const;
	long createNewInterval(); // return index
	void deleteInterval(long id);
	bool changeInterval(long id, HorizonIntervalParameter param);


	// json
	QString projectPath() const;
	void setProjectPath(const QString& projectPath);
	QString trainingSetName() const;
	void setTrainingSetName(const QString& trainingSetName);
	QString outputJsonFile() const;

	static bool isLogEmpty(long logId, std::map<long, std::pair<QString, QString>> logMap);

	SampleUnit seismicUnit() const;
	void updateSeismicUnitFromSeismics();

signals:
	void outputJsonFileChanged(QString jsonPath);
	void seismicUnitChanged(SampleUnit sampleUnit);

private:
	QString computeOutputJsonFile(const QString& projectPath, const QString& trainingSetName);
	void searchNewTrainingSetName();
	void setOutputJsonFile(const QString& jsonPath);

	long nextId() const;

	void setSeismicUnit(SampleUnit sampleUnit);

	float m_pasSampleSurrechantillon = 0.5;

	QString m_projectPath;
	QString m_trainingSetName;
	QString m_outputJsonFile = "";
	const QString DEFAULT_TRAINING_SET_NAME = "trainingset";
	const QString DEFAULT_JSON_NAME = "trainingset.json";
	const QString DEFAULT_RELATIVE_PATH = "DATA/NEURONS/neurons2/LogInversion2Problem3";

	bool m_useBandPassHighFrequency = false;
	float m_bandPassHighFrequency = 100.0;
	float m_mdSamplingRate = 0.001;
	bool m_useAugmentation = false;
	int m_augmentationDistance = 11; // 2*ws+1 with ws being the default window size for genetic algorithm search
	float m_gaussianNoiseStd = 0.001; // use 0 mean
	bool m_useCnxAugmentation = true;

	std::map<long, SeismicParameter> m_seismics;
	std::map<long, WellParameter> m_wellBores;
	std::map<long, BnniWellHeader> m_wellHeaders;
	QString m_tfpFilter = "tfp";
	std::map<long, HorizonIntervalParameter> m_horizonIntervals;
	int m_halfWindow = 20;

	SampleUnit m_seismicUnit = SampleUnit::NONE;

	mutable long m_nextId = 0;
};

#endif
