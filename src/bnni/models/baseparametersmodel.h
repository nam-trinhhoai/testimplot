#ifndef SRC_BNNI_MODELS_BASEPARAMETERSMODEL_H
#define SRC_BNNI_MODELS_BASEPARAMETERSMODEL_H

#include "structures.h"

#include <QObject>

#include <vector>

class BaseParametersModel : public QObject {
	Q_OBJECT
public:
	BaseParametersModel(QObject* parent=nullptr);
	~BaseParametersModel();

	double getLearningRate() const;
	void setLearningRate(double val);
	SeismicPreprocessing getSeismicPreprocessing() const;
	void setSeismicPreprocessing(SeismicPreprocessing val);
	int getHatParameter() const;
	void setHatParameter(int val);
	unsigned int getEpochSaveStep() const;
	void setEpochSaveStep(unsigned int val);
	unsigned int getNGpus() const;
	void setNGpus(unsigned int val);
	QString getSavePrefix() const;
	void setSavePrefix(const QString& val);
	WellPostprocessing getWellPostprocessing() const;
	void setWellPostprocessing(WellPostprocessing val);
	float getWellFilterFrequency() const;
	void setWellFilterFrequency(float val);
	bool hasReferenceCheckpoint() const;
	QString getReferenceCheckpoint() const;
	bool setReferenceCheckpoint(const QString& val);

	QString getCheckpointDir() const;
	void setCheckpointDir(const QString& checkpointDir);

	bool loadLearningRate(QString txt);
	bool loadSeismicPreprocessing(QString txt);
	bool loadHatParameter(QString txt);
	bool loadEpochSaveStep(QString);
	bool loadNGpus(QString txt);
	bool loadSavePrefix(QString txt);
	bool loadReferenceCheckpoint(QString txt);
	bool loadWellPostprocessing(QString txt);
	bool loadWellFilterFrequency(QString txt);

	// this could return a QStringList if needed
	virtual QString getExtension() const = 0;

	virtual bool validateArguments();
	virtual QStringList warningForArguments();

	std::vector<QString> getAvailableCheckpoints() const;

public slots:
	// this function is here to avoid code duplicates
	void editReferenceCheckpoint();

signals:
	void learningRateChanged(double val);
	void epochSaveStepChanged(unsigned int val);
	void nGpusChanged(unsigned int val);
	void savePrefixChanged(QString val);
	void referenceCheckpointChanged(QString val);
	void seismicPreprocessingChanged(int val);
	void hatParameterChanged(int val);
	void wellPostprocessingChanged(int val);
	void wellFilterFrequencyChanged(float val);
	void checkpointDirChanged(QString txt);

private:
	void setReferenceCheckpointPrivate(const QString& val);

	double m_learningRate = 0.1;
	int m_epochSaveStep = 1;
	int m_nGpus = 1;
	QString m_savePrefix = "xgb_ckpt";
	SeismicPreprocessing m_seismicPreprocessing = SeismicPreprocessing::SeismicNone;
	int m_hatParameter = 1;
	WellPostprocessing m_wellPostprocessing = WellPostprocessing::WellNone;
	float m_wellFilterFrequency = 100.0;
	QString m_referenceCheckpoint;
	QString m_checkpointDir;
};

#endif
