#ifndef SRC_BNNI_MODELS_NETWORKPARAMETERSMODEL_H
#define SRC_BNNI_MODELS_NETWORKPARAMETERSMODEL_H

#include "structures.h"

#include <QObject>

class DenseParametersModel;
class TreeParametersModel;

class NetworkParametersModel : public QObject {
	Q_OBJECT
public:
	NetworkParametersModel(QObject* parent=nullptr);
	~NetworkParametersModel();

	DenseParametersModel* denseModel();
	TreeParametersModel* treeModel();

	NeuralNetwork getNetwork() const;
	bool loadNetwork(QString txt);
	void setNetwork(NeuralNetwork val);

	// getter/load to set
	virtual bool validateArguments();
	virtual QStringList warningForArguments();

	double getLearningRate() const;
	SeismicPreprocessing getSeismicPreprocessing() const;
	int getHatParameter() const;
	unsigned int getEpochSaveStep() const;
	unsigned int getNGpus() const;
	QString getSavePrefix() const;
	WellPostprocessing getWellPostprocessing() const;
	float getWellFilterFrequency() const;
	bool hasReferenceCheckpoint() const;
	QString getReferenceCheckpoint() const;
	bool setReferenceCheckpoint(const QString& reference);

	unsigned int getNEstimator() const;
	unsigned int getMaxDepth() const;
	double getSubSample() const;
	double getColSampleByTree() const;

	QVector<unsigned int> getHiddenLayers() const;
	void setHiddenLayers(const QVector<unsigned int>& val);
	double getMomentum() const;
	unsigned int getNumEpochs() const;
	Optimizer getOptimizer() const;
	float getDropout() const;
	bool getBatchNorm() const;
	Activation getLayerActivation() const;
	unsigned int getBatchSize() const;
	PrecisionType getPrecision() const;
	bool getUseBias() const;

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

	bool loadNEstimator(QString txt);
	bool loadMaxDepth(QString txt);
	bool loadSubSample(QString txt);
	bool loadColSampleByTree(QString txt);

	bool loadHiddenLayers(QStringList txt);
	bool loadMomentum(QString txt);
	bool loadNumEpochs(QString txt);
	bool loadOptimizer(QString txt);
	bool loadDropout(QString txt);
	bool loadBatchNorm(QString txt);
	bool loadLayerActivation(QString txt);
	bool loadBatchSize(QString txt);
	bool loadPrecision(QString txt);
	bool loadUseBias(QString txt);

signals:
	void networkChanged(NeuralNetwork val);

private:
	DenseParametersModel* m_denseModel;
	TreeParametersModel* m_treeModel;

	NeuralNetwork m_network = NeuralNetwork::Dense;
};

#endif // SRC_BNNI_MODELS_NETWORKPARAMETERSMODEL_H
