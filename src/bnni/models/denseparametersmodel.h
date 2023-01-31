#ifndef SRC_BNNI_MODELS_DESNSEPARAMETERSMODEL_H
#define SRC_BNNI_MODELS_DESNSEPARAMETERSMODEL_H

#include "baseparametersmodel.h"

#include <QString>
#include <QVector>

class DenseParametersModel : public BaseParametersModel {
	Q_OBJECT
public:
	DenseParametersModel(QObject* parent=nullptr);
	~DenseParametersModel();

	QVector<unsigned int> getHiddenLayers() const;
	void setHiddenLayers(const QVector<unsigned int>& val);
	double getMomentum() const;
	void setMomentum(double val);
	unsigned int getNumEpochs() const;
	void setNumEpochs(unsigned int val);
	Optimizer getOptimizer() const;
	void setOptimizer(Optimizer val);
	float getDropout() const;
	void setDropout(float val);

	bool getBatchNorm() const;
	void setBatchNorm(bool val) ;
	Activation getLayerActivation() const;
	void setLayerActivation(Activation val);
	unsigned int getBatchSize() const;
	void setBatchSize(unsigned int val);

	PrecisionType getPrecision() const;
	void setPrecision(PrecisionType val);
	bool getUseBias() const;
	void setUseBias(bool val);

	// update with key
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

	virtual QString getExtension() const override;
    virtual bool validateArguments() override;
    virtual QStringList warningForArguments() override;

signals:
	void hiddenLayersChanged(QVector<unsigned int> val);
	void momentumChanged(double val);
	void numEpochsChanged(unsigned int val);
	void batchSizeChanged(unsigned int val);
	void precisionChanged(int val);
	void useBiasChanged(bool val);

	void optimizerChanged(int val);
	void dropoutChanged(float val);
	void batchNormChanged(bool val);
	void layerActivationChanged(int val);

private:
	QVector<unsigned int> m_hiddenLayers = {50};
	double m_momentum = 0.7;
	unsigned int m_numEpochs = 10000;
	unsigned int m_batchSize = 50;
	PrecisionType m_precision = PrecisionType::float32;
	bool m_useBias = false;

	Optimizer m_optimizer = Optimizer::adam;
	float m_dropout = 0.0;
	bool m_batchNorm = true;
	Activation m_layerActivation = Activation::sigmoid;
};

#endif
