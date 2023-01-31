#include "denseparametersmodel.h"

DenseParametersModel::DenseParametersModel(QObject* parent) : BaseParametersModel(parent) {
	setLearningRate(0.001);
	setEpochSaveStep(50);
	setNGpus(1);
	setSavePrefix("ckpt");
	setReferenceCheckpoint("");

	setSeismicPreprocessing(SeismicPreprocessing::SeismicNone);
	setHatParameter(1);
	setWellPostprocessing(WellPostprocessing::WellNone);
	setWellFilterFrequency(100.0);
}

DenseParametersModel::~DenseParametersModel() {

}

QString DenseParametersModel::getExtension() const {
	return "index";
}

bool DenseParametersModel::validateArguments() {
	bool test = BaseParametersModel::validateArguments() && m_dropout>=0 && m_dropout < 1 && m_batchSize > 0;

	return test;
}

QStringList DenseParametersModel::warningForArguments() {
	QStringList warnings = BaseParametersModel::warningForArguments();
	if (m_dropout<0|| m_dropout >= 1) {
		warnings << "Dropout out of range [0, 1[";
	}
	if (m_batchSize<=0) {
		warnings << "Batch size need to be greater than 0";
	}
	return warnings;
}

QVector<unsigned int> DenseParametersModel::getHiddenLayers() const {
	return m_hiddenLayers;
}

bool DenseParametersModel::loadHiddenLayers(QStringList txt) {
	QVector<unsigned int> layers;
	bool ok = true;
	int index = 0;
	while (ok && index < txt.size()) {
		layers.append(txt[index].toUInt(&ok));
		index++;
	}
	if (ok) {
		setHiddenLayers(layers);
	}
	return ok;
}

void DenseParametersModel::setHiddenLayers(const QVector<unsigned int>& val) {
	if (m_hiddenLayers!=val) {
		m_hiddenLayers = val;
		emit hiddenLayersChanged(m_hiddenLayers);
	}
}

double DenseParametersModel::getMomentum() const {
	return m_momentum;
}

bool DenseParametersModel::loadMomentum(QString txt) {
	bool ok;
	float val = txt.toFloat(&ok);
	ok = ok && val > 0;
	if (ok) {
		setMomentum(val);
	}
	return ok;
}

void DenseParametersModel::setMomentum(double val) {
	if (m_momentum!=val) {
		m_momentum= val;
		emit momentumChanged(m_momentum);
	}
}

unsigned int DenseParametersModel::getNumEpochs() const {
	return m_numEpochs;
}

bool DenseParametersModel::loadNumEpochs(QString txt) {
	bool ok;
	unsigned int val = txt.toUInt(&ok);
	if (ok) {
		setNumEpochs(val);
	}
	return ok;
}

void DenseParametersModel::setNumEpochs(unsigned int val) {
	if (m_numEpochs!=val) {
		m_numEpochs = val;
		emit numEpochsChanged(m_numEpochs);
	}
}

Optimizer DenseParametersModel::getOptimizer() const {
	return m_optimizer;
}

bool DenseParametersModel::loadOptimizer(QString txt) {
	if (txt.compare("adam")==0) {
		setOptimizer(Optimizer::adam);
	} else if (txt.compare("momentum")==0) {
		setOptimizer(Optimizer::momentum);
	} else {
		setOptimizer(Optimizer::gradientDescent);
	}
	return true;
}

void DenseParametersModel::setOptimizer(Optimizer val) {
	if (m_optimizer!=val) {
		m_optimizer = val;
		emit optimizerChanged(m_optimizer);
	}
}

Activation DenseParametersModel::getLayerActivation() const {
	return m_layerActivation;
}

bool DenseParametersModel::loadLayerActivation(QString txt) {
	if (txt.compare("linear")==0) {
		setLayerActivation(Activation::linear);
	} else if (txt.compare("relu")==0) {
		setLayerActivation(Activation::relu);
	} else if (txt.compare("selu")==0) {
		setLayerActivation(Activation::selu);
	} else if (txt.compare("leaky_relu")==0) {
		setLayerActivation(Activation::leaky_relu);
	} else {
		setLayerActivation(Activation::sigmoid);
	}
	return true;
}

void DenseParametersModel::setLayerActivation(Activation val) {
	if (m_layerActivation!=val) {
		m_layerActivation = val;
		emit layerActivationChanged(m_layerActivation);
	}
}

float DenseParametersModel::getDropout() const {
	return m_dropout;
}

bool DenseParametersModel::loadDropout(QString txt) {
	bool ok;

	float val = txt.toFloat(&ok);
	ok = ok && val >= 0 && val < 1;
	if (ok) {
		setDropout(val);
	}
	return ok;
}

void DenseParametersModel::setDropout(float val) {
	if (m_dropout!=val) {
		m_dropout = val;
		emit dropoutChanged(m_dropout);
	}
}

bool DenseParametersModel::getBatchNorm() const {
	return m_batchNorm;
}

bool DenseParametersModel::loadBatchNorm(QString txt) {
	bool ok = txt.compare("false")!=0;
	setBatchNorm(ok);
	return true;
}

void DenseParametersModel::setBatchNorm(bool val) {
	if (m_batchNorm!=val) {
		m_batchNorm = val;
		emit batchNormChanged(m_batchNorm);
	}
}

unsigned int DenseParametersModel::getBatchSize() const {
    return m_batchSize;
}

bool DenseParametersModel::loadBatchSize(QString txt) {
	bool ok;
	unsigned int val = txt.toUInt(&ok);
	ok = ok && val > 0;
	if (ok) {
		setBatchSize(val);
	}
	return ok;
}

void DenseParametersModel::setBatchSize(unsigned int val) {
	if (m_batchSize!=val) {
		m_batchSize = val;
		emit batchSizeChanged(m_batchSize);
	}
}

PrecisionType DenseParametersModel::getPrecision() const {
	return m_precision;
}

bool DenseParametersModel::loadPrecision(QString txt) {
	if (txt.compare("fp16")==0) {
		setPrecision(PrecisionType::float16);
	} else {
		setPrecision(PrecisionType::float32);
	}
	return true;
}

void DenseParametersModel::setPrecision(PrecisionType val) {
	if (m_precision!=val) {
		m_precision = val;
		emit precisionChanged(m_precision);
	}
}

bool DenseParametersModel::getUseBias() const {
	return m_useBias;
}

bool DenseParametersModel::loadUseBias(QString txt) {
	bool ok = txt.compare("false")!=0;
	setUseBias(ok);
	return true;
}

void DenseParametersModel::setUseBias(bool val) {
	if (m_useBias!=val) {
		m_useBias = val;
		emit useBiasChanged(m_useBias);
	}
}
