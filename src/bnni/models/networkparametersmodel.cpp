#include "networkparametersmodel.h"

#include "denseparametersmodel.h"
#include "treeparametersmodel.h"

NetworkParametersModel::NetworkParametersModel(QObject* parent) : QObject(parent) {
	m_denseModel = new DenseParametersModel(this);
	m_treeModel = new TreeParametersModel(this);
}

NetworkParametersModel::~NetworkParametersModel() {

}

DenseParametersModel* NetworkParametersModel::denseModel() {
	return m_denseModel;
}

TreeParametersModel* NetworkParametersModel::treeModel() {
	return m_treeModel;
}

NeuralNetwork NetworkParametersModel::getNetwork() const {
	return m_network;
}

bool NetworkParametersModel::loadNetwork(QString txt) {
	// no other choice for now
	if (txt.toLower().compare("dnn")==0) {
		setNetwork(NeuralNetwork::Dnn);
	} else if (txt.toLower().compare("xgboost")==0) {
		setNetwork(NeuralNetwork::Xgboost);
	} else {
		setNetwork(NeuralNetwork::Dense);
	}
	return true;
}

void NetworkParametersModel::setNetwork(NeuralNetwork val) {
	if (m_network!=val) {
		m_network = val;
		emit networkChanged(m_network);
	}
}

bool NetworkParametersModel::validateArguments() {
	bool test;
	if (m_network==NeuralNetwork::Xgboost) {
		test = m_treeModel->validateArguments();
	} else {
		test = m_denseModel->validateArguments();
	}

	return test;
}

QStringList NetworkParametersModel::warningForArguments() {
	QStringList warnings;
	if (m_network==NeuralNetwork::Xgboost) {
		warnings = m_treeModel->warningForArguments();
	} else {
		warnings = m_denseModel->warningForArguments();
	}
	return warnings;
}

// general params
double NetworkParametersModel::getLearningRate() const{
	double learningRate;
	if (m_network==NeuralNetwork::Xgboost) {
		learningRate = m_treeModel->getLearningRate();
	} else {
		learningRate = m_denseModel->getLearningRate();
	}
	return learningRate;
}

bool NetworkParametersModel::loadLearningRate(QString txt) {
	bool output;
	if (m_network==NeuralNetwork::Xgboost) {
		output = m_treeModel->loadLearningRate(txt);
	} else {
		output= m_denseModel->loadLearningRate(txt);
	}
	return output;
}

unsigned int NetworkParametersModel::getEpochSaveStep() const {
	int epochSavingStep;
	if (m_network==NeuralNetwork::Xgboost) {
		epochSavingStep = m_treeModel->getEpochSaveStep();
	} else {
		epochSavingStep = m_denseModel->getEpochSaveStep();
	}
	return epochSavingStep;
}

bool NetworkParametersModel::loadEpochSaveStep(QString txt) {
	bool output;
	if (m_network==NeuralNetwork::Xgboost) {
		output = m_treeModel->loadEpochSaveStep(txt);
	} else {
		output = m_denseModel->loadEpochSaveStep(txt);
	}
	return output;
}

unsigned int NetworkParametersModel::getNGpus() const {
	unsigned int nGpus;
	if (m_network==NeuralNetwork::Xgboost) {
		nGpus = m_treeModel->getNGpus();
	} else {
		nGpus = m_denseModel->getNGpus();
	}
	return nGpus;
}

bool NetworkParametersModel::loadNGpus(QString txt) {
	bool output;
	if (m_network==NeuralNetwork::Xgboost) {
		output = m_treeModel->loadNGpus(txt);
	} else {
		output = m_denseModel->loadNGpus(txt);
	}
	return output;
}

QString NetworkParametersModel::getSavePrefix() const {
	QString savePrefix;
	if (m_network==NeuralNetwork::Xgboost) {
		savePrefix = m_treeModel->getSavePrefix();
	} else {
		savePrefix = m_denseModel->getSavePrefix();
	}
	return savePrefix;
}

bool NetworkParametersModel::loadSavePrefix(QString txt) {
	bool output;
	if (m_network==NeuralNetwork::Xgboost) {
		output = m_treeModel->loadSavePrefix(txt);
	} else {
		output = m_denseModel->loadSavePrefix(txt);
	}
	return output;
}

SeismicPreprocessing NetworkParametersModel::getSeismicPreprocessing() const {
	SeismicPreprocessing preprocessing;
	if (m_network==NeuralNetwork::Xgboost) {
		preprocessing = m_treeModel->getSeismicPreprocessing();
	} else {
		preprocessing = m_denseModel->getSeismicPreprocessing();
	}
	return preprocessing;
}

bool NetworkParametersModel::loadSeismicPreprocessing(QString txt) {
	bool output;
	if (m_network==NeuralNetwork::Xgboost) {
		output = m_treeModel->loadSeismicPreprocessing(txt);
	} else {
		output = m_denseModel->loadSeismicPreprocessing(txt);
	}
	return output;
}

int NetworkParametersModel::getHatParameter() const {
	int hat;
	if (m_network==NeuralNetwork::Xgboost) {
		hat = m_treeModel->getHatParameter();
	} else {
		hat = m_denseModel->getHatParameter();
	}
	return hat;
}

bool NetworkParametersModel::loadHatParameter(QString txt) {
	bool output;
	if (m_network==NeuralNetwork::Xgboost) {
		output = m_treeModel->loadHatParameter(txt);
	} else {
		output = m_denseModel->loadHatParameter(txt);
	}
	return output;
}

WellPostprocessing NetworkParametersModel::getWellPostprocessing() const {
	WellPostprocessing postprocessing;
	if (m_network==NeuralNetwork::Xgboost) {
		postprocessing = m_treeModel->getWellPostprocessing();
	} else {
		postprocessing = m_denseModel->getWellPostprocessing();
	}
	return postprocessing;
}

bool NetworkParametersModel::loadWellPostprocessing(QString txt) {
	bool output;
	if (m_network==NeuralNetwork::Xgboost) {
		output = m_treeModel->loadWellPostprocessing(txt);
	} else {
		output = m_denseModel->loadWellPostprocessing(txt);
	}
	return output;
}

float NetworkParametersModel::getWellFilterFrequency() const {
	int freq;
	if (m_network==NeuralNetwork::Xgboost) {
		freq = m_treeModel->getWellFilterFrequency();
	} else {
		freq = m_denseModel->getWellFilterFrequency();
	}
	return freq;
}

bool NetworkParametersModel::loadWellFilterFrequency(QString txt) {
	bool output;
	if (m_network==NeuralNetwork::Xgboost) {
		output = m_treeModel->loadWellFilterFrequency(txt);
	} else {
		output = m_denseModel->loadWellFilterFrequency(txt);
	}
	return output;
}

bool NetworkParametersModel::hasReferenceCheckpoint() const {
	bool output;
	if (m_network==NeuralNetwork::Xgboost) {
		output = m_treeModel->hasReferenceCheckpoint();
	} else {
		output = m_denseModel->hasReferenceCheckpoint();
	}
	return output;
}

QString NetworkParametersModel::getReferenceCheckpoint() const {
	QString output;
	if (m_network==NeuralNetwork::Xgboost) {
		output = m_treeModel->getReferenceCheckpoint();
	} else {
		output = m_denseModel->getReferenceCheckpoint();
	}
	return output;
}

bool NetworkParametersModel::setReferenceCheckpoint(const QString& reference) {
	bool output;
	if (m_network==NeuralNetwork::Xgboost) {
		output = m_treeModel->setReferenceCheckpoint(reference);
	} else {
		output = m_denseModel->setReferenceCheckpoint(reference);
	}
	return output;
}

bool NetworkParametersModel::loadReferenceCheckpoint(QString txt) {
	bool output;
	if (m_network==NeuralNetwork::Xgboost) {
		output = m_treeModel->loadReferenceCheckpoint(txt);
	} else {
		output = m_denseModel->loadReferenceCheckpoint(txt);
	}
	return output;
}

QString NetworkParametersModel::getCheckpointDir() const {
	QString path;
	if (m_network==NeuralNetwork::Xgboost) {
		path = m_treeModel->getCheckpointDir();
	} else {
		path = m_denseModel->getCheckpointDir();
	}
	return path;
}

void NetworkParametersModel::setCheckpointDir(const QString& checkpointDir) {
	m_treeModel->setCheckpointDir(checkpointDir);
	m_denseModel->setCheckpointDir(checkpointDir);
}

// only for dense params
QVector<unsigned int> NetworkParametersModel::getHiddenLayers() const {
	return m_denseModel->getHiddenLayers();
}

void NetworkParametersModel::setHiddenLayers(const QVector<unsigned int>& val) {
	return m_denseModel->setHiddenLayers(val);
}

bool NetworkParametersModel::loadHiddenLayers(QStringList txt) {
	return m_denseModel->loadHiddenLayers(txt);
}

double NetworkParametersModel::getMomentum() const {
	return m_denseModel->getMomentum();
}

bool NetworkParametersModel::loadMomentum(QString txt) {
	return m_denseModel->loadMomentum(txt);
}

Optimizer NetworkParametersModel::getOptimizer() const {
	return m_denseModel->getOptimizer();
}

bool NetworkParametersModel::loadOptimizer(QString txt) {
	return m_denseModel->loadOptimizer(txt);
}

float NetworkParametersModel::getDropout() const {
	return m_denseModel->getDropout();
}

bool NetworkParametersModel::loadDropout(QString txt) {
	return m_denseModel->loadDropout(txt);
}

bool NetworkParametersModel::getBatchNorm() const {
	return m_denseModel->getBatchNorm();
}

bool NetworkParametersModel::loadBatchNorm(QString txt) {
	return m_denseModel->loadBatchNorm(txt);
}

Activation NetworkParametersModel::getLayerActivation() const {
	return m_denseModel->getLayerActivation();
}

bool NetworkParametersModel::loadLayerActivation(QString txt) {
	return m_denseModel->loadLayerActivation(txt);
}

unsigned int NetworkParametersModel::getBatchSize() const {
	return m_denseModel->getBatchSize();
}

bool NetworkParametersModel::loadBatchSize(QString txt) {
	return m_denseModel->loadBatchSize(txt);
}

PrecisionType NetworkParametersModel::getPrecision() const {
	return m_denseModel->getPrecision();
}

bool NetworkParametersModel::loadPrecision(QString txt) {
	return m_denseModel->loadPrecision(txt);
}

unsigned int NetworkParametersModel::getNumEpochs() const {
	unsigned int numEpochs;
	numEpochs = m_denseModel->getNumEpochs();
	return numEpochs;
}

bool NetworkParametersModel::loadNumEpochs(QString txt) {
	bool output = m_denseModel->loadNumEpochs(txt);
	return output;
}

bool NetworkParametersModel::getUseBias() const {
	return m_denseModel->getUseBias();
}

bool NetworkParametersModel::loadUseBias(QString txt) {
	return m_denseModel->loadUseBias(txt);
}

// tree params
unsigned int NetworkParametersModel::getMaxDepth() const {
	return m_treeModel->getMaxDepth();
}

bool NetworkParametersModel::loadMaxDepth(QString txt) {
	return m_treeModel->loadMaxDepth(txt);
}

double NetworkParametersModel::getSubSample() const {
	return m_treeModel->getSubSample();
}

bool NetworkParametersModel::loadSubSample(QString txt) {
	return m_treeModel->loadSubSample(txt);
}

double NetworkParametersModel::getColSampleByTree() const {
	return m_treeModel->getColSampleByTree();
}

bool NetworkParametersModel::loadColSampleByTree(QString txt) {
	return m_treeModel->loadColSampleByTree(txt);
}

unsigned int NetworkParametersModel::getNEstimator() const {
	return m_treeModel->getNEstimator();
}

bool NetworkParametersModel::loadNEstimator(QString txt) {
	return m_treeModel->loadNEstimator(txt);
}
