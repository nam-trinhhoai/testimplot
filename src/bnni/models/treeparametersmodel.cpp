#include "treeparametersmodel.h"

TreeParametersModel::TreeParametersModel(QObject* parent) : BaseParametersModel(parent) {
	setLearningRate(0.1);
	setEpochSaveStep(1);
	setNGpus(1);
	setSavePrefix("xgb_ckpt");
	setSeismicPreprocessing(SeismicPreprocessing::SeismicNone);
	setHatParameter(1);
	setWellPostprocessing(WellPostprocessing::WellNone);
	setWellFilterFrequency(100.0);
	setReferenceCheckpoint("");
}

TreeParametersModel::~TreeParametersModel() {

}

QString TreeParametersModel::getExtension() const {
	return "ubj";
}

bool TreeParametersModel::validateArguments() {
	bool test = BaseParametersModel::validateArguments() && m_subSample>0 && m_subSample<=1 &&
			m_maxDepth>0 && m_colSampleByTree>0 && m_colSampleByTree<=1 && m_nEstimator>0;

	return test;
}

QStringList TreeParametersModel::warningForArguments() {
	QStringList warnings = BaseParametersModel::warningForArguments();
	if (m_maxDepth<=0) {
		warnings << "Max depth <= 0";
	}
	if (m_subSample<=0 || m_subSample>1) {
		warnings << "Sub sample out of range ]0, 1]";
	}
	if (m_colSampleByTree<=0 || m_colSampleByTree>1) {
		warnings << "Col sample by tree out of range ]0, 1]";
	}
	if (m_nEstimator<=0) {
		warnings << "Number of tress need to be greater than 0";
	}
	return warnings;
}

unsigned int TreeParametersModel::getNEstimator() const {
	return m_nEstimator;
}

bool TreeParametersModel::loadNEstimator(QString txt) {
	bool ok;
	int val = txt.toUInt(&ok);
	ok = ok && val > 0;
	if (ok) {
		setNEstimator(val);
	}
	return ok;
}

void TreeParametersModel::setNEstimator(unsigned int val) {
	if (m_nEstimator!=val) {
		m_nEstimator = val;
		emit nEstimatorChanged(m_nEstimator);
	}
}

unsigned int TreeParametersModel::getMaxDepth() const {
	return m_maxDepth;
}

bool TreeParametersModel::loadMaxDepth(QString txt) {
	bool ok = true;
	int val = txt.toUInt(&ok);
	ok = ok && val > 0;
	if (ok) {
		setMaxDepth(val);
	}
	return ok;
}

void TreeParametersModel::setMaxDepth(unsigned int val) {
	if (m_maxDepth!=val) {
		m_maxDepth = val;
		emit maxDepthChanged(m_maxDepth);
	}
}

double TreeParametersModel::getSubSample() const {
	return m_subSample;
}

bool TreeParametersModel::loadSubSample(QString txt) {
	bool ok = true;
	float val = txt.toFloat(&ok);
	ok = ok && val > 0;
	if (ok) {
		setSubSample(val);
	}
	return ok;
}

void TreeParametersModel::setSubSample(double val) {
	if (m_subSample!=val) {
		m_subSample = val;
		emit subSampleChanged(m_subSample);
	}
}

double TreeParametersModel::getColSampleByTree() const {
	return m_colSampleByTree;
}

bool TreeParametersModel::loadColSampleByTree(QString txt) {
    bool ok = true;
	float val = txt.toFloat(&ok);
    ok = ok && val > 0;
    if (ok) {
    	setColSampleByTree(val);
    }
	return ok;
}

void TreeParametersModel::setColSampleByTree(double val) {
	if (m_colSampleByTree!=val) {
		m_colSampleByTree = val;
		emit colSampleByTreeChanged(m_colSampleByTree);
	}
}
