#include "xgboostgeneticparams.h"

XgBoostGeneticParams::XgBoostGeneticParams(QObject* parent) : QObject(parent) {

}

XgBoostGeneticParams::~XgBoostGeneticParams() {

}

int XgBoostGeneticParams::maxDepth() const {
	return m_maxDepth;
}

int XgBoostGeneticParams::nEstimators() const {
	return m_nEstimators;
}

double XgBoostGeneticParams::learningRate() const {
	return m_learningRate;
}

double XgBoostGeneticParams::subsample() const {
	return m_subsample;
}
double XgBoostGeneticParams::colsampleByTree() const {
	return m_colsampleByTree;
}

void XgBoostGeneticParams::setMaxDepth(int val) {
	if (val!=m_maxDepth) {
		m_maxDepth = val;
		emit maxDepthChanged(m_maxDepth);
	}
}

void XgBoostGeneticParams::setNEstimators(int val) {
	if (val!=m_nEstimators) {
		m_nEstimators = val;
		emit nEstimatorsChanged(m_nEstimators);
	}
}

void XgBoostGeneticParams::setLearningRate(double val) {
	if (val!=m_learningRate) {
		m_learningRate = val;
		emit learningRateChanged(m_learningRate);
	}
}
void XgBoostGeneticParams::setSubsample(double val) {
	if (val!=m_subsample) {
		m_subsample = val;
		emit subsampleChanged(m_subsample);
	}
}

void XgBoostGeneticParams::setColsampleByTree(double val) {
	if (val!=m_colsampleByTree) {
		m_colsampleByTree = val;
		emit colsampleByTreeChanged(m_colsampleByTree);
	}
}
