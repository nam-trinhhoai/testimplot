#include "densegeneticparams.h"


DenseGeneticParams::DenseGeneticParams(QObject* parent) : QObject(parent) {

}

DenseGeneticParams::~DenseGeneticParams() {

}

QVector<unsigned int> DenseGeneticParams::layerSizes() const {
	return m_layerSizes;
}

void DenseGeneticParams::setLayerSizes(const QVector<unsigned int>& array) {
	if (m_layerSizes!=array) {
		m_layerSizes = array;
		emit layerSizesChanged(m_layerSizes);
	}
}

bool DenseGeneticParams::useDropout() const {
	return m_useDropout;
}

void DenseGeneticParams::toggleDropout(bool val) {
	if (m_useDropout!=val) {
		m_useDropout = val;
		emit useDropoutChanged(m_useDropout);
	}
}

double DenseGeneticParams::dropout() const {
	return m_dropout;
}

void DenseGeneticParams::setDropout(double val) {
	if (m_dropout!=val) {
		m_dropout = val;
		emit dropoutChanged(m_dropout);
	}
}

bool DenseGeneticParams::useNormalisation() const {
	return m_useNormalisation;
}

void DenseGeneticParams::toggleNormalisation(bool val) {
	if (m_useNormalisation!=val) {
		m_useNormalisation = val;
		emit useNormalisationChanged(m_useNormalisation);
	}
}

Activation DenseGeneticParams::activation() const {
	return m_activation;
}

void DenseGeneticParams::setActivation(Activation val) {
	if (m_activation!=val) {
		m_activation = val;
		emit activationChanged(m_activation);
	}
}
