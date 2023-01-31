#include "fixedlayerfromdataset.h"
#include "referencecurvewrapper.h"

ReferenceCurveWrapper::ReferenceCurveWrapper(FixedLayerFromDataset* layer, QObject* parent) : QObject(parent) {
	m_layer = layer;

	if (m_layer) {
		connect(m_layer, &FixedLayerFromDataset::destroyed, this, &ReferenceCurveWrapper::layerDestroyed);
		connect(m_layer, &FixedLayerFromDataset::colorChanged, this, &ReferenceCurveWrapper::updateColor);
	}
}

ReferenceCurveWrapper::~ReferenceCurveWrapper() {
	if (m_layer) {
		disconnect(m_layer, &FixedLayerFromDataset::destroyed, this, &ReferenceCurveWrapper::layerDestroyed);
		disconnect(m_layer, &FixedLayerFromDataset::colorChanged, this, &ReferenceCurveWrapper::updateColor);
	}
}

Curve* ReferenceCurveWrapper::curve() {
	return m_curve.get();
}

void ReferenceCurveWrapper::setCurve(Curve* curve) {
	m_curve.reset(curve);
	if (m_layer) {
		QPen pen = m_curve->getPen();
		pen.setColor(m_layer->getColor());
		m_curve->setPen(pen);
	}
}

FixedLayerFromDataset* ReferenceCurveWrapper::layer() {
	return m_layer;
}

void ReferenceCurveWrapper::updateColor(QColor color) {
	if (m_curve!=nullptr) {
		QPen pen = m_curve->getPen();
		pen.setColor(color);
		m_curve->setPen(pen);
	}
}

void ReferenceCurveWrapper::layerDestroyed() {
	if (m_layer) {
		disconnect(m_layer, &FixedLayerFromDataset::destroyed, this, &ReferenceCurveWrapper::layerDestroyed);
		disconnect(m_layer, &FixedLayerFromDataset::colorChanged, this, &ReferenceCurveWrapper::updateColor);

		m_layer = nullptr;
	}
}
