#ifndef SRC_SLICER_GRAPHICSREP_MULTISEEDHORIZON_REFERENCECURVEWRAPPER_H
#define SRC_SLICER_GRAPHICSREP_MULTISEEDHORIZON_REFERENCECURVEWRAPPER_H

#include "curve.h"

#include <QColor>
#include <QObject>

#include <memory>

class FixedLayerFromDataset;

class ReferenceCurveWrapper : public QObject {
	Q_OBJECT
public:
	ReferenceCurveWrapper(FixedLayerFromDataset* layer, QObject* parent=0);
	~ReferenceCurveWrapper();

	Curve* curve();
	void setCurve(Curve* curve);
	FixedLayerFromDataset* layer();

private slots:
	void updateColor(QColor color);
	void layerDestroyed();

private:
	std::unique_ptr<Curve> m_curve;
	FixedLayerFromDataset* m_layer;
};

#endif
