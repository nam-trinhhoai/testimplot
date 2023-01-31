#include "fixedlayerfromdatasetreponrandom.h"

#include "cudaimagepaletteholder.h"
#include "fixedlayerfromdatasetlayeronrandom.h"
#include "qgllineitem.h"
#include "fixedlayerfromdataset.h"
#include "seismic3dabstractdataset.h"
#include "affinetransformation.h"
#include "abstractinnerview.h"
#include "fixedlayerfromdatasetproppanelonrandom.h"

FixedLayerFromDatasetRepOnRandom::FixedLayerFromDatasetRepOnRandom(FixedLayerFromDataset *fixedLayer,AbstractInnerView *parent) :
		AbstractGraphicRep(parent) {
	m_fixedLayer = fixedLayer;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_name = m_fixedLayer->name();

	connect(m_fixedLayer, &FixedLayerFromDataset::colorChanged, this, &FixedLayerFromDatasetRepOnRandom::setLayerColor);
}

FixedLayerFromDatasetRepOnRandom::~FixedLayerFromDatasetRepOnRandom() {
	disconnect(m_fixedLayer, &FixedLayerFromDataset::colorChanged, this, &FixedLayerFromDatasetRepOnRandom::setLayerColor);

	if (m_layer != nullptr)
		delete m_layer;
	if (m_propPanel)
		delete m_propPanel;
}

FixedLayerFromDataset* FixedLayerFromDatasetRepOnRandom::fixedLayer() const {
	return m_fixedLayer;
}

IData* FixedLayerFromDatasetRepOnRandom::data() const {
	return m_fixedLayer;
}

QWidget* FixedLayerFromDatasetRepOnRandom::propertyPanel() {
	if (m_propPanel == nullptr) {
		m_propPanel = new FixedLayerFromDatasetPropPanelOnRandom(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}
	return m_propPanel;
}

GraphicLayer* FixedLayerFromDatasetRepOnRandom::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		m_layer = new FixedLayerFromDatasetLayerOnRandom(this, scene, defaultZDepth, parent);
		m_layer->setPenColor(m_fixedLayer->getColor());
	}
	return m_layer;
}

void FixedLayerFromDatasetRepOnRandom::setLayerColor(QColor color) {
	if (m_layer!=nullptr) {
		m_layer->setPenColor(color);
	}
}

bool FixedLayerFromDatasetRepOnRandom::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> FixedLayerFromDatasetRepOnRandom::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_fixedLayer->dataset()->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString FixedLayerFromDatasetRepOnRandom::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep FixedLayerFromDatasetRepOnRandom::getTypeGraphicRep() {
    return AbstractGraphicRep::Courbe;
}

void FixedLayerFromDatasetRepOnRandom::deleteLayer(){
    if (m_layer != nullptr) {
        delete m_layer;
        m_layer = nullptr;
    }

    if (m_propPanel != nullptr){
        delete m_propPanel;
        m_propPanel = nullptr;
    }
}
