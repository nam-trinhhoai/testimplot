#include "fixedlayerfromdatasetreponslice.h"

#include "cudaimagepaletteholder.h"
#include "fixedlayerfromdatasetlayeronslice.h"
#include "qgllineitem.h"
#include "fixedlayerfromdataset.h"
#include "seismic3dabstractdataset.h"
#include "affinetransformation.h"
#include "abstractinnerview.h"
#include "fixedlayerfromdatasetproppanelonslice.h"

FixedLayerFromDatasetRepOnSlice::FixedLayerFromDatasetRepOnSlice(FixedLayerFromDataset *fixedLayer, const IGeorefImage * const transfoProvider, SliceDirection dir,AbstractInnerView *parent) :
		AbstractGraphicRep(parent),ISliceableRep(),m_transfoProvider(transfoProvider) {
	m_fixedLayer = fixedLayer;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_dir=dir;
	m_name = m_fixedLayer->name();
	m_currentSlice=0;

	connect(m_fixedLayer, &FixedLayerFromDataset::colorChanged, this, &FixedLayerFromDatasetRepOnSlice::setLayerColor);
}

FixedLayerFromDatasetRepOnSlice::~FixedLayerFromDatasetRepOnSlice() {
	disconnect(m_fixedLayer, &FixedLayerFromDataset::colorChanged, this, &FixedLayerFromDatasetRepOnSlice::setLayerColor);
	if (m_layer != nullptr)
		delete m_layer;
	if (m_propPanel)
		delete m_propPanel;
}

FixedLayerFromDataset* FixedLayerFromDatasetRepOnSlice::fixedLayer() const {
	return m_fixedLayer;
}

void FixedLayerFromDatasetRepOnSlice::setSliceIJPosition(int imageVal)
{
	m_currentSlice=imageVal;
	if(m_layer!=nullptr)
		m_layer->setSliceIJPosition(imageVal);
}

IData* FixedLayerFromDatasetRepOnSlice::data() const {
	return m_fixedLayer;
}

QWidget* FixedLayerFromDatasetRepOnSlice::propertyPanel() {
	if (m_propPanel == nullptr) {
		m_propPanel = new FixedLayerFromDatasetPropPanelOnSlice(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}
	return m_propPanel;
}
GraphicLayer* FixedLayerFromDatasetRepOnSlice::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		m_layer = new FixedLayerFromDatasetLayerOnSlice(this,m_dir,m_transfoProvider,m_currentSlice, scene, defaultZDepth, parent);
		m_layer->setPenColor(m_fixedLayer->getColor());
	}
	return m_layer;
}

void FixedLayerFromDatasetRepOnSlice::setLayerColor(QColor color) {
	if (m_layer!=nullptr) {
		m_layer->setPenColor(color);
	}
}

bool FixedLayerFromDatasetRepOnSlice::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> FixedLayerFromDatasetRepOnSlice::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_fixedLayer->dataset()->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString FixedLayerFromDatasetRepOnSlice::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep FixedLayerFromDatasetRepOnSlice::getTypeGraphicRep() {
    return AbstractGraphicRep::Courbe;
}
