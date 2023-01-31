#include "fixedrgblayersfromdatasetreponslice.h"

#include "fixedrgblayersfromdatasetproppanelonslice.h"
#include "fixedrgblayersfromdataset.h"
#include "abstractinnerview.h"
#include "fixedrgblayersfromdatasetlayeronslice.h"
#include "cubeseismicaddon.h"
#include "seismic3dabstractdataset.h"

FixedRGBLayersFromDatasetRepOnSlice::FixedRGBLayersFromDatasetRepOnSlice(FixedRGBLayersFromDataset *fixedLayer,const IGeorefImage * const transfoProvider, SliceDirection dir,
		AbstractInnerView *parent) :
		AbstractGraphicRep(parent),ISliceableRep(),m_transfoProvider(transfoProvider){
	m_fixedLayer = fixedLayer;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_dir=dir;
	m_name = m_fixedLayer->name();
	m_currentSlice=0;
}


FixedRGBLayersFromDatasetRepOnSlice::~FixedRGBLayersFromDatasetRepOnSlice() {
	if (m_layer != nullptr)
		delete m_layer;
	if (m_propPanel != nullptr)
		delete m_propPanel;
}

FixedRGBLayersFromDataset* FixedRGBLayersFromDatasetRepOnSlice::fixedRGBLayersFromDataset() const {
	return m_fixedLayer;
}

void FixedRGBLayersFromDatasetRepOnSlice::setSliceIJPosition(int imageVal)
{
	m_currentSlice=imageVal;
	if(m_layer!=nullptr)
		m_layer->setSliceIJPosition(imageVal);
}

IData* FixedRGBLayersFromDatasetRepOnSlice::data() const {
	return m_fixedLayer;
}


QWidget* FixedRGBLayersFromDatasetRepOnSlice::propertyPanel() {
	if (m_propPanel == nullptr)
	{
		m_propPanel = new FixedRGBLayersFromDatasetPropPanelOnSlice(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer* FixedRGBLayersFromDatasetRepOnSlice::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr)
	{
		m_layer = new FixedRGBLayersFromDatasetLayerOnSlice(this,m_dir,m_transfoProvider,m_currentSlice, scene, defaultZDepth, parent);
	}
	return m_layer;
}

bool FixedRGBLayersFromDatasetRepOnSlice::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> FixedRGBLayersFromDatasetRepOnSlice::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_fixedLayer->dataset()->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString FixedRGBLayersFromDatasetRepOnSlice::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep FixedRGBLayersFromDatasetRepOnSlice::getTypeGraphicRep() {
    return AbstractGraphicRep::Courbe;
}
