#include "fixedrgblayersfromdatasetrep.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "fixedrgblayersfromdatasetproppanel.h"
#include "fixedrgblayersfromdataset3dlayer.h"
#include "fixedrgblayersfromdatasetlayeronmap.h"
#include "fixedrgblayersfromdataset.h"
#include "seismic3dabstractdataset.h"
#include "affinetransformation.h"
#include "abstractinnerview.h"


FixedRGBLayersFromDatasetRep::FixedRGBLayersFromDatasetRep(FixedRGBLayersFromDataset *layer,
		AbstractInnerView *parent) :
		AbstractGraphicRep(parent),  IMouseImageDataProvider() {
	m_data = layer;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_layer3D=nullptr;
	m_name = m_data->name();

	connect(m_data->image()->get(0), SIGNAL(dataChanged()), this,
			SLOT(dataChangedRed()));
	connect(m_data->image()->get(1), SIGNAL(dataChanged()), this,
			SLOT(dataChangedGreen()));
	connect(m_data->image()->get(2), SIGNAL(dataChanged()), this,
			SLOT(dataChangedBlue()));
}


FixedRGBLayersFromDatasetRep::~FixedRGBLayersFromDatasetRep() {
	if (m_layer3D != nullptr)
		delete m_layer3D;
	if (m_propPanel!=nullptr)
		delete m_propPanel;
	if (m_layer != nullptr)
		delete m_layer;
}


FixedRGBLayersFromDataset* FixedRGBLayersFromDatasetRep::fixedRGBLayersFromDataset() const {
	return m_data;
}

void FixedRGBLayersFromDatasetRep::dataChangedRed() {
	if (m_propPanel != nullptr) {
		m_propPanel->updatePalette(0);
	}
	if (m_layer3D != nullptr)
		m_layer3D->refresh();
}

void FixedRGBLayersFromDatasetRep::dataChangedGreen() {
	if (m_propPanel != nullptr) {
		m_propPanel->updatePalette(1);
	}
	if (m_layer3D != nullptr)
		m_layer3D->refresh();
}

void FixedRGBLayersFromDatasetRep::dataChangedBlue() {
	if (m_propPanel != nullptr) {
		m_propPanel->updatePalette(2);
	}
	if (m_layer3D != nullptr)
		m_layer3D->refresh();
}

IData* FixedRGBLayersFromDatasetRep::data() const {
	return m_data;
}

QWidget* FixedRGBLayersFromDatasetRep::propertyPanel() {
	if (m_propPanel == nullptr)
	{
		m_propPanel = new FixedRGBLayersFromDatasetPropPanel(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer* FixedRGBLayersFromDatasetRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		m_layer = new FixedRGBLayersFromDatasetLayerOnMap(this, scene, defaultZDepth, parent);
	}
	return m_layer;
}

Graphic3DLayer * FixedRGBLayersFromDatasetRep::layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera)
{
	if (m_layer3D == nullptr) {
		m_layer3D = new FixedRGBLayersFromDataset3DLayer(this, parent, root, camera);
	}
	return m_layer3D;
}


bool FixedRGBLayersFromDatasetRep::mouseData(double x, double y, MouseInfo &info) {
	double v1, v2, v3;
	bool valid = IGeorefImage::value(m_data->image()->get(0), x, y,
			info.i, info.j, v1);
	valid = IGeorefImage::value(m_data->image()->get(1), x, y, info.i,
			info.j, v2);
	valid = IGeorefImage::value(m_data->image()->get(2), x, y, info.i,
			info.j, v3);

	info.values.push_back(v1);
	info.values.push_back(v2);
	info.values.push_back(v3);

	info.valuesDesc.push_back("Red");
	info.valuesDesc.push_back("Green");
	info.valuesDesc.push_back("Blue");
	info.depthValue = true;

	IGeorefImage::value(m_data->isoSurfaceHolder(), x, y, info.i, info.j,
			v1);
	double realDepth;
	m_data->dataset()->sampleTransformation()->direct(v1, realDepth);
	info.depth = realDepth;
	info.depthUnit = m_data->dataset()->cubeSeismicAddon().getSampleUnit();

	return valid;
}

bool FixedRGBLayersFromDatasetRep::setSampleUnit(SampleUnit sampleUnit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(sampleUnit);
}

QList<SampleUnit> FixedRGBLayersFromDatasetRep::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_data->dataset()->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString FixedRGBLayersFromDatasetRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep FixedRGBLayersFromDatasetRep::getTypeGraphicRep() {
    return AbstractGraphicRep::Image;
}
