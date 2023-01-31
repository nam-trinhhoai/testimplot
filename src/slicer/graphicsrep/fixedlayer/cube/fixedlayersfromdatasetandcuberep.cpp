#include "fixedlayersfromdatasetandcuberep.h"
#include "cudaimagepaletteholder.h"
#include "fixedlayersfromdatasetandcubeproppanel.h"
#include "fixedlayersfromdatasetandcube3dlayer.h"
#include "fixedlayersfromdatasetandcubelayeronmap.h"
#include "fixedlayersfromdatasetandcube.h"
#include "seismic3dabstractdataset.h"
#include "affinetransformation.h"
#include "abstractinnerview.h"
#include "cpuimagepaletteholder.h"
#include <fixedlayersfreehorizonproppanel.h>


FixedLayersFromDatasetAndCubeRep::FixedLayersFromDatasetAndCubeRep(FixedLayersFromDatasetAndCube *layer,
		AbstractInnerView *parent) :
		AbstractGraphicRep(parent),  IMouseImageDataProvider() {
	m_data = layer;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_layer3D=nullptr;
	m_name = m_data->name();

	connect(m_data->image(), SIGNAL(dataChanged()), this,
			SLOT(dataChanged()));
}


FixedLayersFromDatasetAndCubeRep::~FixedLayersFromDatasetAndCubeRep() {
	if (m_layer3D != nullptr)
		delete m_layer3D;
	if (m_propPanel!=nullptr)
		delete m_propPanel;
	if (m_layer != nullptr)
		delete m_layer;
}


FixedLayersFromDatasetAndCube* FixedLayersFromDatasetAndCubeRep::fixedLayersFromDataset() const {
	return m_data;
}

void FixedLayersFromDatasetAndCubeRep::dataChanged() {
	if (m_propPanel != nullptr) {
		m_propPanel->updatePalette();
	}
	if (m_layer3D != nullptr)
		m_layer3D->refresh();
}

IData* FixedLayersFromDatasetAndCubeRep::data() const {
	return m_data;
}

QWidget* FixedLayersFromDatasetAndCubeRep::propertyPanel() {
	if (m_propPanel == nullptr)
	{
		QString type = m_data->propertyPanelType();
		if ( type == "default" )
		{
			m_propPanel = new FixedLayersFromDatasetAndCubePropPanel(this, m_parent);
			connect(m_propPanel, &QWidget::destroyed, [this]() {
				m_propPanel = nullptr;
			});
		}
		else if ( type == "freehorizon" )
		{	m_propPanel = new FixedLayersFreeHorizonPropPanel(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
		}
	}
	return m_propPanel;
}

GraphicLayer* FixedLayersFromDatasetAndCubeRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		m_data->initialize();
		m_layer = new FixedLayersFromDatasetAndCubeLayerOnMap(this, scene, defaultZDepth, parent);
	}
	return m_layer;
	// return nullptr;
}

Graphic3DLayer * FixedLayersFromDatasetAndCubeRep::layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera)
{
	if (m_layer3D == nullptr) {
		m_layer3D = new FixedLayersFromDatasetAndCube3DLayer(this, parent, root, camera);
	}
	return m_layer3D;
}


bool FixedLayersFromDatasetAndCubeRep::mouseData(double x, double y, MouseInfo &info) {
	double v1;
	bool valid = IGeorefImage::value(m_data->image(), x, y,
			info.i, info.j, v1);

	info.values.push_back(v1);

	info.valuesDesc.push_back("Attribute");
	info.depthValue = true;

	IGeorefImage::value(m_data->isoSurfaceHolder(), x, y, info.i, info.j,
			v1);
	double realDepth;
	m_data->sampleTransformation()->direct(v1, realDepth);
	info.depth = realDepth;
	info.depthUnit = m_data->cubeSeismicAddon().getSampleUnit();

	return valid;
}

bool FixedLayersFromDatasetAndCubeRep::setSampleUnit(SampleUnit sampleUnit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(sampleUnit);
}

QList<SampleUnit> FixedLayersFromDatasetAndCubeRep::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_data->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString FixedLayersFromDatasetAndCubeRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep FixedLayersFromDatasetAndCubeRep::getTypeGraphicRep() {
	return AbstractGraphicRep::Image;
}
