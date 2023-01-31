#include "fixedrgblayersfromdatasetandcuberep.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "fixedrgblayersfromdatasetandcubeproppanel.h"
#include "fixedrgblayersfromdatasetandcube3dlayer.h"
#include "fixedrgblayersfromdatasetandcubelayeronmap.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "seismic3dabstractdataset.h"
#include "affinetransformation.h"
#include "abstractinnerview.h"
#include "rgbinterleavedqglcudaimageitem.h"
#include <gccOnSpectrumAttributWidget.h>
#include <FixedRGBLayersFromDatasetAndCube3FreqPropPanel.h>


FixedRGBLayersFromDatasetAndCubeRep::FixedRGBLayersFromDatasetAndCubeRep(FixedRGBLayersFromDatasetAndCube *layer,
		AbstractInnerView *parent) :
		AbstractGraphicRep(parent),  IMouseImageDataProvider() {
	m_data = layer;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_layer3D=nullptr;
	m_name = m_data->name();

	connect(m_data->image(), SIGNAL(dataChanged()), this,
			SLOT(dataChangedAll()));
}


FixedRGBLayersFromDatasetAndCubeRep::~FixedRGBLayersFromDatasetAndCubeRep() {
	if (m_layer3D != nullptr)
		delete m_layer3D;
	if (m_propPanel!=nullptr)
		delete m_propPanel;
	if (m_layer != nullptr)
		delete m_layer;
}


FixedRGBLayersFromDatasetAndCube* FixedRGBLayersFromDatasetAndCubeRep::fixedRGBLayersFromDataset() const {
	return m_data;
}

void FixedRGBLayersFromDatasetAndCubeRep::dataChangedAll() {
	if (m_propPanel != nullptr) {
		m_propPanel->updatePalette(0);
		m_propPanel->updatePalette(1);
		m_propPanel->updatePalette(2);
	}
	if (m_layer3D != nullptr)
		m_layer3D->refresh();
}

void FixedRGBLayersFromDatasetAndCubeRep::dataChangedRed() {
	if (m_propPanel != nullptr) {
		m_propPanel->updatePalette(0);
	}
	if (m_layer3D != nullptr)
		m_layer3D->refresh();
}

void FixedRGBLayersFromDatasetAndCubeRep::dataChangedGreen() {
	if (m_propPanel != nullptr) {
		m_propPanel->updatePalette(1);
	}
	if (m_layer3D != nullptr)
		m_layer3D->refresh();
}

void FixedRGBLayersFromDatasetAndCubeRep::dataChangedBlue() {
	if (m_propPanel != nullptr) {
		m_propPanel->updatePalette(2);
	}
	if (m_layer3D != nullptr)
		m_layer3D->refresh();
}

IData* FixedRGBLayersFromDatasetAndCubeRep::data() const {
	return m_data;
}

QWidget* FixedRGBLayersFromDatasetAndCubeRep::propertyPanel() {
	if (m_propPanel == nullptr)
	{
		QString type = m_data->propertyPanelType();
		if ( type == "default" )
		{
			m_propPanel = new FixedRGBLayersFromDatasetAndCubePropPanel(this, m_parent);
			connect(m_propPanel, &QWidget::destroyed, [this]() {
				m_propPanel = nullptr;
			});
		}
		else
		{
			m_propPanel = new FixedRGBLayersFromDatasetAndCube3FreqPropPanel(this, m_parent);
			connect(m_propPanel, &QWidget::destroyed, [this]() {
				m_propPanel = nullptr;
			});
		}
	}

	return m_propPanel;
}
GraphicLayer* FixedRGBLayersFromDatasetAndCubeRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		m_data->initialize();
		m_layer = new FixedRGBLayersFromDatasetAndCubeLayerOnMap(this, scene, defaultZDepth, parent);
	}
	return m_layer;
}

Graphic3DLayer * FixedRGBLayersFromDatasetAndCubeRep::layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera)
{
	if (m_layer3D == nullptr) {
		m_layer3D = new FixedRGBLayersFromDatasetAndCube3DLayer(this, parent, root, camera);
	}
	return m_layer3D;
}


bool FixedRGBLayersFromDatasetAndCubeRep::mouseData(double x, double y, MouseInfo &info) {
	double v1=0, v2=0, v3=0;
	bool valid = m_data->image()->value(x, y, 0,
			info.i, info.j, v1);
	valid = valid && m_data->image()->value(x, y, 1,
			info.i, info.j, v2);
	valid = valid && m_data->image()->value(x, y, 2,
			info.i, info.j, v3);

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
	m_data->sampleTransformation()->direct(v1, realDepth);
	info.depth = realDepth;
	info.depthUnit = m_data->cubeSeismicAddon().getSampleUnit();

	return valid;
}

bool FixedRGBLayersFromDatasetAndCubeRep::setSampleUnit(SampleUnit sampleUnit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(sampleUnit);
}

QList<SampleUnit> FixedRGBLayersFromDatasetAndCubeRep::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_data->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString FixedRGBLayersFromDatasetAndCubeRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep FixedRGBLayersFromDatasetAndCubeRep::getTypeGraphicRep() {
    return AbstractGraphicRep::Image;
}

void FixedRGBLayersFromDatasetAndCubeRep::deleteGraphicItemDataContent(QGraphicsItem *item)
{
	deleteData(fixedRGBLayersFromDataset()->image(),item);
	deleteData(fixedRGBLayersFromDataset()->isoSurfaceHolder(),item);
}

QGraphicsObject* FixedRGBLayersFromDatasetAndCubeRep::cloneCUDAImageWithMask(QGraphicsItem *parent)
{
	RGBInterleavedQGLCUDAImageItem* outItem = new RGBInterleavedQGLCUDAImageItem(fixedRGBLayersFromDataset()->isoSurfaceHolder(),
				fixedRGBLayersFromDataset()->image(), 0,
				parent, true);

	outItem->setMinimumValue(m_data->minimumValue());
	outItem->setMinimumValueActive(m_data->isMinimumValueActive());

	return outItem;
}

void FixedRGBLayersFromDatasetAndCubeRep::computeGccOnSpectrum()
{
	qDebug() << m_data->name();
	qDebug() << m_data->dirPath();
	WorkingSetManager *manager = data()->workingSetManager();
	GccOnSpectrumAttributWidget *p = new GccOnSpectrumAttributWidget(m_data->surveyPath(), m_data->dirPath(), m_data->name(), manager);
	p->show();
}

void FixedRGBLayersFromDatasetAndCubeRep::buildContextMenu(QMenu *menu) {
	QAction *attribut = new QAction(tr("Compute gcc on spectrum"), this);
	menu->addAction(attribut);
	connect(attribut, SIGNAL(triggered()), this, SLOT(computeGccOnSpectrum()));
}

