#include "stratislicergbattributerep.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "stratislicergbproppanel.h"
#include "stratislicergblayer.h"
#include "stratislicergb3Dlayer.h"
#include "rgbstratisliceattribute.h"
#include "seismic3dabstractdataset.h"
#include "affinetransformation.h"
#include "abstractinnerview.h"
#include "stratislice.h"


StratiSliceRGBAttributeRep::StratiSliceRGBAttributeRep(RGBStratiSliceAttribute *stratislice,
		AbstractInnerView *parent) :
		AbstractGraphicRep(parent),  IMouseImageDataProvider() {
	m_stratislice = stratislice;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_layer3D=nullptr;
	m_name = m_stratislice->name();

	connect(m_stratislice->image()->get(0), SIGNAL(dataChanged()), this,
			SLOT(dataChangedRed()));
	connect(m_stratislice->image()->get(1), SIGNAL(dataChanged()), this,
			SLOT(dataChangedGreen()));
	connect(m_stratislice->image()->get(2), SIGNAL(dataChanged()), this,
			SLOT(dataChangedBlue()));
}


StratiSliceRGBAttributeRep::~StratiSliceRGBAttributeRep() {
	if (m_layer != nullptr)
		delete m_layer;
	if (m_layer3D != nullptr)
		delete m_layer3D;
	if (m_propPanel != nullptr)
		delete m_propPanel;
}


RGBStratiSliceAttribute* StratiSliceRGBAttributeRep::stratiSliceAttribute() const {
	return m_stratislice;
}

void StratiSliceRGBAttributeRep::dataChangedRed() {
	if (m_propPanel != nullptr) {
		m_propPanel->updatePalette(0);
	}
	if (m_layer != nullptr)
		m_layer->refresh();
}

void StratiSliceRGBAttributeRep::dataChangedGreen() {
	if (m_propPanel != nullptr) {
		m_propPanel->updatePalette(1);
	}
	if (m_layer != nullptr)
		m_layer->refresh();
}

void StratiSliceRGBAttributeRep::dataChangedBlue() {
	if (m_propPanel != nullptr) {
		m_propPanel->updatePalette(2);
	}
	if (m_layer != nullptr)
		m_layer->refresh();
}

IData* StratiSliceRGBAttributeRep::data() const {
	return m_stratislice;
}

QWidget* StratiSliceRGBAttributeRep::propertyPanel() {
	if (m_propPanel == nullptr)
	{
		m_stratislice->initialize();
		m_propPanel = new StratiSliceRGBPropPanel(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer* StratiSliceRGBAttributeRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr)
	{
		m_stratislice->initialize();
		m_layer = new StratiSliceRGBLayer(this, scene, defaultZDepth, parent);
	}

	return m_layer;
}

Graphic3DLayer * StratiSliceRGBAttributeRep::layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera)
{
	if (m_layer3D == nullptr) {
		m_stratislice->initialize();
		m_layer3D = new StratiSliceRGB3DLayer(this, parent, root, camera);
	}
	return m_layer3D;
}


bool StratiSliceRGBAttributeRep::mouseData(double x, double y, MouseInfo &info) {
	double v1, v2, v3;
	bool valid = IGeorefImage::value(m_stratislice->image()->get(0), x, y,
			info.i, info.j, v1);
	valid = IGeorefImage::value(m_stratislice->image()->get(1), x, y, info.i,
			info.j, v2);
	valid = IGeorefImage::value(m_stratislice->image()->get(2), x, y, info.i,
			info.j, v3);

	info.values.push_back(v1);
	info.values.push_back(v2);
	info.values.push_back(v3);

	info.valuesDesc.push_back("Red");
	info.valuesDesc.push_back("Green");
	info.valuesDesc.push_back("Blue");
	info.depthValue = true;

	IGeorefImage::value(m_stratislice->isoSurfaceHolder(), x, y, info.i, info.j,
			v1);
	double realDepth;
	m_stratislice->stratiSlice()->seismic()->sampleTransformation()->direct(v1, realDepth);
	info.depth = realDepth;
	info.depthUnit = m_stratislice->stratiSlice()->seismic()->cubeSeismicAddon().getSampleUnit();

	return valid;
}

bool StratiSliceRGBAttributeRep::setSampleUnit(SampleUnit sampleUnit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(sampleUnit);
}

QList<SampleUnit> StratiSliceRGBAttributeRep::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_stratislice->stratiSlice()->seismic()->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString StratiSliceRGBAttributeRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep StratiSliceRGBAttributeRep::getTypeGraphicRep() {
    return AbstractGraphicRep::Image;
}
