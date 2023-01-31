#include "stratislicefrequencyrep.h"
#include "cudaimagepaletteholder.h"
#include "stratislicefrequencyproppanel.h"
#include "stratislicefrequencylayer.h"
#include "stratislicefrequency3Dlayer.h"
#include "qgllineitem.h"
#include "datacontroler.h"
#include "slicepositioncontroler.h"
#include "stratislice.h"
#include "seismic3dabstractdataset.h"
#include "affinetransformation.h"
#include "abstractinnerview.h"
#include "frequencystratisliceattribute.h"

StratiSliceFrequencyRep::StratiSliceFrequencyRep(FrequencyStratiSliceAttribute *stratislice, AbstractInnerView *parent) :
		AbstractGraphicRep(parent),  IMouseImageDataProvider() {
	m_stratislice = stratislice;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_layer3D=nullptr;
	m_showCrossHair = false;
	m_name = m_stratislice->name();

	connect(m_stratislice->image(), SIGNAL(dataChanged()), this,
			SLOT(dataChanged()));
}

StratiSliceFrequencyRep::~StratiSliceFrequencyRep() {
	if (m_layer != nullptr)
		delete m_layer;
	if (m_layer3D != nullptr)
		delete m_layer3D;
	if (m_propPanel != nullptr)
		delete m_propPanel;
}

FrequencyStratiSliceAttribute* StratiSliceFrequencyRep::stratiSliceAttribute() const {
	return m_stratislice;
}

void StratiSliceFrequencyRep::dataChanged() {
	if (m_propPanel != nullptr)
		m_propPanel->updatePalette();
	if (m_layer != nullptr)
		m_layer->refresh();
	if (m_layer3D != nullptr)
		m_layer3D->refresh();
}

IData* StratiSliceFrequencyRep::data() const {
	return m_stratislice;
}

QWidget* StratiSliceFrequencyRep::propertyPanel() {
	if (m_propPanel == nullptr) {
		m_stratislice->initialize();
		m_propPanel = new StratiSliceFrequencyPropPanel(this,
				m_parent->viewType() == ViewType::View3D, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}
	return m_propPanel;
}
GraphicLayer* StratiSliceFrequencyRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		m_stratislice->initialize();
		m_layer = new StratiSliceFrequencyLayer(this, scene, defaultZDepth, parent);
		m_layer->showCrossHair(m_showCrossHair);
	}
	return m_layer;
}

Graphic3DLayer* StratiSliceFrequencyRep::layer3D(QWindow *parent, Qt3DCore::QEntity *root,
		Qt3DRender::QCamera *camera) {
	if (m_layer3D == nullptr) {
		m_stratislice->initialize();
		m_layer3D = new StratiSliceFrequency3DLayer(this, parent, root, camera);
	}
	return m_layer3D;
}

void StratiSliceFrequencyRep::showCrossHair(bool val) {
	m_showCrossHair = val;
	m_layer->showCrossHair(m_showCrossHair);
}

bool StratiSliceFrequencyRep::crossHair() const {
	return m_showCrossHair;
}


bool StratiSliceFrequencyRep::mouseData(double x, double y, MouseInfo &info) {
	double value;
	bool valid = IGeorefImage::value(m_stratislice->image(), x, y, info.i,
			info.j, value);
	info.valuesDesc.push_back("Attribute");
	info.values.push_back(value);
	info.depthValue = true;
	IGeorefImage::value(m_stratislice->isoSurfaceHolder(), x, y, info.i, info.j,
			value);
	double realDepth;
	m_stratislice->stratiSlice()->seismic()->sampleTransformation()->direct(value, realDepth);
	info.depth = realDepth;
	info.depthUnit = m_stratislice->stratiSlice()->seismic()->cubeSeismicAddon().getSampleUnit();
	return valid;
}

bool StratiSliceFrequencyRep::setSampleUnit(SampleUnit sampleUnit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(sampleUnit);
}

QList<SampleUnit> StratiSliceFrequencyRep::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_stratislice->stratiSlice()->seismic()->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString StratiSliceFrequencyRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep StratiSliceFrequencyRep::getTypeGraphicRep() {
    return Image;
}

