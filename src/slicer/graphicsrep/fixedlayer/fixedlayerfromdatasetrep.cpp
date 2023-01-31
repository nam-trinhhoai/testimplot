#include "fixedlayerfromdatasetrep.h"
#include "fixedlayerfromdataset.h"
#include "fixedlayerfromdatasetproppanel.h"
#include "fixedlayerfromdatasetlayer.h"
#include "seismic3dabstractdataset.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "abstractinnerview.h"
#include "colortableregistry.h"

FixedLayerFromDatasetRep::FixedLayerFromDatasetRep(FixedLayerFromDataset * fixedLayer, AbstractInnerView *parent) :
		AbstractGraphicRep(parent),  IMouseImageDataProvider() {
	m_fixedLayer = fixedLayer;
	m_currentIso = new CUDAImagePaletteHolder(m_fixedLayer->width(), m_fixedLayer->depth(), ImageFormats::QSampleType::FLOAT32,
				m_fixedLayer->dataset()->ijToXYTransfo(), nullptr);
	m_currentAttribute = new CUDAImagePaletteHolder(m_fixedLayer->width(), m_fixedLayer->depth(), ImageFormats::QSampleType::FLOAT32,
				m_fixedLayer->dataset()->ijToXYTransfo(), nullptr);

	connect(m_fixedLayer, &FixedLayerFromDataset::propertyModified, this, &FixedLayerFromDatasetRep::updateAttribute);
	connect(m_fixedLayer, &FixedLayerFromDataset::propertyModified, this, &FixedLayerFromDatasetRep::updateIso);
	connect(m_fixedLayer, &FixedLayerFromDataset::newPropertyCreated, this, &FixedLayerFromDatasetRep::initProperties);

	if (fixedLayer->keys().size()==0) {
		chooseIsochrone();
	} else {
		QVector<QString> keys = fixedLayer->keys();
		initProperties(keys[keys.size()-1]);
	}
}

FixedLayerFromDatasetRep::~FixedLayerFromDatasetRep() {
	delete m_currentIso;
	delete m_currentAttribute;
	if (m_propPanel!=nullptr)
		delete m_propPanel;
	if (m_layer!=nullptr)
		delete m_layer;
}

QString FixedLayerFromDatasetRep::name() const {
	return m_fixedLayer->name();
}

FixedLayerFromDataset* FixedLayerFromDatasetRep::fixedLayer() const {
	return m_fixedLayer;
}

void FixedLayerFromDatasetRep::showCrossHair(bool val) {
	m_showCrossHair = val;
	m_layer->showCrossHair(m_showCrossHair);
}

bool FixedLayerFromDatasetRep::crossHair() const {
	return m_showCrossHair;
}

//AbstractGraphicRep
QWidget* FixedLayerFromDatasetRep::propertyPanel() {
	if (m_propPanel == nullptr) {
		m_propPanel = new FixedLayerFromDatasetPropPanel(this,
					m_parent->viewType() == ViewType::View3D, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this](){
			m_propPanel = nullptr;
		});
	}
	return m_propPanel;
}

GraphicLayer * FixedLayerFromDatasetRep::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		m_layer = new FixedLayerFromDatasetLayer(this, scene, defaultZDepth, parent);
		m_layer->showCrossHair(m_showCrossHair);
	}
	return m_layer;
}

IData* FixedLayerFromDatasetRep::data() const {
	return m_fixedLayer;
}

//IMouseImageDataProvider
bool FixedLayerFromDatasetRep::mouseData(double x, double y,MouseInfo & info) {
	double value;
	bool valid = IGeorefImage::value(m_currentAttribute, x, y, info.i,
			info.j, value);
	info.valuesDesc.push_back("Attribute");
	info.values.push_back(value);
	info.depthValue = true;
	IGeorefImage::value(m_currentIso, x, y, info.i, info.j,
			value);
//	double realDepth;
//	m_fixedLayer->dataset()->sampleTransformation()->direct(value, realDepth);
	info.depth = value;
	info.depthUnit = m_fixedLayer->dataset()->cubeSeismicAddon().getSampleUnit();
	return valid;
}

void FixedLayerFromDatasetRep::dataChanged() {
	if (m_propPanel != nullptr)
		m_propPanel->updatePalette();
	if (m_layer != nullptr)
		m_layer->refresh();
}

void FixedLayerFromDatasetRep::chooseAttribute(QString attributeName) {
	m_currentAttributeName = attributeName;
	CUDAImagePaletteHolder* attribute = m_fixedLayer->image(attributeName);
	if (attribute!=nullptr) {
		attribute->lockPointer();
		m_currentAttribute->updateTexture(attribute->backingPointer(), false);
		attribute->unlockPointer();

		dataChanged();
		updateAttributePalette();

		if (m_propPanel!=nullptr) {
			m_propPanel->updateComboValue(m_currentAttributeName);
		}
	}
}

void FixedLayerFromDatasetRep::updateAttribute(QString propName) {
	if (m_currentAttributeName.compare(propName)==0) {
		chooseAttribute(m_currentAttributeName);
	}
}

void FixedLayerFromDatasetRep::chooseIsochrone() {
	m_currentIsoName = FixedLayerFromDataset::ISOCHRONE;
	CUDAImagePaletteHolder* iso = m_fixedLayer->image(FixedLayerFromDataset::ISOCHRONE);
	if (iso!=nullptr) {
		iso->lockPointer();
		m_currentIso->updateTexture(iso->backingPointer(), false);
		iso->unlockPointer();
	}
}

void FixedLayerFromDatasetRep::updateIso(QString propName) {
	if (m_currentIsoName.compare(propName)==0) {
		chooseIsochrone();
	}
}

void FixedLayerFromDatasetRep::updateAttributePalette() {
	if (m_currentAttributeName.compare(FixedLayerFromDataset::ISOCHRONE)==0) {
		m_currentAttribute->setRange(QVector2D(0, std::numeric_limits<short>::max()));
		m_currentAttribute->setLookupTable(ColorTableRegistry::PALETTE_REGISTRY().findColorTable("CLASSIC",
                "Iles-2"));
	} else {
		m_currentAttribute->setRange(QVector2D(std::numeric_limits<short>::min(), std::numeric_limits<short>::max()));
		m_currentAttribute->setLookupTable(ColorTableRegistry::PALETTE_REGISTRY().findColorTable("CLASSIC",
		        "Black-White"));
	}
}

void FixedLayerFromDatasetRep::initProperties(QString propName) {
	if (m_currentAttributeName.isNull() || m_currentAttributeName.isEmpty()) {
		chooseAttribute(propName);
	}
	if (FixedLayerFromDataset::ISOCHRONE.compare(propName)==0) {
		chooseIsochrone();
	}
}

bool FixedLayerFromDatasetRep::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> FixedLayerFromDatasetRep::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_fixedLayer->dataset()->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString FixedLayerFromDatasetRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep FixedLayerFromDatasetRep::getTypeGraphicRep() {
    return AbstractGraphicRep::Image;
}

void FixedLayerFromDatasetRep::deleteGraphicItemDataContent(QGraphicsItem* item) {
	m_fixedLayer->deleteGraphicItemDataContent(item);
}
