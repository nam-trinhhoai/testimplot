#include "rgblayerfromdatasetrep.h"
#include "rgblayerfromdataset.h"
#include "rgblayerfromdatasetproppanel.h"
#include "rgblayerfromdatasetlayer.h"
#include "seismic3dabstractdataset.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "abstractinnerview.h"
#include "colortableregistry.h"
#include "cudargbimage.h"

RgbLayerFromDatasetRep::RgbLayerFromDatasetRep(RgbLayerFromDataset * fixedLayer, AbstractInnerView *parent) :
		AbstractGraphicRep(parent),  IMouseImageDataProvider() {
	m_fixedLayer = fixedLayer;
	m_currentIso = new CUDAImagePaletteHolder(m_fixedLayer->width(), m_fixedLayer->depth(), ImageFormats::QSampleType::FLOAT32,
				m_fixedLayer->dataset()->ijToXYTransfo(), nullptr);
	m_currentAttribute = new CUDARGBImage(m_fixedLayer->width(), m_fixedLayer->depth(), ImageFormats::QSampleType::FLOAT32,
				m_fixedLayer->dataset()->ijToXYTransfo(), nullptr);

	connect(m_fixedLayer, &RgbLayerFromDataset::propertyModified, this, &RgbLayerFromDatasetRep::updateRed);
	connect(m_fixedLayer, &RgbLayerFromDataset::propertyModified, this, &RgbLayerFromDatasetRep::updateGreen);
	connect(m_fixedLayer, &RgbLayerFromDataset::propertyModified, this, &RgbLayerFromDatasetRep::updateBlue);
	connect(m_fixedLayer, &RgbLayerFromDataset::propertyModified, this, &RgbLayerFromDatasetRep::updateIso);
	connect(m_fixedLayer, &RgbLayerFromDataset::newPropertyCreated, this, &RgbLayerFromDatasetRep::initProperties);

	chooseIsochrone();
}

RgbLayerFromDatasetRep::~RgbLayerFromDatasetRep() {
	delete m_currentIso;
	delete m_currentAttribute;
	if (m_propPanel!=nullptr)
		delete m_propPanel;
	if (m_layer!=nullptr)
		delete m_layer;
}

QString RgbLayerFromDatasetRep::name() const {
	return m_fixedLayer->name();
}

RgbLayerFromDataset* RgbLayerFromDatasetRep::fixedLayer() const {
	return m_fixedLayer;
}

void RgbLayerFromDatasetRep::showCrossHair(bool val) {
	m_showCrossHair = val;
}

bool RgbLayerFromDatasetRep::crossHair() const {
	return m_showCrossHair;
}

//AbstractGraphicRep
QWidget* RgbLayerFromDatasetRep::propertyPanel() {
	if (m_propPanel == nullptr) {
		m_propPanel = new RgbLayerFromDatasetPropPanel(this,
					m_parent->viewType() == ViewType::View3D, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this](){
			m_propPanel = nullptr;
		});
	}
	return m_propPanel;
}

GraphicLayer * RgbLayerFromDatasetRep::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		m_layer = new RgbLayerFromDatasetLayer(this, scene, defaultZDepth, parent);
	}
	return m_layer;
}

IData* RgbLayerFromDatasetRep::data() const {
	return m_fixedLayer;
}

//IMouseImageDataProvider
bool RgbLayerFromDatasetRep::mouseData(double x, double y,MouseInfo & info) {
	double valueIso, valueRed, valueGreen, valueBlue;
	bool valid = IGeorefImage::value(m_currentAttribute->get(0), x, y, info.i,
			info.j, valueRed);
	valid = valid && IGeorefImage::value(m_currentAttribute->get(1), x, y, info.i,
			info.j, valueGreen);
	valid = valid && IGeorefImage::value(m_currentAttribute->get(2), x, y, info.i,
			info.j, valueBlue);
	info.valuesDesc.push_back("Attribute");
	info.values.push_back(valueRed);
	info.values.push_back(valueGreen);
	info.values.push_back(valueBlue);
	info.depthValue = true;
	IGeorefImage::value(m_currentIso, x, y, info.i, info.j,
			valueIso);
//	double realDepth;
//	m_fixedLayer->dataset()->sampleTransformation()->direct(value, realDepth);
	info.depth = valueIso;
	info.depthUnit = m_fixedLayer->dataset()->cubeSeismicAddon().getSampleUnit();
	return valid;
}

void RgbLayerFromDatasetRep::dataChanged() {
	if (m_propPanel != nullptr)
		m_propPanel->updatePalette();
	if (m_layer != nullptr)
		m_layer->refresh();
}

void RgbLayerFromDatasetRep::chooseRed(QString attributeName) {
	m_currentRedName = attributeName;
	CUDAImagePaletteHolder* attribute = m_fixedLayer->image(attributeName);
	if (attribute!=nullptr) {
		attribute->lockPointer();
		m_currentAttribute->get(0)->updateTexture(attribute->backingPointer(), false);
		attribute->unlockPointer();

		dataChanged();
		updateRedPalette();

		if (m_propPanel!=nullptr) {
			m_propPanel->updateComboValueRed(m_currentRedName);
		}
	}
}

void RgbLayerFromDatasetRep::chooseGreen(QString attributeName) {
	m_currentGreenName = attributeName;
	CUDAImagePaletteHolder* attribute = m_fixedLayer->image(attributeName);
	if (attribute!=nullptr) {
		attribute->lockPointer();
		m_currentAttribute->get(1)->updateTexture(attribute->backingPointer(), false);
		attribute->unlockPointer();

		dataChanged();
		updateGreenPalette();

		if (m_propPanel!=nullptr) {
			m_propPanel->updateComboValueGreen(m_currentGreenName);
		}
	}
}

void RgbLayerFromDatasetRep::chooseBlue(QString attributeName) {
	m_currentBlueName = attributeName;
	CUDAImagePaletteHolder* attribute = m_fixedLayer->image(attributeName);
	if (attribute!=nullptr) {
		attribute->lockPointer();
		m_currentAttribute->get(2)->updateTexture(attribute->backingPointer(), false);
		attribute->unlockPointer();

		dataChanged();
		updateBluePalette();

		if (m_propPanel!=nullptr) {
			m_propPanel->updateComboValueBlue(m_currentBlueName);
		}
	}
}

void RgbLayerFromDatasetRep::updateRed(QString propName) {
	if (m_currentRedName.compare(propName)==0) {
		chooseRed(m_currentRedName);
	}
}

void RgbLayerFromDatasetRep::updateGreen(QString propName) {
	if (m_currentGreenName.compare(propName)==0) {
		chooseGreen(m_currentGreenName);
	}
}

void RgbLayerFromDatasetRep::updateBlue(QString propName) {
	if (m_currentBlueName.compare(propName)==0) {
		chooseBlue(m_currentBlueName);
	}
}

void RgbLayerFromDatasetRep::chooseIsochrone() {
	m_currentIsoName = RgbLayerFromDataset::ISOCHRONE;
	CUDAImagePaletteHolder* iso = m_fixedLayer->image(RgbLayerFromDataset::ISOCHRONE);
	if (iso!=nullptr) {
		iso->lockPointer();
		m_currentIso->updateTexture(iso->backingPointer(), false);
		iso->unlockPointer();
	}
}

void RgbLayerFromDatasetRep::updateIso(QString propName) {
	if (m_currentIsoName.compare(propName)==0) {
		chooseIsochrone();
	}
}

void RgbLayerFromDatasetRep::updateRedPalette() {
	if (m_currentRedName.compare(RgbLayerFromDataset::ISOCHRONE)==0) {
		m_currentAttribute->get(0)->setRange(QVector2D(0, std::numeric_limits<short>::max()));
		m_currentAttribute->get(0)->setLookupTable(ColorTableRegistry::PALETTE_REGISTRY().findColorTable("CLASSIC",
                "Iles-2"));
	} else {
		m_currentAttribute->get(0)->setRange(QVector2D(std::numeric_limits<short>::min(), std::numeric_limits<short>::max()));
		m_currentAttribute->get(0)->setLookupTable(ColorTableRegistry::PALETTE_REGISTRY().findColorTable("CLASSIC",
		        "Black-White"));
	}
}

void RgbLayerFromDatasetRep::updateGreenPalette() {
	if (m_currentGreenName.compare(RgbLayerFromDataset::ISOCHRONE)==0) {
		m_currentAttribute->get(1)->setRange(QVector2D(0, std::numeric_limits<short>::max()));
		m_currentAttribute->get(1)->setLookupTable(ColorTableRegistry::PALETTE_REGISTRY().findColorTable("CLASSIC",
                "Iles-2"));
	} else {
		m_currentAttribute->get(1)->setRange(QVector2D(std::numeric_limits<short>::min(), std::numeric_limits<short>::max()));
		m_currentAttribute->get(1)->setLookupTable(ColorTableRegistry::PALETTE_REGISTRY().findColorTable("CLASSIC",
		        "Black-White"));
	}
}

void RgbLayerFromDatasetRep::updateBluePalette() {
	if (m_currentBlueName.compare(RgbLayerFromDataset::ISOCHRONE)==0) {
		m_currentAttribute->get(2)->setRange(QVector2D(0, std::numeric_limits<short>::max()));
		m_currentAttribute->get(2)->setLookupTable(ColorTableRegistry::PALETTE_REGISTRY().findColorTable("CLASSIC",
                "Iles-2"));
	} else {
		m_currentAttribute->get(2)->setRange(QVector2D(std::numeric_limits<short>::min(), std::numeric_limits<short>::max()));
		m_currentAttribute->get(2)->setLookupTable(ColorTableRegistry::PALETTE_REGISTRY().findColorTable("CLASSIC",
		        "Black-White"));
	}
}

void RgbLayerFromDatasetRep::initProperties(QString propName) {
	if (m_currentRedName.isNull() || m_currentRedName.isEmpty()) {
		chooseRed(propName);
	}
	if (m_currentGreenName.isNull() || m_currentGreenName.isEmpty()) {
		chooseGreen(propName);
	}
	if (m_currentBlueName.isNull() || m_currentBlueName.isEmpty()) {
		chooseBlue(propName);
	}
	if (RgbLayerFromDataset::ISOCHRONE.compare(propName)==0) {
		chooseIsochrone();
	}
}

bool RgbLayerFromDatasetRep::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> RgbLayerFromDatasetRep::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_fixedLayer->dataset()->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString RgbLayerFromDatasetRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep RgbLayerFromDatasetRep::getTypeGraphicRep() {
    return AbstractGraphicRep::Image;
}
