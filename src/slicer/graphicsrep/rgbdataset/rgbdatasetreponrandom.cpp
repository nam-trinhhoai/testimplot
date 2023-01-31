#include "rgbdatasetreponrandom.h"
#include "cudaimagepaletteholder.h"
#include "rgbdatasetproppanelonrandom.h"
#include "rgbdatasetlayeronrandom.h"
#include "rgbdataset.h"
#include "abstractinnerview.h"
#include "randomlineview.h"
#include "cudargbimage.h"
#include "seismic3dabstractdataset.h"
#include <iostream>
#include <QGraphicsScene>

RgbDatasetRepOnRandom::RgbDatasetRepOnRandom(RgbDataset *data, AbstractInnerView *parent) :
	  AbstractGraphicRep(parent), IMouseImageDataProvider() {
	m_data = data;

	m_red = nullptr;
	m_green = nullptr;
	m_blue = nullptr;
	m_alpha = nullptr;
	m_opacity = 1.0; // used if alpha==nullptr
	m_transformation = nullptr;

	m_propPanel = nullptr;
	m_layer = nullptr;

	m_name=m_data->name();
}

void RgbDatasetRepOnRandom::dataChanged() {
	if (m_propPanel != nullptr)
		m_propPanel->updatePalette();
	if (m_layer != nullptr)
		m_layer->refresh();
}

IData* RgbDatasetRepOnRandom::data() const {
	return m_data;
}

RgbDataset* RgbDatasetRepOnRandom::rgbDataset() const {
	return m_data;
}

QWidget* RgbDatasetRepOnRandom::propertyPanel() {
	if (m_propPanel == nullptr) {
		m_propPanel = new RgbDatasetPropPanelOnRandom(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer* RgbDatasetRepOnRandom::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		bool result = createImagePaletteHolder();
		if (result) {
			m_layer = new RgbDatasetLayerOnRandom(this, scene, defaultZDepth, parent);
		}
	}
	return m_layer;
}

void RgbDatasetRepOnRandom::refreshLayer() {
	if (m_layer == nullptr)
		return;

	m_layer->refresh();
}

RgbDatasetRepOnRandom::~RgbDatasetRepOnRandom() {
	if (m_layer != nullptr)
		delete m_layer;
	if (m_propPanel != nullptr)
		delete m_propPanel;
}

void RgbDatasetRepOnRandom::loadRandom() {
	if (m_red!=nullptr && m_green!=nullptr && m_blue!=nullptr) {
		m_data->loadRandomLine(m_red, m_green, m_blue, m_alpha, m_discreatePolygon,
				m_redCache.get(), m_greenCache.get(), m_blueCache.get(), m_alphaCache.get());
	}
}

bool RgbDatasetRepOnRandom::mouseData(double x, double y, MouseInfo &info) {
	double valueRed, valueGreen, valueBlue, valueAlpha;
	bool valid = IGeorefImage::value(m_red, x, y, info.i, info.j, valueRed);
	valid = valid && IGeorefImage::value(m_green, x, y, info.i, info.j, valueGreen);
	valid = valid && IGeorefImage::value(m_blue, x, y, info.i, info.j, valueBlue);
	if (m_alpha) {
		valid = valid && IGeorefImage::value(m_alpha, x, y, info.i, info.j, valueAlpha);
	} else {
		valueAlpha = 1.0;
	}
	info.valuesDesc.push_back("Red");
	info.values.push_back( valueRed);
	info.valuesDesc.push_back("Green");
	info.values.push_back( valueGreen);
	info.valuesDesc.push_back("Blue");
	info.values.push_back( valueBlue);
	info.valuesDesc.push_back("Alpha");
	info.values.push_back( valueAlpha);
	info.depthValue = true;
	info.depth=y;
	info.depthUnit = m_data->sampleUnit();
	return valid;
}

bool RgbDatasetRepOnRandom::createImagePaletteHolder() {
	bool isValid = false;
	RandomLineView* randomView = dynamic_cast<RandomLineView*>(view());
	if (randomView) {
		m_discreatePolygon = randomView->discreatePolyLine();
		if (m_discreatePolygon.size()>0) {
			const AffineTransformation* sampleTransform = m_data->sampleTransformation();
			std::array<double, 6> transform;

			transform[0]=0;
			transform[1]=1;
			transform[2]=0;

			transform[3]=sampleTransform->b();
			transform[4]=0;
			transform[5]=sampleTransform->a();

			m_transformation = new Affine2DTransformation(m_discreatePolygon.size(), m_data->height(), transform, this);
			m_red = new CUDAImagePaletteHolder(
					m_discreatePolygon.size(), m_data->height(),
					m_data->red()->sampleType(),
					m_transformation, randomView);
			m_green = new CUDAImagePaletteHolder(
					m_discreatePolygon.size(), m_data->height(),
					m_data->green()->sampleType(),
					m_transformation, randomView);
			m_blue = new CUDAImagePaletteHolder(
					m_discreatePolygon.size(), m_data->height(),
					m_data->blue()->sampleType(),
					m_transformation, randomView);
			if (m_data->alpha()!=nullptr) {
				m_alpha = new CUDAImagePaletteHolder(
						m_discreatePolygon.size(), m_data->height(),
						m_data->alpha()->sampleType(),
						m_transformation, randomView);
			}

			//m_cache.resize(m_image->width() * m_image->height() * m_data->dimV() * m_data->sampleType().byte_size());
			connect(m_red, SIGNAL(dataChanged()), this, SLOT(dataChanged()));
			connect(m_data, SIGNAL(redChannelChanged()), this, SLOT(redChannelChanged()));
			connect(m_green, SIGNAL(dataChanged()), this, SLOT(dataChanged()));
			connect(m_data, SIGNAL(greenChannelChanged()), this, SLOT(greenChannelChanged()));
			connect(m_blue, SIGNAL(dataChanged()), this, SLOT(dataChanged()));
			connect(m_data, SIGNAL(blueChannelChanged()), this, SLOT(blueChannelChanged()));
			if (m_alpha) {
				connect(m_alpha, SIGNAL(dataChanged()), this, SLOT(dataChanged()));
				connect(m_data, SIGNAL(alphaChannelChanged()), this, SLOT(alphaChannelChanged()));
			}
			createCache();
			isValid = true;
		}
	}
	if (isValid) {
		loadRandom();
	}
	return isValid;
}

bool RgbDatasetRepOnRandom::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> RgbDatasetRepOnRandom::getAvailableSampleUnits() const {
	SampleUnit sampleUnit = m_data->sampleUnit();
	QList<SampleUnit> list;
	list.push_back(sampleUnit);
	return list;
}

QString RgbDatasetRepOnRandom::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

float RgbDatasetRepOnRandom::getOpacity() const {
	return m_opacity;
}

void RgbDatasetRepOnRandom::setOpacity(float val) {
	m_opacity = val;
	emit opacityChanged();
}

void RgbDatasetRepOnRandom::createCache() {
	// red
	m_redCache.reset(m_data->red()->createRandomCache(m_discreatePolygon));

	// green
	if (m_data->red()==m_data->green()) {
		m_greenCache = m_redCache;
	} else {
		m_greenCache.reset(m_data->green()->createRandomCache(m_discreatePolygon));
	}

	// blue
	if (m_data->red()==m_data->blue()) {
		m_blueCache = m_redCache;
	} else if (m_data->green()==m_data->blue()) {
		m_blueCache = m_greenCache;
	} else {
		m_blueCache.reset(m_data->blue()->createRandomCache(m_discreatePolygon));
	}

	// alpha
	if (m_data->red()==m_data->alpha()) {
		m_alphaCache = m_redCache;
	} else if (m_data->green()==m_data->alpha()) {
		m_alphaCache = m_greenCache;
	} else if (m_data->blue()==m_data->alpha()) {
		m_alphaCache = m_blueCache;
	} else if (m_data->alpha()) {
		m_alphaCache.reset(m_data->alpha()->createRandomCache(m_discreatePolygon));
	}
}

void RgbDatasetRepOnRandom::redChannelChanged() {
	if (m_redCache!=nullptr) {
		m_redCache->copy(m_red, m_data->channelRed());
	}
}

void RgbDatasetRepOnRandom::greenChannelChanged() {
	if (m_greenCache!=nullptr) {
		m_greenCache->copy(m_green, m_data->channelGreen());
	}
}

void RgbDatasetRepOnRandom::blueChannelChanged() {
	if (m_blueCache!=nullptr) {
		m_blueCache->copy(m_blue, m_data->channelBlue());
	}
}

void RgbDatasetRepOnRandom::alphaChannelChanged() {
	if (m_alphaCache!=nullptr && m_alpha!=nullptr && m_data->alpha()!=nullptr) {
		m_alphaCache->copy(m_alpha, m_data->channelAlpha());
	}
}

AbstractGraphicRep::TypeRep RgbDatasetRepOnRandom::getTypeGraphicRep() {
    return AbstractGraphicRep::Image;
}


