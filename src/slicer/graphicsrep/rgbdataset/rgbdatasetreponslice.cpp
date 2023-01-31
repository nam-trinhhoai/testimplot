#include "rgbdatasetreponslice.h"
#include "cudaimagepaletteholder.h"
#include "rgbdatasetproppanelonslice.h"
#include "rgbdatasetlayeronslice.h"
#include "rgbdataset.h"
#include "seismic3dabstractdataset.h"
#include "abstractinnerview.h"
#include "cudargbimage.h"
#include <iostream>
#include <QGraphicsScene>

RgbDatasetRepOnSlice::RgbDatasetRepOnSlice(RgbDataset *data, CUDAImagePaletteHolder* red,
		CUDAImagePaletteHolder* green, CUDAImagePaletteHolder* blue,
		CUDAImagePaletteHolder *alpha, const QPair<QVector2D,AffineTransformation> & sliceRangeAndTransfo,
		SliceDirection dir, AbstractInnerView *parent) :
	  AbstractGraphicRep(parent), IMouseImageDataProvider() {
	m_data = data;
	m_sliceRangeAndTransfo = sliceRangeAndTransfo;
	m_currentSlice = 0;
	m_isCurrentSliceLoaded = false;
	m_dir = dir;

	m_red = red;
	m_green = green;
	m_blue = blue;
	m_alpha = alpha;
	m_opacity = 1.0; // used if alpha==nullptr

	m_propPanel = nullptr;
	m_layer = nullptr;

	m_name=m_data->name();

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
}

void RgbDatasetRepOnSlice::createCache() {
	// red
	m_redCache.reset(m_data->red()->createInlineXLineCache(m_dir));

	// green
	if (m_data->red()==m_data->green()) {
		m_greenCache = m_redCache;
	} else {
		m_greenCache.reset(m_data->green()->createInlineXLineCache(m_dir));
	}

	// blue
	if (m_data->red()==m_data->blue()) {
		m_blueCache = m_redCache;
	} else if (m_data->green()==m_data->blue()) {
		m_blueCache = m_greenCache;
	} else {
		m_blueCache.reset(m_data->blue()->createInlineXLineCache(m_dir));
	}

	// alpha
	if (m_data->red()==m_data->alpha()) {
		m_alphaCache = m_redCache;
	} else if (m_data->green()==m_data->alpha()) {
		m_alphaCache = m_greenCache;
	} else if (m_data->blue()==m_data->alpha()) {
		m_alphaCache = m_blueCache;
	} else if (m_data->alpha()) {
		m_alphaCache.reset(m_data->alpha()->createInlineXLineCache(m_dir));
	}
}

void RgbDatasetRepOnSlice::dataChanged() {
	if (m_propPanel != nullptr)
		m_propPanel->updatePalette();
	if (m_layer != nullptr)
		m_layer->refresh();
}

IData* RgbDatasetRepOnSlice::data() const {
	return m_data;
}

RgbDataset* RgbDatasetRepOnSlice::rgbDataset() const {
	return m_data;
}

QPair<QVector2D,AffineTransformation>  RgbDatasetRepOnSlice::sliceRangeAndTransfo() const {
	return m_sliceRangeAndTransfo;
}

QWidget* RgbDatasetRepOnSlice::propertyPanel() {
	if (m_propPanel == nullptr) {
		m_propPanel = new RgbDatasetPropPanelOnSlice(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer* RgbDatasetRepOnSlice::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		m_layer = new RgbDatasetLayerOnSlice(this, scene, defaultZDepth, parent);
	}
	return m_layer;
}

void RgbDatasetRepOnSlice::refreshLayer() {
	if (m_layer == nullptr)
		return;

	m_layer->refresh();
}

RgbDatasetRepOnSlice::~RgbDatasetRepOnSlice() {
	if (m_layer != nullptr)
		delete m_layer;
	if (m_propPanel != nullptr)
		delete m_propPanel;
}

int RgbDatasetRepOnSlice::currentSliceWorldPosition() const {
	double val;
	m_sliceRangeAndTransfo.second.direct((double)m_currentSlice,val);
	return (int)val;
}

int RgbDatasetRepOnSlice::currentSliceIJPosition() const {
	return m_currentSlice;
}

void RgbDatasetRepOnSlice::setSliceWorldPosition(int val,bool force) {
	double imagePositionD;
	m_sliceRangeAndTransfo.second.indirect((double)val, imagePositionD);


	// avoid to go outside the bounds

	if (imagePositionD<0) {
		imagePositionD = 0;
		double _val;
		m_sliceRangeAndTransfo.second.direct(imagePositionD, _val);
		val = _val;
	} else if (m_dir==SliceDirection::Inline && imagePositionD>=m_data->depth()) {
		imagePositionD = m_data->depth() - 1;
		double _val;
		m_sliceRangeAndTransfo.second.direct(imagePositionD, _val);
		val = _val;
	} else if (m_dir==SliceDirection::XLine && imagePositionD>=m_data->width()) {
		imagePositionD = m_data->width() - 1;
		double _val;
		m_sliceRangeAndTransfo.second.direct(imagePositionD, _val);
		val = _val;
	}

	int pos=(int)imagePositionD;
	if(m_currentSlice==pos && !force)
		return;

	loadSlice(pos);
	refreshLayer();

	emit sliceWordPositionChanged(val);
	emit sliceIJPositionChanged(pos);
}

void RgbDatasetRepOnSlice::setSliceIJPosition(int val,bool force) {
	if(m_currentSlice==val && !force && m_isCurrentSliceLoaded)
		return;
	loadSlice(val);
	refreshLayer();

	double imagePositionD;
	m_sliceRangeAndTransfo.second.direct((double)val, imagePositionD);
	int pos=(int)imagePositionD;

	emit sliceIJPositionChanged(val);
	emit sliceWordPositionChanged(pos);
}

void RgbDatasetRepOnSlice::setSliceIJPosition(int val) {
	setSliceIJPosition(val, false);
}

void RgbDatasetRepOnSlice::loadSlice(unsigned int z) {
	m_currentSlice = z;
	m_data->loadInlineXLine(m_red, m_green, m_blue, m_alpha, m_dir, z, m_redCache.get(),
			m_greenCache.get(), m_blueCache.get(), m_alphaCache.get());
	m_isCurrentSliceLoaded = true;
}

bool RgbDatasetRepOnSlice::mouseData(double x, double y, MouseInfo &info) {
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

bool RgbDatasetRepOnSlice::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> RgbDatasetRepOnSlice::getAvailableSampleUnits() const {
	SampleUnit sampleUnit = m_data->sampleUnit();
	QList<SampleUnit> list;
	list.push_back(sampleUnit);
	return list;
}

QString RgbDatasetRepOnSlice::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

float RgbDatasetRepOnSlice::getOpacity() const {
	return m_opacity;
}

void RgbDatasetRepOnSlice::setOpacity(float val) {
	m_opacity = val;
	emit opacityChanged();
}

void RgbDatasetRepOnSlice::redChannelChanged() {
	if (m_redCache!=nullptr) {
		m_redCache->copy(m_red, m_data->channelRed());
	}
}

void RgbDatasetRepOnSlice::greenChannelChanged() {
	if (m_greenCache!=nullptr) {
		m_greenCache->copy(m_green, m_data->channelGreen());
	}
}

void RgbDatasetRepOnSlice::blueChannelChanged() {
	if (m_blueCache!=nullptr) {
		m_blueCache->copy(m_blue, m_data->channelBlue());
	}
}

void RgbDatasetRepOnSlice::alphaChannelChanged() {
	if (m_alphaCache!=nullptr && m_alpha!=nullptr && m_data->alpha()!=nullptr) {
		m_alphaCache->copy(m_alpha, m_data->channelAlpha());
	}
}

AbstractGraphicRep::TypeRep RgbDatasetRepOnSlice::getTypeGraphicRep() {
    return AbstractGraphicRep::Image;
}
