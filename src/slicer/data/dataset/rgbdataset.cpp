#include "rgbdataset.h"
#include "rgbdatasetgraphicrepfactory.h"

#include "seismic3dabstractdataset.h"
#include "sampletypebinder.h"
#include "issame.h"
#include "cudaimagepaletteholder.h"
#include "cubeseismicaddon.h"

RgbDataset::RgbDataset(const QString name, Volume* r, int channelR,
		Volume* g, int channelG,
		Volume* b, int channelB,
		Volume* a, int channelA,
		WorkingSetManager *workingSet, QObject *parent) :
			IData(workingSet, parent),
			m_sampleTransformation(*r->sampleTransformation()),
			m_ijToInlineXline(*r->ijToInlineXlineTransfo()),
			m_ijToInlineXlineForInline(*r->ijToInlineXlineTransfoForInline()),
			m_ijToInlineXlineForXline(*r->ijToInlineXlineTransfoForXline()),
			m_ijToXY(*r->ijToXYTransfo()) {
	m_red = r;
	m_channelRed = channelR;
	m_green = g;
	m_channelGreen = channelG;
	m_blue = b;
	m_channelBlue = channelB;
	m_alpha = a;
	m_channelAlpha = channelA;

	if (m_alpha==nullptr) {
		m_alphaMode = Mode::NONE;
	} else {
		m_alphaMode = Mode::OTHER;
	}

	m_uuid = QUuid::createUuid();
	m_name = name;
	m_repFactory = new RgbDatasetGraphicRepFactory(this);

	m_width = m_red->width();
	m_height = m_red->height();
	m_depth = m_red->depth();
	m_sampleUnit = m_red->cubeSeismicAddon().getSampleUnit();
}

RgbDataset::~RgbDataset() {
	delete m_repFactory;
}

QUuid RgbDataset::dataID() const {
	return m_uuid;
}

IGraphicRepFactory* RgbDataset::graphicRepFactory() {
	return m_repFactory;
}

RgbDataset* RgbDataset::createRgbDataset(const QString name, Volume* r, int channelR,
		Volume* g, int channelG,
		Volume* b, int channelB,
		Volume* a, int channelA,
		WorkingSetManager *workingSet, QObject *parent) {
	RgbDataset* out = nullptr;

	// fast way is to check that the surveys are the same and that transforms are the same
	// can be improved by checking grid on basemap (coordinate system XY)

	// use red as base dataset
	bool isGreenValid = true;
	if (r!=g) {
		isGreenValid = r->survey()==g->survey() && r->width()==g->width() && r->height()==g->height() && r->depth()==g->depth() &&
				r->sampleTransformation()->a()==g->sampleTransformation()->a() && r->sampleTransformation()->b()==g->sampleTransformation()->b() &&
				r->ijToInlineXlineTransfo()->imageToWorldTransformation()==g->ijToInlineXlineTransfo()->imageToWorldTransformation();
	}
	bool isBlueValid = true;
	if (r!=b) {
		isBlueValid = r->survey()==b->survey() && r->width()==b->width() && r->height()==b->height() && r->depth()==b->depth() &&
				r->sampleTransformation()->a()==b->sampleTransformation()->a() && r->sampleTransformation()->b()==b->sampleTransformation()->b() &&
				r->ijToInlineXlineTransfo()->imageToWorldTransformation()==b->ijToInlineXlineTransfo()->imageToWorldTransformation();
	}
	if (isBlueValid && isGreenValid) {
		out = new RgbDataset(name, r, channelR, g, channelG, b, channelB, a, channelA, workingSet, parent);
	}
	return out;
}

const AffineTransformation  * const RgbDataset::sampleTransformation() const {
	return &m_sampleTransformation;
}

const Affine2DTransformation  * const RgbDataset::ijToInlineXlineTransfo() const {
	return &m_ijToInlineXline;
}

const Affine2DTransformation  * const RgbDataset::ijToInlineXlineTransfoForInline() const {
	return &m_ijToInlineXlineForInline;
}

const Affine2DTransformation  * const RgbDataset::ijToInlineXlineTransfoForXline() const {
	return &m_ijToInlineXlineForXline;
}

const Affine2DTransformation* const RgbDataset::ijToXYTransfo() const {
	return &m_ijToXY;
}

QRectF RgbDataset::inlineXlineExtent() const {
	return m_ijToInlineXline.worldExtent();
}

template<typename DataType>
struct SetMaxAlphaKernel {
	static void run(CUDAImagePaletteHolder* image) {
		std::size_t N = image->width() * image->height();
		DataType maxVal = std::numeric_limits<DataType>::max();
		if (isSameType<DataType, float>::value || isSameType<DataType, double>::value) {
			maxVal = 1.0;
		}
		std::vector<DataType> buffer;
		buffer.resize(N, maxVal);
		image->updateTexture(buffer.data(), false);
	}
};

void RgbDataset::loadInlineXLine(CUDAImagePaletteHolder *redImage, CUDAImagePaletteHolder *greenImage,
		CUDAImagePaletteHolder *blueImage, CUDAImagePaletteHolder *alphaImage, SliceDirection dir,
		unsigned int z,  SpectralImageCache* redCache, SpectralImageCache* greenCache,
		SpectralImageCache* blueCache, SpectralImageCache* alphaCache) {
	if (m_red) {
		m_red->loadInlineXLine(redImage, dir, z, m_channelRed, redCache);
	} else {
		SampleTypeBinder binder(redImage->sampleType());
		binder.bind<SetMaxAlphaKernel>(redImage);
	}
	if (m_green && greenCache!=nullptr && redCache==greenCache) {
		greenCache->copy(greenImage, m_channelGreen);
	} else if (m_green) {
		m_green->loadInlineXLine(greenImage, dir, z, m_channelGreen, greenCache);
	} else {
		SampleTypeBinder binder(greenImage->sampleType());
		binder.bind<SetMaxAlphaKernel>(greenImage);
	}
	if (m_blue && blueCache!=nullptr && (redCache==blueCache || greenCache==blueCache)) {
		blueCache->copy(blueImage, m_channelBlue);
	} else if (m_blue) {
		m_blue->loadInlineXLine(blueImage, dir, z, m_channelBlue, blueCache);
	} else {
		SampleTypeBinder binder(blueImage->sampleType());
		binder.bind<SetMaxAlphaKernel>(blueImage);
	}
	if (alphaImage!=nullptr) {
		if (m_alpha && alphaCache!=nullptr && (redCache==alphaCache || greenCache==alphaCache || blueCache==alphaCache)) {
			alphaCache->copy(alphaImage, m_channelAlpha);
		} else if (m_alpha) {
			m_alpha->loadInlineXLine(alphaImage, dir, z, m_channelAlpha, alphaCache);
		} else {
			SampleTypeBinder binder(alphaImage->sampleType());
			binder.bind<SetMaxAlphaKernel>(alphaImage);
		}
	}
}

void RgbDataset::loadRandomLine(CUDAImagePaletteHolder *redImage, CUDAImagePaletteHolder *greenImage,
		CUDAImagePaletteHolder *blueImage, CUDAImagePaletteHolder *alphaImage,
		const QPolygon& randomLine, SpectralImageCache* redCache, SpectralImageCache* greenCache,
		SpectralImageCache* blueCache, SpectralImageCache* alphaCache) {
	if (m_red) {
		m_red->loadRandomLine(redImage, randomLine, m_channelRed, redCache);
	} else {
		SampleTypeBinder binder(redImage->sampleType());
		binder.bind<SetMaxAlphaKernel>(redImage);
	}
	if (m_green && greenCache!=nullptr && redCache==greenCache) {
		greenCache->copy(greenImage, m_channelGreen);
	} else if (m_green) {
		m_green->loadRandomLine(greenImage, randomLine, m_channelGreen, greenCache);
	} else {
		SampleTypeBinder binder(greenImage->sampleType());
		binder.bind<SetMaxAlphaKernel>(greenImage);
	}
	if (m_blue && blueCache!=nullptr && (redCache==blueCache || greenCache==blueCache)) {
		blueCache->copy(blueImage, m_channelBlue);
	} else if (m_blue) {
		m_blue->loadRandomLine(blueImage, randomLine, m_channelBlue, blueCache);
	} else {
		SampleTypeBinder binder(blueImage->sampleType());
		binder.bind<SetMaxAlphaKernel>(blueImage);
	}
	if (alphaImage!=nullptr) {
		if (m_alpha && alphaCache!=nullptr && (redCache==alphaCache || greenCache==alphaCache || blueCache==alphaCache)) {
			alphaCache->copy(alphaImage, m_channelAlpha);
		} else if (m_alpha) {
			m_alpha->loadRandomLine(alphaImage, randomLine, m_channelAlpha, alphaCache);
		} else {
			SampleTypeBinder binder(alphaImage->sampleType());
			binder.bind<SetMaxAlphaKernel>(alphaImage);
		}
	}
}

SampleUnit RgbDataset::sampleUnit() const {
	return m_sampleUnit;
}

Volume* RgbDataset::red() const {
	return m_red;
}

int RgbDataset::channelRed() const {
	return m_channelRed;
}

void RgbDataset::setChannelRed(int val) {
	if (val>=0 && val<m_red->dimV() && val!=m_channelRed) {
		m_channelRed = val;
		emit redChannelChanged();
	}
}

Volume* RgbDataset::green() const {
	return m_green;
}

int RgbDataset::channelGreen() const {
	return m_channelGreen;
}

void RgbDataset::setChannelGreen(int val) {
	if (val>=0 && val<m_green->dimV() && val!=m_channelGreen) {
		m_channelGreen = val;
		emit greenChannelChanged();
	}
}

Volume* RgbDataset::blue() const {
	return m_blue;
}

int RgbDataset::channelBlue() const {
	return m_channelBlue;
}

void RgbDataset::setChannelBlue(int val) {
	if (val>=0 && val<m_blue->dimV() && val!=m_channelBlue) {
		m_channelBlue = val;
		emit blueChannelChanged();
	}
}

Volume* RgbDataset::alpha() const {
	return m_alpha;
}

int RgbDataset::channelAlpha() const {
	return m_channelAlpha;
}

void RgbDataset::setChannelAlpha(int val) {
	if (val>=0 && val<m_alpha->dimV() && val!=m_channelAlpha) {
		m_channelAlpha = val;
		emit alphaChannelChanged();
	}
}

RgbDataset::Mode RgbDataset::alphaMode() const {
	return m_alphaMode;
}

void RgbDataset::setAlphaMode(Mode mode) {
	m_alphaMode = mode;
	emit alphaModeChanged();
}

float RgbDataset::constantAlpha() const {
	return m_constantAlpha;
}

void RgbDataset::setConstantAlpha(float alpha) {
	m_constantAlpha = alpha;
	emit constantAlphaChanged();
}

float RgbDataset::radiusAlpha() const {
	return m_radiusAlpha;
}

void RgbDataset::setRadiusAlpha(float radius) {
	m_radiusAlpha = radius;
	emit radiusAlphaChanged();
}
