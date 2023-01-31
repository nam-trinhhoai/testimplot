#include "computationoperatordataset.h"
#include "cudaimagepaletteholder.h"
#include "affinetransformation.h"
#include "affine2dtransformation.h"
#include "colortableregistry.h"
#include "sampletypebinder.h"
#include "computationoperatordatasetgraphicrepfactory.h"
#include "ivolumecomputationoperator.h"
#include "smsurvey3D.h"
#include "workingsetmanager.h"
#include "folderdata.h"
#include "seismicsurvey.h"
#include "textcolortreewidgetitemdecorator.h"

#include "gdal.h"

#include <cmath>

ComputationOperatorDataset::ComputationOperatorDataset(IVolumeComputationOperator* op, WorkingSetManager *workingSet,
		QObject *parent) : Volume(workingSet, parent) {
	m_op = op;
	m_name = m_op->name();
	// same convention as seismic datasets
	m_width = m_op->dimJ();
	m_height = m_op->dimI();
	m_depth = m_op->dimK();
	m_dimV = m_op->dimV();
	m_sampleTransformation.reset(new AffineTransformation(m_op->stepI(), m_op->originI()));
	m_uuid = QUuid::createUuid();
	m_internalMinMaxCache.initialized = false;
	m_sampleType = m_op->sampleType();

	setupTransforms();

	// did not find an example to find angles, maybe avi generation could help to find it
	m_seismicAddon = CubeSeismicAddon(m_op->originI(), m_op->stepI(),m_op->originJ(), m_op->stepJ(),
			m_op->originK(), m_op->stepK(), m_op->sampleUnit());

	m_repFactory = new ComputationOperatorDatasetGraphicRepFactory(this);

	m_decorator = nullptr;
}

ComputationOperatorDataset::~ComputationOperatorDataset() {

}

void ComputationOperatorDataset::setupInlineXlineTransfo() {
	std::array<double, 6> result;

	result[0]=m_op->originJ();
	result[1]=m_op->stepJ();
	result[2]=0;

	result[3]=m_op->originK();
	result[4]=0;
	result[5]=m_op->stepK();

	m_ijToInlineXline.reset(new Affine2DTransformation(m_op->dimJ(),m_op->dimK(),result));
}

void ComputationOperatorDataset::setupInlineXlineTransfoForInline() {
	std::array<double, 6> result;
	result[0]=m_op->originJ();
	result[1]=m_op->stepJ();
	result[2]=0;

	result[3]=m_op->originI();
	result[4]=0;
	result[5]=m_op->stepI();

	m_ijToInlineXlineForInline.reset(new Affine2DTransformation(m_op->dimJ(),m_op->dimI(),result));
}

void ComputationOperatorDataset::setupInlineXlineTransfoForXline() {
	std::array<double, 6> result;
	result[0]=m_op->originK();
	result[1]=m_op->stepK();
	result[2]=0;

	result[3]=m_op->originI();
	result[4]=0;
	result[5]=m_op->stepI();

	m_ijToInlineXlineForXline.reset(new Affine2DTransformation(m_op->dimK(),m_op->dimI(),result));
}

void ComputationOperatorDataset::setupTransforms() {
	SmSurvey3D survey(m_op->getSurveyPath().toStdString());

	setupInlineXlineTransfo();
	setupInlineXlineTransfoForInline();
	setupInlineXlineTransfoForXline();

	std::array<double, 6> inlineXlineTransfo =
			survey.inlineXlineToXYTransfo().direct();
	std::array<double, 6> ijToInlineXline = m_ijToInlineXline->direct();

	std::array<double, 6> res;
	GDALComposeGeoTransforms(ijToInlineXline.data(), inlineXlineTransfo.data(),
			res.data());
	m_ijToXY.reset(new Affine2DTransformation(m_ijToInlineXline->width(),
			m_ijToInlineXline->height(), res));
}

const AffineTransformation * const ComputationOperatorDataset::sampleTransformation() const {
	return m_sampleTransformation.get();
}

const Affine2DTransformation * const ComputationOperatorDataset::ijToXYTransfo() const {
	return m_ijToXY.get();
}

const Affine2DTransformation * const ComputationOperatorDataset::ijToInlineXlineTransfo() const {
	return m_ijToInlineXline.get();
}

const Affine2DTransformation * const ComputationOperatorDataset::ijToInlineXlineTransfoForInline() const {
	return m_ijToInlineXlineForInline.get();
}

const Affine2DTransformation * const ComputationOperatorDataset::ijToInlineXlineTransfoForXline() const {
	return m_ijToInlineXlineForXline.get();
}

void ComputationOperatorDataset::loadInlineXLine(CUDAImagePaletteHolder *cudaImage,
		SliceDirection dir, unsigned int z, unsigned int c, SpectralImageCache* cache) {
	if (c<0 || c>=m_dimV) {
		c = 0;
	}
	std::vector<char*> buffer;
	std::vector<std::vector<char>> bufferArray;
	buffer.resize(m_dimV);
	void* cachePtr = nullptr;
	ArraySpectralImageCache* arrayCache = dynamic_cast<ArraySpectralImageCache*>(cache);
	if (arrayCache!=nullptr) {
		for (std::size_t i=0; i<buffer.size(); i++) {
			buffer[i] = static_cast<char*>(static_cast<void*>(arrayCache->buffer()[i].data()));
		}
	} else {
		bufferArray.resize(m_dimV);
		for (std::size_t i=0; i<buffer.size(); i++) {
			bufferArray[i].resize(m_width*m_height*m_sampleType.byte_size());
			buffer[i] = bufferArray[i].data();
		}
	}
	if (dir==SliceDirection::Inline) {
		m_op->computeInline(z, static_cast<void**>(static_cast<void*>(buffer.data())));
		std::vector<char> tempData;
		tempData.resize(m_width*m_height*m_sampleType.byte_size());
		for (int channelIdx = 0; channelIdx<m_dimV; channelIdx++) {
			memcpy(tempData.data(), buffer[channelIdx], tempData.size());

			for (long i=0; i<m_width; i++) {
				for (long j=0; j<m_height; j++) {
					long oriIdx = (j + i * m_height)*m_sampleType.byte_size();
					long outIdx = (i + j * m_width)*m_sampleType.byte_size();
					memcpy(buffer[c]+outIdx, tempData.data()+oriIdx, m_sampleType.byte_size());
				}
			}
		}
	} else {
		m_op->computeXline(z, static_cast<void**>(static_cast<void*>(buffer.data())));
		std::vector<char> tempData;
		tempData.resize(m_depth*m_height*m_sampleType.byte_size());
		for (int channelIdx = 0; channelIdx<m_dimV; channelIdx++) {
			memcpy(tempData.data(), buffer[channelIdx], tempData.size());

			for (long i=0; i<m_depth; i++) {
				for (long j=0; j<m_height; j++) {
					long oriIdx = (j + i * m_height)*m_sampleType.byte_size();
					long outIdx = (i + j * m_depth)*m_sampleType.byte_size();
					memcpy(buffer[c]+outIdx, tempData.data()+oriIdx, m_sampleType.byte_size());
				}
			}
		}
	}
	cudaImage->updateTexture(buffer[c], false);
	if (m_rangeLock) {
		cudaImage->setRange(m_lockedRange);
	}
}

void ComputationOperatorDataset::loadRandomLine(CUDAImagePaletteHolder *cudaImage,
		const QPolygon& randomLine, unsigned int c, SpectralImageCache* cache) {
	if (c<0 || c>=m_dimV) {
		c = 0;
	}
	std::vector<char*> buffer;
	std::vector<std::vector<char>> bufferArray;
	buffer.resize(m_dimV);
	void* cachePtr = nullptr;
	ArraySpectralImageCache* arrayCache = dynamic_cast<ArraySpectralImageCache*>(cache);
	if (arrayCache!=nullptr) {
		for (std::size_t i=0; i<buffer.size(); i++) {
			buffer[i] = static_cast<char*>(static_cast<void*>(arrayCache->buffer()[i].data()));
		}
	} else {
		bufferArray.resize(m_dimV);
		for (std::size_t i=0; i<buffer.size(); i++) {
			bufferArray[i].resize(m_width*m_height*m_sampleType.byte_size());
			buffer[i] = bufferArray[i].data();
		}
	}
	m_op->computeRandom(randomLine, static_cast<void**>(static_cast<void*>(buffer.data())));
	std::vector<char> tempData;
	tempData.resize(m_width*m_height*m_sampleType.byte_size());
	for (int channelIdx = 0; channelIdx<m_dimV; channelIdx++) {
		memcpy(tempData.data(), buffer[channelIdx], tempData.size());

		for (long i=0; i<m_width; i++) {
			for (long j=0; j<m_height; j++) {
				long oriIdx = (j + i * m_height)*m_sampleType.byte_size();
				long outIdx = (i + j * m_width)*m_sampleType.byte_size();
				memcpy(buffer[c]+outIdx, tempData.data()+oriIdx, m_sampleType.byte_size());
			}
		}
	}
	cudaImage->updateTexture(buffer[c], false);
	if (m_rangeLock) {
		cudaImage->setRange(m_lockedRange);
	}
}

IGraphicRepFactory *ComputationOperatorDataset::graphicRepFactory() {
	return m_repFactory;
}

//IData
QUuid ComputationOperatorDataset::dataID() const {
	return m_uuid;
}

SeismicSurvey* ComputationOperatorDataset::survey() const {
	if (m_survey==nullptr) {
		// try to find survey in working set
		QList<IData*> data = workingSetManager()->folders().seismics->data();

		int i = 0;
		while (i<data.size() && m_survey==nullptr) {
			SeismicSurvey* survey = dynamic_cast<SeismicSurvey*>(data[i]);
			bool same = survey!=nullptr && survey->isIdPathIdentical(m_op->getSurveyPath());
			if (same) {
				m_survey = survey;
			}
			i++;
		}
	}

	return m_survey;
}

QRectF ComputationOperatorDataset::inlineXlineExtent() const {
	return m_ijToInlineXline->worldExtent();
}

LookupTable ComputationOperatorDataset::defaultLookupTable() const {
	return ColorTableRegistry::PALETTE_REGISTRY().findColorTable("CLASSIC",
			"Iles-2");
}

CubeSeismicAddon ComputationOperatorDataset::cubeSeismicAddon() const {
	return m_seismicAddon;
}

ArraySpectralImageCache* ComputationOperatorDataset::createInlineXLineCache(SliceDirection dir) const {
	ArraySpectralImageCache* out;
	if (dir==SliceDirection::Inline) {
		out = new ArraySpectralImageCache(m_width, m_height, m_dimV, m_sampleType);
	} else {
		out = new ArraySpectralImageCache(m_depth, m_height, m_dimV, m_sampleType);
	}
	return out;
}

ArraySpectralImageCache* ComputationOperatorDataset::createRandomCache(const QPolygon& poly) const {
	return new ArraySpectralImageCache(poly.size(), m_height, m_dimV, m_sampleType);
}

bool ComputationOperatorDataset::isRangeLocked() const {
	return m_rangeLock;
}

const QVector2D& ComputationOperatorDataset::lockedRange() const {
	return m_lockedRange;
}

void ComputationOperatorDataset::lockRange(const QVector2D& range) {
	if ((range!=m_lockedRange || !m_rangeLock) && range.x()<range.y()) {
		m_lockedRange = range;
		m_rangeLock = true;
		emit rangeLockChanged();
	}
}

void ComputationOperatorDataset::unlockRange() {
	if (m_rangeLock) {
		m_rangeLock = false;
		emit rangeLockChanged();
	}
}

IVolumeComputationOperator* ComputationOperatorDataset::computationOperator() {
	return m_op;
}

ITreeWidgetItemDecorator* ComputationOperatorDataset::getTreeWidgetItemDecorator() {
	if (m_decorator==nullptr) {
		m_decorator = new TextColorTreeWidgetItemDecorator(QColor(Qt::cyan), this);
	}
	return m_decorator;
}

