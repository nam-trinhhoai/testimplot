#include "rgbcomputationondataset.h"
#include "seismic3dabstractdataset.h"
#include "cudaimagepaletteholder.h"
#include "affinetransformation.h"
#include "affine2dtransformation.h"
#include "colortableregistry.h"
#include "sampletypebinder.h"
#include "RGT_Spectrum_Memory.cuh"
#include "rgbcomputationondatasetgraphicrepfactory.h"

#include <cmath>

RgbComputationOnDataset::RgbComputationOnDataset(Seismic3DAbstractDataset* dataset, int channel, WorkingSetManager *workingSet,
		QObject *parent) : Volume(workingSet, parent) {
	m_dataset = dataset;
	m_channel = channel;
	m_name = "Spectrum " + m_dataset->name();
	m_width = m_dataset->width();
	m_height = m_dataset->height();
	m_depth = m_dataset->depth();
	m_uuid = QUuid::createUuid();
	m_internalMinMaxCache.initialized = false;
	m_ijToInlineXline.reset(new Affine2DTransformation(*m_dataset->ijToInlineXlineTransfo()));
	m_ijToInlineXlineForInline.reset(new Affine2DTransformation(*m_dataset->ijToInlineXlineTransfoForInline()));
	m_ijToInlineXlineForXline.reset(new Affine2DTransformation(*m_dataset->ijToInlineXlineTransfoForXline()));
	m_ijToXY.reset(new Affine2DTransformation(*m_dataset->ijToXYTransfo()));
	m_sampleTransformation.reset(new AffineTransformation(*m_dataset->sampleTransformation()));

	m_seismicAddon = m_dataset->cubeSeismicAddon();
	m_sampleType = ImageFormats::QSampleType::INT16;

	m_dimV = m_windowSize/2;

	m_repFactory = new RgbComputationOnDatasetGraphicRepFactory(this);
}
RgbComputationOnDataset::~RgbComputationOnDataset() {
	delete m_repFactory;
}

const AffineTransformation * const RgbComputationOnDataset::sampleTransformation() const {
	return m_sampleTransformation.get();
}

const Affine2DTransformation * const RgbComputationOnDataset::ijToXYTransfo() const {
	return m_ijToXY.get();
}

const Affine2DTransformation * const RgbComputationOnDataset::ijToInlineXlineTransfo() const {
	return m_ijToInlineXline.get();
}

const Affine2DTransformation * const RgbComputationOnDataset::ijToInlineXlineTransfoForInline() const {
	return m_ijToInlineXlineForInline.get();
}

const Affine2DTransformation * const RgbComputationOnDataset::ijToInlineXlineTransfoForXline() const {
	return m_ijToInlineXlineForXline.get();
}

template<typename ImageType>
struct ComputeSpectrumOnImageKernel {
	static void run(CUDAImagePaletteHolder* image, short** buffer, int windowSize, int hatPower) {
		long width = image->width();// width is number of trace
		long height = image->height();// height is number of sample

		// in this case the map is a section, apply on transposed section because input and output are transposed
		long blocSizeS = RGTMemorySpectrum_CPUGPUMemory_getBlocSize( 0, (std::size_t) width, (std::size_t) height, windowSize);

		cufftReal *hostInputData = new cufftReal[ width * blocSizeS * windowSize];

		// is windowSize is odd windowSizeLeftSide==windowSizeRightSide else windowSizeRightSide = windowSizeLeftSide - 1
		long windowSizeLeftSide = windowSize/2;
		long windowSizeRightSide = windowSize/2 - 1 + (windowSize%2);

		// set to zero borders
		for (std::size_t fIndex=0; fIndex<windowSize/2; fIndex++) {
			memset(buffer[fIndex], 0, sizeof(short) * width * windowSizeLeftSide);
			memset(buffer[fIndex] + width * (height - windowSizeRightSide-2), 0, sizeof(short) * width * (2+windowSizeRightSide));
		}

		ImageType* rawData = static_cast<ImageType*>(image->backingPointer());
		image->lockPointer();

		double A = 0.54, B = 0.46;
		double omega = 2.0 * M_PI / (windowSize - 1);

		// because : windowSize - 1 == windowSizeRightSide + windowSizeLeftSide
		for (long indexSMin = windowSizeLeftSide; indexSMin<height - windowSizeRightSide; indexSMin+=blocSizeS) {
			long indexSMax = std::min(height - windowSizeRightSide, indexSMin + blocSizeS);
			#pragma omp parallel for collapse(3)
			for (long indexS=indexSMin; indexS<indexSMax; indexS++) {
				for (long indexX=0; indexX<width; indexX++) {
					for (long i=0; i<windowSize; i++) {
						double wt = pow(1 - std::fabs(i-windowSize/2)/(windowSize/2), hatPower) ;
						double hammingCoef = A -B * std::cos(omega * i);
						hostInputData[((indexS-indexSMin)*width + indexX)* windowSize + i] =
								(cufftReal)( hammingCoef * wt * rawData[indexX + (indexS + i - windowSizeLeftSide)*width]);
					}
				}
			}
			std::vector<short*> pseudoModule;
			pseudoModule.resize(windowSize/2+2);
			for (std::size_t attrIndex=0; attrIndex<pseudoModule.size(); attrIndex++) {
				if (attrIndex>1) {
					pseudoModule[attrIndex] = buffer[attrIndex-2] + indexSMin*width;
				} else {
					pseudoModule[attrIndex] = nullptr;
				}
			}

			int result =  RGTMemorySpectrumBis ( hostInputData, pseudoModule.data(),
								(std::size_t) 0, (std::size_t) width, (std::size_t) indexSMax - indexSMin,
								windowSize);
		}
		image->unlockPointer();
	}
};

void RgbComputationOnDataset::computeSpectrumOnInlineXLine(SliceDirection dir,
		unsigned int z, short** buffer) {
	std::unique_ptr<CUDAImagePaletteHolder> holder;
	if (dir==SliceDirection::Inline) {
		holder.reset(new CUDAImagePaletteHolder(
				width(), height(),
				m_dataset->sampleType(),
				ijToInlineXlineTransfoForInline()));
	} else {
		holder.reset(new CUDAImagePaletteHolder(
				depth(), height(),
				m_dataset->sampleType(),
				ijToInlineXlineTransfoForXline()));
	}
	// will no longer work if function exit before end of execution.
	m_dataset->loadInlineXLine(holder.get(), dir, z, m_channel);

	SampleTypeBinder binder(holder->sampleType());
	binder.bind<ComputeSpectrumOnImageKernel>(holder.get(), buffer, m_windowSize, m_hatPower);
}

void RgbComputationOnDataset::computeSpectrumOnRandom(const QPolygon& randomLine, short** buffer) {
	std::unique_ptr<CUDAImagePaletteHolder> holder;
	holder.reset(new CUDAImagePaletteHolder(
			randomLine.size(), height(),
			m_dataset->sampleType()));

	// will no longer work if function exit before end of execution.
	m_dataset->loadRandomLine(holder.get(), randomLine, m_channel);

	SampleTypeBinder binder(holder->sampleType());
	binder.bind<ComputeSpectrumOnImageKernel>(holder.get(), buffer, m_windowSize, m_hatPower);
}

void RgbComputationOnDataset::loadInlineXLine(CUDAImagePaletteHolder *cudaImage,
		SliceDirection dir, unsigned int z, unsigned int c, SpectralImageCache* cache) {
	if (c<0 || c>=m_dimV) {
		c = 0;
	}
	std::vector<short*> buffer;
	std::vector<std::vector<short>> bufferArray;
	buffer.resize(m_dimV);
	void* cachePtr = nullptr;
	ArraySpectralImageCache* arrayCache = dynamic_cast<ArraySpectralImageCache*>(cache);
	if (arrayCache!=nullptr) {
		for (std::size_t i=0; i<buffer.size(); i++) {
			buffer[i] = static_cast<short*>(static_cast<void*>(arrayCache->buffer()[i].data()));
		}
	} else {
		bufferArray.resize(m_dimV);
		for (std::size_t i=0; i<buffer.size(); i++) {
			bufferArray[i].resize(m_width*m_height);
			buffer[i] = bufferArray[i].data();
		}
	}
	computeSpectrumOnInlineXLine(dir, z, buffer.data());
	cudaImage->updateTexture(buffer[c], false);
}

void RgbComputationOnDataset::loadRandomLine(CUDAImagePaletteHolder *cudaImage,
		const QPolygon& randomLine, unsigned int c, SpectralImageCache* cache) {
	if (c<0 || c>=m_dimV) {
		c = 0;
	}
	std::vector<short*> buffer;
	std::vector<std::vector<short>> bufferArray;
	buffer.resize(m_dimV);
	void* cachePtr = nullptr;
	ArraySpectralImageCache* arrayCache = dynamic_cast<ArraySpectralImageCache*>(cache);
	if (arrayCache!=nullptr) {
		for (std::size_t i=0; i<buffer.size(); i++) {
			buffer[i] = static_cast<short*>(static_cast<void*>(arrayCache->buffer()[i].data()));
		}
	} else {
		bufferArray.resize(m_dimV);
		for (std::size_t i=0; i<buffer.size(); i++) {
			bufferArray[i].resize(m_width*m_height);
			buffer[i] = bufferArray[i].data();
		}
	}
	computeSpectrumOnRandom(randomLine, buffer.data());
	cudaImage->updateTexture(buffer[c], false);
}

IGraphicRepFactory *RgbComputationOnDataset::graphicRepFactory() {
	return m_repFactory;
}

//IData
QUuid RgbComputationOnDataset::dataID() const {
	return m_uuid;
}

SeismicSurvey* RgbComputationOnDataset::survey() const {
	return m_dataset->survey();
}

QRectF RgbComputationOnDataset::inlineXlineExtent() const {
	return m_ijToInlineXline->worldExtent();
}

LookupTable RgbComputationOnDataset::defaultLookupTable() const {
	return ColorTableRegistry::DEFAULT();
}

CubeSeismicAddon RgbComputationOnDataset::cubeSeismicAddon() const {
	return m_seismicAddon;
}

ArraySpectralImageCache* RgbComputationOnDataset::createInlineXLineCache(SliceDirection dir) const {
	ArraySpectralImageCache* out;
	if (dir==SliceDirection::Inline) {
		out = new ArraySpectralImageCache(m_width, m_height, m_dimV, m_sampleType);
	} else {
		out = new ArraySpectralImageCache(m_depth, m_height, m_dimV, m_sampleType);
	}
	return out;
}

ArraySpectralImageCache* RgbComputationOnDataset::createRandomCache(const QPolygon& poly) const {
	return new ArraySpectralImageCache(poly.size(), m_height, m_dimV, m_sampleType);
}

double RgbComputationOnDataset::getFrequency(long fIdx) const {
	return getFrequencyStatic(fIdx, m_sampleTransformation->a(), m_windowSize);
}

double RgbComputationOnDataset::getFrequencyStatic(long freqIndex, double pasech, long windowSize)  {
	return 1000/(pasech*(windowSize-1)) * (freqIndex + 0.5f);
}

