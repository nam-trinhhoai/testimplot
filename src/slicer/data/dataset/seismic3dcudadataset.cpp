#include "seismic3dcudadataset.h"
#include "seismic3ddatasetgrahicrepfactory.h"
#include <QFileInfo>
#include <QDebug>
#include <iostream>
#include "Xt.h"
#include "cudaimagepaletteholder.h"
#include "slicerep.h"
#include "slicepositioncontroler.h"
#include <cuda.h>
#include "cuda_volume.h"
#include "cuda_algo.h"
#include "cuda_common_helpers.h"
#include "GeotimeProjectManagerWidget.h" // to get file axis
#include "sampletypebinder.h"

Seismic3DCUDADataset::Seismic3DCUDADataset(SeismicSurvey *survey,const QString &name,
		WorkingSetManager *workingSet, CUBE_TYPE type, QString idPath, QObject *parent) :
		Seismic3DAbstractDataset(survey,name, workingSet, type, idPath, parent) {
	m_content = nullptr;
	m_cudaBuffer = nullptr;
	m_repFactory = new Seismic3DDatasetGraphicRepFactory(this);
}

IGraphicRepFactory* Seismic3DCUDADataset::graphicRepFactory() {
	return m_repFactory;
}

void Seismic3DCUDADataset::releaseContent() {
	if (m_content) {
		delete[] m_content;
		m_content = nullptr;
	}
}

template<typename InputType>
size_t Seismic3DCUDADataset::InitContentKernel<InputType>::run(Seismic3DCUDADataset* obj, long trueDimV, FILE* fp) {
	size_t read = 0;
	std::vector<InputType> tmp;
	long NsectionXdimV = trueDimV*obj->m_width*obj->m_height;
	long Nsection = obj->m_width*obj->m_height;
	tmp.resize(NsectionXdimV);
	for (std::size_t z=0; z<obj->m_depth; z++) {
		read += fread(tmp.data(), sizeof(InputType), tmp.size(), fp);
		#pragma omp parallel for
		for (std::size_t idx=0; idx<Nsection; idx++) {
			if (isSameType<InputType, short>::value) {
				obj->m_content[idx] = tmp[idx*trueDimV];
			} else {
				char tmpSwap;
				// swap to get real value
				InputType val = tmp[idx*trueDimV];
				if (sizeof(InputType)>1) {
					char* beg = (char*) &val;
					char* end = beg + sizeof(InputType) - 1;
					while (beg < end) {
						tmpSwap = *beg;
						*beg = *end;
						*end = tmpSwap;
						beg++;
						end--;
					}
				}
				short valS = val;
				// swap to get real value swapped
				char* beg = (char*) &valS;
				char* end = beg + 1;

				tmpSwap = *beg;
				*beg = *end;
				*end = tmpSwap;

				obj->m_content[idx] = valS;
			}
		}
	}
	return read;
}

void Seismic3DCUDADataset::loadFromXt(const std::string &path, int dimVHint) {
	std::size_t offset;
	long trueDimV;
	ImageFormats::QSampleType realSampleType;
	{
		inri::Xt xt(path.c_str());
		if (!xt.is_valid()) {
			std::cerr << "xt cube is not valid (" << path << ")" << std::endl;
			return;
		}
		m_height = xt.nSamples();
		m_width = xt.nRecords();
		m_depth = xt.nSlices();
		m_dimV = 1; // only use first channel if there is more than one
		trueDimV = xt.pixel_dimensions();
		m_sampleType = ImageFormats::QSampleType::INT16; // translateType(xt.type());
		realSampleType = translateType(xt.type());

		if (dimVHint>0 && trueDimV==1 && m_height%dimVHint==0) {
			m_height = m_height / dimVHint;
			trueDimV = dimVHint;
		}

		m_seismicAddon.set(
			xt.startSamples(), xt.stepSamples(),
			xt.startRecord(), xt.stepRecords(),
			xt.startSlice(), xt.stepSlices());
		offset = (size_t)xt.header_size();
		int timeOrDepth = GeotimeProjectManagerWidget::filext_axis(QString::fromStdString(path));
		m_seismicAddon.setSampleUnit((timeOrDepth==0) ? SampleUnit::TIME : SampleUnit::DEPTH);

		float recordStep = xt.stepRecords();
		float sampleStep = xt.stepSamples();
		float sliceStep = xt.stepSlices();
		std::cerr << recordStep << "\t" << sampleStep << "\t" << sliceStep
				<< std::endl;

		m_xtFile = path;
	}

	tryInitRangeLock(path);

	size_t size = m_width * m_height * m_depth;
	m_content = new short[size];
	memset(m_content, 0, size * sizeof(short));

	FILE *fp = fopen(path.c_str(), "rb");
	if (!fp) {
		fprintf(stderr, "Error opening file '%s'\n", path.c_str());
		return;
	}
	char ent[offset];
	fread(ent, sizeof(char), offset, fp);

	size_t read = 0;
	if (trueDimV==1 && realSampleType==ImageFormats::QSampleType::INT16) {
		read = fread(m_content, 1, size * sizeof(short), fp);
	} else {
		SampleTypeBinder binder(realSampleType);
		read = binder.bind<InitContentKernel>(this, trueDimV, fp);
	}
	fclose(fp);

	printf("Read '%s', %lu bytes\n", path.c_str(), read);

	//initialize here a default transformation
	initializeTransformation();

}

short* Seismic3DCUDADataset::cudaBuffer() {
	if (m_cudaBuffer == nullptr) {
		m_cudaBuffer = initCuda3DVolume(m_content, cudaExtent { width(),
				height(), depth() });
		releaseContent();
	}
	return m_cudaBuffer;
}

QVector2D Seismic3DCUDADataset::minMax(int channel, bool forced) {
	if (channel!=0) {// dimV = 1 in cuda dataset for now
		return QVector2D(0, 0);
	}

	if (m_internalMinMaxCache.initialized && !forced)
		return m_internalMinMaxCache.range;

	short *buffer = cudaBuffer();
	checkCudaErrors(cudaDeviceSynchronize());
	computeMinMaxOptimized(buffer, (size_t) width() * height() * depth(),
			m_internalMinMaxCache.range[0], m_internalMinMaxCache.range[1]);
	checkCudaErrors(cudaDeviceSynchronize());
	m_internalMinMaxCache.initialized = true;
	return m_internalMinMaxCache.range;
}

void Seismic3DCUDADataset::loadInlineXLine(CUDAImagePaletteHolder *image,
		SliceDirection dir, unsigned int z, unsigned int c, SpectralImageCache* cache) {

	// to assure compatibility parameter c is not taken into account because cuda code work only on gray
	std::cout << "Load slice" << std::endl;
	cudaExtent extent = cudaExtent { width(), height(), depth() };
	long imageWidth = image->width();
	long imageHeight = image->height();
	image->lockPointer();
	if (dir == SliceDirection::Inline) {
		renderInline((short*) image->cudaPointer(), (short*) cudaBuffer(),
				extent, imageWidth, imageHeight, z);
	} else if (dir == SliceDirection::XLine) {
		renderXline((short*) image->cudaPointer(), (short*) cudaBuffer(),
				extent, imageWidth, imageHeight, z);
	}
	void* cachePtr = nullptr;
	MonoBlockSpectralImageCache* monoCache = dynamic_cast<MonoBlockSpectralImageCache*>(cache);
	if (monoCache!=nullptr) {
		cachePtr = static_cast<void*>(monoCache->buffer().data());
	}
	if (cachePtr!=nullptr) {
		short* tab = (short*) cachePtr;

		cudaMemcpy(tab, image->cudaPointer(), sizeof(short) * imageWidth * imageHeight,
					cudaMemcpyDeviceToHost);
	}

	image->unlockPointer();
	image->swapCudaPointer();

	// apply range lock
	if (m_rangeLock) {
		image->setRange(m_lockedRange);
	}
}

void Seismic3DCUDADataset::loadRandomLine(CUDAImagePaletteHolder *cudaImage,
		const QPolygon& randomLine, unsigned int c, SpectralImageCache* cache) {
	// TODO
	qDebug() << "!!! WARNING !!! Seismic3DCUDADataset::loadRandomLine function not coded, buffer will not be updated";
}

bool Seismic3DCUDADataset::writeRangeToFile(const QVector2D& range) {
	return Seismic3DAbstractDataset::writeRangeToFile(range, m_xtFile);
}

Seismic3DCUDADataset::~Seismic3DCUDADataset() {
	releaseContent();
	if (m_cudaBuffer) {
		cudaFree(m_cudaBuffer);
		m_cudaBuffer = nullptr;
	}
}

