#include "abstractstratisliceattribute.h"
#include "stratislice.h"
#include "seismic3dabstractdataset.h"
#include "stratislicegrahicrepfactory.h"
#include "cudaimagepaletteholder.h"
#include "cuda_volume.h"
#include "seismic3dcudadataset.h"
#include "seismic3ddataset.h"
#include "cuda_common_helpers.h"
#include "datasetbloccache.h"
#include "affine2dtransformation.h"
#include "cudargbimage.h"
#include  "cudargttile.h"
#include  "cudargbtile.h"
#include  "cudafrequencytile.h"

#include "stratislice.h"
#define EXTRACTION_INIT_WINDOW 21

//Beware that twice the WxH dataset size memory buffer is allocated. Both needs to be allocated without saturating
//the GPU... In pratice in can be low and performances are not killed... most of the time is spent on data transfers
#define TILE_SIZE 128
AbstractStratiSliceAttribute::AbstractStratiSliceAttribute(
		WorkingSetManager *workingSet, StratiSlice *slice, QObject *parent) :
		IData(workingSet, parent) {
	m_stratiSlice = slice;
	m_extractionWindow = EXTRACTION_INIT_WINDOW;
	m_isoSurfaceHolder = new CUDAImagePaletteHolder(slice->width(),
			slice->depth(), ImageFormats::QSampleType::INT16,
			slice->seismic()->ijToXYTransfo(), this);
	QVector2D minMax = slice->rgtMinMax();
	m_currentSlice = minMax[0];

	m_isInitialized = false;
}

CUDAImagePaletteHolder* AbstractStratiSliceAttribute::isoSurfaceHolder() {
	return m_isoSurfaceHolder;
}

void AbstractStratiSliceAttribute::initialize() {
	if (m_isInitialized)
		return;
	loadSlice(m_currentSlice);
	m_isInitialized = true;
}

void AbstractStratiSliceAttribute::setExtractionWindow(uint w) {
	m_extractionWindow = w;
	loadSlice(m_currentSlice);
	emit extractionWindowChanged(w);
}

uint AbstractStratiSliceAttribute::extractionWindow() const {
	return m_extractionWindow;
}

int AbstractStratiSliceAttribute::currentPosition() const {
	return m_currentSlice;
}
void AbstractStratiSliceAttribute::setSlicePosition(int pos) {
	loadSlice(pos);
	emit RGTIsoValueChanged(pos);
}

void AbstractStratiSliceAttribute::loadSlice(unsigned int z) {
	m_currentSlice = z;
}

QString AbstractStratiSliceAttribute::name() const {
	return "Attribute";
}

QUuid AbstractStratiSliceAttribute::dataID() const {
	return m_stratiSlice->dataID();
}

AbstractStratiSliceAttribute::~AbstractStratiSliceAttribute() {
}

/*
 * Attribute Display
 */
void AbstractStratiSliceAttribute::loadSlice(
		CUDAImagePaletteHolder *isoSurfaceImage, CUDAImagePaletteHolder *image,
		unsigned int extractionWindow, unsigned int z) {
	Seismic3DCUDADataset *cudaSeismic =
			dynamic_cast<Seismic3DCUDADataset*>(m_stratiSlice->seismic());
	Seismic3DCUDADataset *cudaRGT =
			dynamic_cast<Seismic3DCUDADataset*>(m_stratiSlice->rgt());
	if (cudaSeismic != nullptr && cudaRGT != nullptr) {
//		QElapsedTimer timer;
//		timer.start();
		isoSurfaceImage->lockPointer();
		image->lockPointer();
		attributeAndIsoValueBlocExtract(cudaSeismic->cudaBuffer(),
				cudaRGT->cudaBuffer(), (short*) image->cudaPointer(),
				(short*) isoSurfaceImage->cudaPointer(), m_stratiSlice->width(),
				m_stratiSlice->height(), m_stratiSlice->depth(), z,
				extractionWindow);
		isoSurfaceImage->unlockPointer();
		image->unlockPointer();
//		std::cout << "Global spent time:" << timer.elapsed() << std::endl;
		image->swapCudaPointer();
		isoSurfaceImage->swapCudaPointer();
	} else {
		loadConcurrentSlice(isoSurfaceImage, image, extractionWindow, z);
	}
}

void AbstractStratiSliceAttribute::loadConcurrentSlice(
		CUDAImagePaletteHolder *isoSurfaceImage, CUDAImagePaletteHolder *image,
		unsigned int extractionWindow, unsigned int z) {
	int d = m_stratiSlice->seismic()->depth();
	int w = m_stratiSlice->seismic()->width();
	int h = m_stratiSlice->seismic()->height();

	Seismic3DDataset *seismic =
			dynamic_cast<Seismic3DDataset*>(m_stratiSlice->seismic());
	Seismic3DDataset *rgt =
			dynamic_cast<Seismic3DDataset*>(m_stratiSlice->rgt());

	isoSurfaceImage->lockPointer();
	image->lockPointer();
	short *isoVal = (short*) isoSurfaceImage->cudaPointer();
	short *attrVal = (short*) image->cudaPointer();

	int numTile = d / TILE_SIZE + 1;
	QVector<QVector2D> tileCoords;
	for (int i = 0; i < numTile; i++) {
		int d0 = i * TILE_SIZE;
		int d1 = d0 + TILE_SIZE;
		if (d1 > d) {
			tileCoords.push_back(QVector2D(d0, d));
			break;
		} else
			tileCoords.push_back(QVector2D(d0, d1));
	}
//	QElapsedTimer timer;
//	timer.start();
	CUDARGTTile *prev = new CUDARGTTile(w, h, TILE_SIZE);
	CUDARGTTile *next = new CUDARGTTile(w, h, TILE_SIZE);
	prev->fillHostBuffer(seismic, 0, rgt, 0, tileCoords[0].x(), tileCoords[0].y()); // 0 for channelS and channelT because dimV should be 0 here
	for (int i = 0; i < tileCoords.size(); i += 2) {
		prev->run(z, extractionWindow, isoVal, attrVal);
		if (i < tileCoords.size() - 1) {
			next->fillHostBuffer(seismic, 0, rgt, 0, tileCoords[i + 1].x(),
					tileCoords[i + 1].y()); // 0 for channelS and channelT because dimV should be 0 here
		}
		prev->writeResult();
		if (i < tileCoords.size() - 1) {
			next->run(z, extractionWindow, isoVal, attrVal);
		}
		if (i < tileCoords.size() - 2) {
			prev->fillHostBuffer(seismic, 0, rgt, 0, tileCoords[i + 2].x(),
					tileCoords[i + 2].y()); // 0 for channelS and channelT because dimV should be 0 here
		}
		if (i < tileCoords.size() - 1) {
			next->writeResult();
		}
	}
	delete prev;
	delete next;

	//std::cout << "Global spent time:" << timer.elapsed() << std::endl;
	image->unlockPointer();
	isoSurfaceImage->unlockPointer();

	image->swapCudaPointer();
	isoSurfaceImage->swapCudaPointer();
}

/*
 * RGB Display
 */
void AbstractStratiSliceAttribute::loadRGBSlice(
		CUDAImagePaletteHolder *isoSurfaceImage, CUDARGBImage *image,
		unsigned int extractionWindow, unsigned int z, int f1, int f2, int f3) {
	Seismic3DCUDADataset *cudaSeismic =
			dynamic_cast<Seismic3DCUDADataset*>(m_stratiSlice->seismic());
	Seismic3DCUDADataset *cudaRGT =
			dynamic_cast<Seismic3DCUDADataset*>(m_stratiSlice->rgt());
	if (cudaSeismic != nullptr && cudaRGT != nullptr) {
		std::cout << "Load slice" << std::endl;

		isoSurfaceImage->lockPointer();
		image->lockPointer();

		CUDAImagePaletteHolder *redChannel = image->get(0);
		CUDAImagePaletteHolder *greenChannel = image->get(1);
		CUDAImagePaletteHolder *blueChannel = image->get(2);

		blocFFT(cudaSeismic->cudaBuffer(), cudaRGT->cudaBuffer(),
				(short*) isoSurfaceImage->cudaPointer(),
				(float*) redChannel->cudaPointer(),
				(float*) greenChannel->cudaPointer(),
				(float*) blueChannel->cudaPointer(), m_stratiSlice->width(),
				m_stratiSlice->height(), m_stratiSlice->depth(), z,
				extractionWindow, f1, f2, f3);

		image->unlockPointer();
		isoSurfaceImage->unlockPointer();

		image->swapCudaPointer();
		isoSurfaceImage->swapCudaPointer();
	} else
		loadConcurrentRGBSlice(isoSurfaceImage, image, extractionWindow, z, f1,
				f2, f3);
}

void AbstractStratiSliceAttribute::loadConcurrentRGBSlice(
		CUDAImagePaletteHolder *isoSurfaceImage, CUDARGBImage *image,
		unsigned int extractionWindow, unsigned int z, int f1, int f2, int f3) {
	int d = m_stratiSlice->seismic()->depth();
	int w = m_stratiSlice->seismic()->width();
	int h = m_stratiSlice->seismic()->height();

	Seismic3DDataset *seismic =
			dynamic_cast<Seismic3DDataset*>(m_stratiSlice->seismic());
	Seismic3DDataset *rgt =
			dynamic_cast<Seismic3DDataset*>(m_stratiSlice->rgt());

	size_t dims = isoSurfaceImage->width() * isoSurfaceImage->height();

	isoSurfaceImage->lockPointer();
	image->lockPointer();
	short *isoVal = (short*) isoSurfaceImage->cudaPointer();

	float *f1Val = (float*) image->get(0)->cudaPointer();
	float *f2Val = (float*) image->get(1)->cudaPointer();
	float *f3Val = (float*) image->get(2)->cudaPointer();

	int numTile = d / TILE_SIZE + 1;
	QVector<QVector2D> tileCoords;
	for (int i = 0; i < numTile; i++) {
		int d0 = i * TILE_SIZE;
		int d1 = d0 + TILE_SIZE;
		if (d1 > d) {
			tileCoords.push_back(QVector2D(d0, d));
			break;
		} else
			tileCoords.push_back(QVector2D(d0, d1));
	}
//	QElapsedTimer timer;
//	timer.start();
	CUDARGBTile *prev = new CUDARGBTile(w, h, TILE_SIZE, extractionWindow);
	CUDARGBTile *next = new CUDARGBTile(w, h, TILE_SIZE, extractionWindow);
	prev->fillHostBuffer(seismic, 0, rgt, 0, tileCoords[0].x(), tileCoords[0].y()); // 0 for channelS and channelT because dimV should be 0 here
	for (int i = 0; i < tileCoords.size(); i += 2) {
		prev->run(z, f1, f2, f3, isoVal, f1Val, f2Val, f3Val);
		if (i < tileCoords.size() - 1) {
			next->fillHostBuffer(seismic, 0, rgt, 0, tileCoords[i + 1].x(),
					tileCoords[i + 1].y()); // 0 for channelS and channelT because dimV should be 0 here
		}
		prev->writeResult();
		if (i < tileCoords.size() - 1) {
			next->run(z, f1, f2, f3, isoVal, f1Val, f2Val, f3Val);
		}
		if (i < tileCoords.size() - 2) {
			prev->fillHostBuffer(seismic, 0, rgt, 0, tileCoords[i + 2].x(),
					tileCoords[i + 2].y()); // 0 for channelS and channelT because dimV should be 0 here
		}
		if (i < tileCoords.size() - 1) {
			next->writeResult();
		}
	}
	delete prev;
	delete next;
	//checkCudaErrors(cudaDeviceSynchronize());
//	std::cout << "Global spent time:" << timer.elapsed() << std::endl;
	image->unlockPointer();
	isoSurfaceImage->unlockPointer();

	image->swapCudaPointer();
	isoSurfaceImage->swapCudaPointer();
}

//Frequency
void AbstractStratiSliceAttribute::loadFrequencySlice(
		CUDAImagePaletteHolder *isoSurfaceImage,
		QVector<CUDAImagePaletteHolder*> images, unsigned int extractionWindow,
		unsigned int z) {
	Seismic3DCUDADataset *cudaSeismic =
			dynamic_cast<Seismic3DCUDADataset*>(m_stratiSlice->seismic());
	Seismic3DCUDADataset *cudaRGT =
			dynamic_cast<Seismic3DCUDADataset*>(m_stratiSlice->rgt());
	if (cudaSeismic != nullptr && cudaRGT != nullptr) {
		std::cout << "Load slice" << std::endl;

		isoSurfaceImage->lockPointer();

		float **cudaArray;
		cudaMalloc(&cudaArray, images.size() * sizeof(float*));

		float *cudaHostArray[images.size()];
		for (int i = 0; i < images.size(); i++) {
			CUDAImagePaletteHolder *img = images[i];
			img->lockPointer();
			cudaHostArray[i] = (float*) img->cudaPointer();
		}
		cudaMemcpy(cudaArray, cudaHostArray, images.size() * sizeof(float*),
				cudaMemcpyHostToDevice);

		checkCudaErrors(cudaDeviceSynchronize());
		blocFFTAll(cudaSeismic->cudaBuffer(), cudaRGT->cudaBuffer(),
				(short*) isoSurfaceImage->cudaPointer(), cudaArray,
				m_stratiSlice->width(), m_stratiSlice->height(),
				m_stratiSlice->depth(), z, extractionWindow);
		checkCudaErrors(cudaDeviceSynchronize());
		for (CUDAImagePaletteHolder *img : images)
			img->unlockPointer();
		isoSurfaceImage->unlockPointer();

		for (CUDAImagePaletteHolder *img : images)
			img->swapCudaPointer();
		isoSurfaceImage->swapCudaPointer();

		cudaFree(cudaArray);
	} else
		loadConcurrentFrequencySlice(isoSurfaceImage, images, extractionWindow,
				z);
}

void AbstractStratiSliceAttribute::loadConcurrentFrequencySlice(
		CUDAImagePaletteHolder *isoSurfaceImage,
		QVector<CUDAImagePaletteHolder*> images, unsigned int extractionWindow,
		unsigned int z) {
	int d = m_stratiSlice->seismic()->depth();
	int w = m_stratiSlice->seismic()->width();
	int h = m_stratiSlice->seismic()->height();

	Seismic3DDataset *seismic =
			dynamic_cast<Seismic3DDataset*>(m_stratiSlice->seismic());
	Seismic3DDataset *rgt =
			dynamic_cast<Seismic3DDataset*>(m_stratiSlice->rgt());

	size_t dims = isoSurfaceImage->width() * isoSurfaceImage->height();

	isoSurfaceImage->lockPointer();

	float **cudaArray;
	cudaMalloc(&cudaArray, images.size() * sizeof(float*));

	float *cudaHostArray[images.size()];
	for (int i = 0; i < images.size(); i++) {
		CUDAImagePaletteHolder *img = images[i];
		img->lockPointer();
		cudaHostArray[i] = (float*) img->cudaPointer();
	}
	cudaMemcpy(cudaArray, cudaHostArray, images.size() * sizeof(float*),
					cudaMemcpyHostToDevice);

	short *isoVal = (short*) isoSurfaceImage->cudaPointer();

	int numTile = d / TILE_SIZE + 1;
	QVector<QVector2D> tileCoords;
	for (int i = 0; i < numTile; i++) {
		int d0 = i * TILE_SIZE;
		int d1 = d0 + TILE_SIZE;
		if (d1 > d) {
			tileCoords.push_back(QVector2D(d0, d));
			break;
		} else
			tileCoords.push_back(QVector2D(d0, d1));
	}
//	QElapsedTimer timer;
//	timer.start();
	CUDAFrequencyTile *prev = new CUDAFrequencyTile(w, h, TILE_SIZE,
			extractionWindow);
	CUDAFrequencyTile *next = new CUDAFrequencyTile(w, h, TILE_SIZE,
			extractionWindow);
	prev->fillHostBuffer(seismic, 0, rgt, 0, tileCoords[0].x(), tileCoords[0].y()); // 0 for channelS and channelT because dimV should be 0 here
	for (int i = 0; i < tileCoords.size(); i += 2) {
		prev->run(z, isoVal, cudaArray);
		if (i < tileCoords.size() - 1) {
			next->fillHostBuffer(seismic, 0, rgt, 0, tileCoords[i + 1].x(),
					tileCoords[i + 1].y()); // 0 for channelS and channelT because dimV should be 0 here
		}
		prev->writeResult();
		if (i < tileCoords.size() - 1) {
			next->run(z, isoVal, cudaArray);
		}
		if (i < tileCoords.size() - 2) {
			prev->fillHostBuffer(seismic, 0, rgt, 0, tileCoords[i + 2].x(),
					tileCoords[i + 2].y()); // 0 for channelS and channelT because dimV should be 0 here
		}
		if (i < tileCoords.size() - 1) {
			next->writeResult();
		}
	}
	delete prev;
	delete next;
	//checkCudaErrors(cudaDeviceSynchronize());
//	std::cout << "Global spent time:" << timer.elapsed() << std::endl;
	for (CUDAImagePaletteHolder *img : images)
		img->unlockPointer();
	isoSurfaceImage->unlockPointer();

	for (CUDAImagePaletteHolder *img : images)
		img->swapCudaPointer();
	isoSurfaceImage->swapCudaPointer();

	cudaFree(cudaArray);
}

