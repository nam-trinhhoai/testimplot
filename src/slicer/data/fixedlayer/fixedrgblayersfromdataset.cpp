#include "fixedrgblayersfromdataset.h"

#include "fixedrgblayersfromdatasetgraphicrepfactory.h"
#include "igraphicrepfactory.h"
#include "seismic3dabstractdataset.h"
#include "ijkhorizon.h"
#include "affinetransformation.h"
#include "gdalloader.h"
#include "workingsetmanager.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "affine2dtransformation.h"
#include "folderdata.h"

#include <gdal_priv.h>
#include <QFileInfo>
#include <QDir>
#include <QDebug>

template<typename T>
void _copyGDALBufToFloatBufPlanar(T* oriBuf, float* outBuf, std::size_t width, std::size_t height,
		std::size_t numBands, std::size_t offset, ImageFormats::QColorFormat colorFormat,
		GDALRasterBand* hBand) {
	std::size_t N = width*height;
	if (colorFormat==ImageFormats::QColorFormat::GRAY) {
		for (std::size_t i=0; i<N; i++) {
			for (std::size_t c=0; c<3; c++) {
				outBuf[N*c + i] = oriBuf[i];
			}
		}
	} else if (colorFormat==ImageFormats::QColorFormat::RGBA_INDEXED) {
		GDALColorTable * colorTable = hBand->GetColorTable();
		long colorTabelSize = colorTable->GetColorEntryCount();
		for (std::size_t i=0; i<N; i++) {
			long idx = oriBuf[i]; // apply index
			if (idx>=0 && idx<colorTabelSize) {
				const GDALColorEntry* entry = colorTable->GetColorEntry(idx);
				float r,g,b;
				if (colorTable->GetPaletteInterpretation()==GDALPaletteInterp::GPI_Gray) {
					r = entry->c1;
					g = r;
					b = r;
				} else if (colorTable->GetPaletteInterpretation()==GDALPaletteInterp::GPI_RGB) {
					r = entry->c1;
					g = entry->c2;
					b = entry->c3;
				} else if (colorTable->GetPaletteInterpretation()==GDALPaletteInterp::GPI_CMYK) {
					r = entry->c1;
					g = entry->c2;
					b = entry->c3;
					qDebug() << "Unexpected color encoding : CMYK";
				} else if (colorTable->GetPaletteInterpretation()==GDALPaletteInterp::GPI_HLS) {
					r = entry->c1;
					g = entry->c2;
					b = entry->c3;
					qDebug() << "Unexpected color encoding : HLS";
				}
				outBuf[i] = r;
				outBuf[N*1 + i] = g;
				outBuf[N*2 + i] = b;
			}
		}
	} else if (colorFormat==ImageFormats::QColorFormat::RGB_INTERLEAVED) {
		for (std::size_t i=0; i<N; i++) {
			for (std::size_t c=0; c<3; c++) {
				outBuf[N*c + i] = oriBuf[i*3+c];
			}
		}
	} else if (colorFormat==ImageFormats::QColorFormat::RGBA_INTERLEAVED) {
		for (std::size_t i=0; i<N; i++) {
			for (std::size_t c=0; c<3; c++) {
				outBuf[N*c + i] = oriBuf[i*4+c];
			}
		}
	} else if (colorFormat==ImageFormats::QColorFormat::RGB_PLANAR) {
		for (std::size_t i=0; i<N*3; i++) {
			outBuf[i] = oriBuf[i];
		}
	} else if (colorFormat==ImageFormats::QColorFormat::RGBA_PLANAR) {
		for (std::size_t i=0; i<N*3; i++) {
			outBuf[i] = oriBuf[i];
		}
	}
}

void copyGDALBufToFloatBufPlanar(void* oriBuf, float* outBuf, std::size_t width, std::size_t height,
		std::size_t numBands, std::size_t offset, ImageFormats::QColorFormat colorFormat,
		ImageFormats::QSampleType sampleType, GDALRasterBand* hBand) {
	if (sampleType==ImageFormats::QSampleType::UINT8) {
		_copyGDALBufToFloatBufPlanar<unsigned char>(static_cast<unsigned char*>(oriBuf), outBuf, width, height, numBands, offset, colorFormat, hBand);
	} else if (sampleType==ImageFormats::QSampleType::UINT16) {
		_copyGDALBufToFloatBufPlanar<unsigned short>(static_cast<unsigned short*>(oriBuf), outBuf, width, height, numBands, offset, colorFormat, hBand);
	} else if (sampleType==ImageFormats::QSampleType::INT16) {
		_copyGDALBufToFloatBufPlanar<short>(static_cast<short*>(oriBuf), outBuf, width, height, numBands, offset, colorFormat, hBand);
	} else if (sampleType==ImageFormats::QSampleType::UINT32) {
		_copyGDALBufToFloatBufPlanar<unsigned int>(static_cast<unsigned int*>(oriBuf), outBuf, width, height, numBands, offset, colorFormat, hBand);
	} else if (sampleType==ImageFormats::QSampleType::INT32) {
		_copyGDALBufToFloatBufPlanar<int>(static_cast<int*>(oriBuf), outBuf, width, height, numBands, offset, colorFormat, hBand);
	} else if (sampleType==ImageFormats::QSampleType::FLOAT32) {
		_copyGDALBufToFloatBufPlanar<float>(static_cast<float*>(oriBuf), outBuf, width, height, numBands, offset, colorFormat, hBand);
	}
}

// oriBuf and outBuf must be fo of same tyme and size
void swapWidthHeight(void* _oriBuf, void* _outBuf, std::size_t oriWidth, std::size_t oriHeight, std::size_t typeSize) {
	char* oriBuf = static_cast<char*>(_oriBuf);
	char* outBuf = static_cast<char*>(_outBuf);
	for (std::size_t i=0; i<oriWidth; i++) {
		for (std::size_t j=0; j<oriHeight; j++) {
			std::size_t indexOri = (i + j*oriWidth) * typeSize;
			std::size_t indexOut = (j + i*oriHeight) * typeSize;
			for (std::size_t k=0; k<typeSize; k++) {
				outBuf[indexOut+k] = oriBuf[indexOri+k];
			}
		}
	}
}

template<typename T1, typename T2>
bool comparePairWithSecond(const std::pair<T1, T2>& a, const std::pair<T1, T2>& b) {
	return a.second < b.second;
}

// cudaBuffer need to be a float RGBD planar stack
FixedRGBLayersFromDataset::FixedRGBLayersFromDataset(QList<std::pair<QString, QString>> layers,
			QString name, WorkingSetManager *workingSet, Seismic3DAbstractDataset* dataset,
			bool takeOwnership, QObject *parent) : IData(workingSet, parent) {
	m_dataset = dataset;
	m_name = name;

	m_repFactory.reset(new FixedRGBLayersFromDatasetGraphicRepFactory(this));

	// size
	std::size_t Nlayers = layers.size();
	std::size_t layerSize = 4 * dataset->width() * dataset->depth();
	std::size_t totalSize = Nlayers * layerSize * sizeof(float);

//	std::vector<float> cpuBuffer;
//	cpuBuffer.resize(layeCrSize);
//	float* cudaBuffer;
//	checkCudaErrors(cudaMalloc(&cudaBuffer, totalSize));

	std::size_t sizeLayerFloat = dataset->width() * dataset->depth();
	std::vector<float> swapBuf;
	swapBuf.resize(sizeLayerFloat*3);

	for (std::size_t i=0; i<layers.size(); i++) {
		// read iso and rgb
		FILE* file = fopen(layers[i].first.toStdString().c_str(), "r");
		GDALDataset* poDataset;
		bool isValid = file != nullptr;
		if (isValid) {
			m_buffers.append(std::vector<float>());
			m_buffers.last().resize(layerSize);
			QString name = QFileInfo(layers[i].second).baseName();
			m_layers.push_back(std::pair<QString, float*>(name, m_buffers.last().data()));

			fread(m_buffers.last().data()+3*sizeLayerFloat, sizeof(float), dataset->width() * dataset->depth(), file);
			fclose(file);

			poDataset = (GDALDataset*) GDALOpen(layers[i].second.toStdString().c_str(),
							GA_ReadOnly);
			isValid = poDataset != nullptr;
		}
		std::size_t width, height;
		if (isValid) {
			width = poDataset->GetRasterXSize();
			height = poDataset->GetRasterYSize();

			isValid = width==dataset->depth() && height==dataset->width();
		}
		if (isValid) {
			GDALRasterBand* hBand = (GDALRasterBand*) GDALGetRasterBand(poDataset, 1);
			ImageFormats::QColorFormat colorFormat = GDALLoader::getColorFormatType(poDataset);
			ImageFormats::QSampleType sampleType = GDALLoader::getSampleType(hBand);
			GDALDataType type = GDALGetRasterDataType(hBand);
			int offset = GDALGetDataTypeSizeBytes(type);
			int numBands = poDataset->GetRasterCount();

			std::vector<char> tmpBuf;
			tmpBuf.resize(width*height*numBands*offset);

			isValid = poDataset->RasterIO(GF_Read, 0, 0, width, height, tmpBuf.data(), width, height,
							type, numBands, nullptr, numBands * offset,
							numBands * offset * width, offset)==CPLErr::CE_None;

			GDALClose(poDataset);

			if (isValid) {
				copyGDALBufToFloatBufPlanar(tmpBuf.data(), swapBuf.data(), width, height, numBands, offset, colorFormat, sampleType, hBand);

				swapWidthHeight(swapBuf.data(), m_buffers.last().data(), width, height, sizeof(float));
				swapWidthHeight(swapBuf.data()+sizeLayerFloat, m_buffers.last().data()+sizeLayerFloat, width, height, sizeof(float));
				swapWidthHeight(swapBuf.data()+sizeLayerFloat*2, m_buffers.last().data()+sizeLayerFloat*2, width, height, sizeof(float));
			}
		} else if (poDataset!=nullptr) {
			GDALClose(poDataset);
		}

		// copy to gpu
		if (isValid) {
//			checkCudaErrors(cudaMemcpy(cudaBuffer + layerSize*i, cpuBuffers.data(), layerSize * sizeof(float), cudaMemcpyHostToDevice));
		} else {
			m_buffers.removeLast();
			m_layers.removeLast();
		}
	}

	// create indexes + get iso mean for all layers
	QList<std::pair<long, double>> listIdxAndMean;
	for (long i=0; i<m_layers.size(); i++) {
		double mean = 0;
		std::size_t w = dataset->width();
		std::size_t d = dataset->depth();
		std::size_t N = w * d;
		float* isoTab = m_layers[i].second + 3*N;
		for (std::size_t i=0; i<w; i++) {
			double tmpMean = 0;
			for (std::size_t j=0; j<d; j++) {
				tmpMean += isoTab[i*d+j];
			}
			tmpMean /= d;
			mean += tmpMean;
		}
		mean /= w;

		listIdxAndMean.push_back(std::pair<long, double>(i, mean));
	}

	// reorder indexes
	std::sort(listIdxAndMean.begin(), listIdxAndMean.end(), comparePairWithSecond<long, double>);
	for (std::pair<long, double> e : listIdxAndMean) {
		m_layersKeys.push_back(e.first);
	}

	// select all
	m_selectedLayersKeys = m_layersKeys;

	m_currentIso.reset(new CUDAImagePaletteHolder(m_dataset->width(), m_dataset->depth(),
			ImageFormats::QSampleType::FLOAT32, m_dataset->ijToXYTransfo(),
			this));
	m_currentRGB.reset(new CUDARGBImage(m_dataset->width(), m_dataset->depth(),
			ImageFormats::QSampleType::FLOAT32, m_dataset->ijToXYTransfo(),
			this));

	setCurrentImageIndex(0);
}

FixedRGBLayersFromDataset::~FixedRGBLayersFromDataset() {
//	if (m_isOwnerOfCudaBuffer) {
//		cudaFree(m_cudaBuffer);
//	}
}

unsigned int FixedRGBLayersFromDataset::width() const {
	return m_dataset->width();
}

unsigned int FixedRGBLayersFromDataset::depth() const {
	return m_dataset->depth();
}

unsigned int FixedRGBLayersFromDataset::getNbProfiles() const {
	return depth();
}

unsigned int FixedRGBLayersFromDataset::getNbTraces() const {
	return width();
}

float FixedRGBLayersFromDataset::getStepSample() {
	return m_dataset->sampleTransformation()->a();
}

float FixedRGBLayersFromDataset::getOriginSample() {
	return m_dataset->sampleTransformation()->b();
}

//IData
IGraphicRepFactory* FixedRGBLayersFromDataset::graphicRepFactory() {
	return m_repFactory.get();
}

QUuid FixedRGBLayersFromDataset::dataID() const {
	return m_dataset->dataID();
}

QString FixedRGBLayersFromDataset::name() const {
	return m_name;
}

// buffer access
//std::size_t layerSize();
const QList<std::vector<float>>& FixedRGBLayersFromDataset::buffers() const {
	return m_buffers;
}

const QList<std::pair<QString, float*>>& FixedRGBLayersFromDataset::layers() const {
	return m_layers;
}

const QList<long>& FixedRGBLayersFromDataset::selectedLayersKeys() const {
	return m_selectedLayersKeys;
}

void FixedRGBLayersFromDataset::setSelectedLayersKeys(const QList<long>& newSelection) {
	bool selectionChanged = newSelection.size()!=m_selectedLayersKeys.size();
	if (!selectionChanged) {
		std::size_t idx=0;
		while(!selectionChanged && idx<newSelection.size()) {
			selectionChanged = newSelection[idx]!=m_selectedLayersKeys[idx];
		}
	}
	if (selectionChanged) {
		// make index invalid to avoid conflicts with size
		if (m_currentImageIndex>=newSelection.size()) {
			setCurrentImageIndex(-1);
		}
		m_selectedLayersKeys = newSelection;
		emit layerSelectionChanged(m_selectedLayersKeys);
		if (m_currentImageIndex!=-1) { // only update if index valid
			setCurrentImageIndex(m_currentImageIndex); // update image
		}
	}
}

const QList<long>& FixedRGBLayersFromDataset::layersKeys() const {
	return m_layersKeys;
}

void FixedRGBLayersFromDataset::setLayersKeys(const QList<long>& newSelection) {
	bool selectionChanged = newSelection.size()!=m_layersKeys.size();
	if (!selectionChanged) {
		std::size_t idx=0;
		while(!selectionChanged && idx<newSelection.size()) {
			selectionChanged = newSelection[idx]!=m_layersKeys[idx];
		}
	}
	if (selectionChanged) {
		m_layersKeys = newSelection;
		emit layerOrderChanged(m_layersKeys);
	}
}

long FixedRGBLayersFromDataset::currentImageIndex() const {
	return m_currentImageIndex;
}

void FixedRGBLayersFromDataset::setCurrentImageIndex(long newIndex) {
	if (newIndex<0 || newIndex>=m_selectedLayersKeys.size()) {
		m_currentImageIndex = -1;
	} else {
		m_currentImageIndex = newIndex;
	}

	if (m_currentImageIndex!=-1) {
		std::size_t layerSize = m_dataset->width() * m_dataset->depth();
		m_currentRGB->get(0)->updateTexture(m_buffers[m_selectedLayersKeys[m_currentImageIndex]].data(), false);
		m_currentRGB->get(1)->updateTexture(m_buffers[m_selectedLayersKeys[m_currentImageIndex]].data()+layerSize, false);
		m_currentRGB->get(2)->updateTexture(m_buffers[m_selectedLayersKeys[m_currentImageIndex]].data()+layerSize*2, false);
		m_currentIso->updateTexture(m_buffers[m_selectedLayersKeys[m_currentImageIndex]].data()+layerSize*3, false);
	}
}

FixedRGBLayersFromDataset* FixedRGBLayersFromDataset::createDataFromDatasetWithUI(QString name,
		WorkingSetManager *workingSet, Seismic3DAbstractDataset* dataset,
		QObject *parent) {
	QList<std::pair<QString, QString>> fileList; // iso first then rgb image

	QList<IData*> datas = workingSet->folders().horizonsFree->data();
	for (IData* data : datas) {
		IJKHorizon* horizon = dynamic_cast<IJKHorizon*>(data);
		if (horizon!=nullptr) {
			// check that seismic are compatible
			bool seismicCompatible = IJKHorizon::filterHorizon(horizon, dataset);


			if (seismicCompatible) {
				// get paths
				QString isoPath =  horizon->path();
				QFileInfo isoFileInfo(isoPath);

				QDir dir = isoFileInfo.dir();
				QString baseNameIso = isoFileInfo.baseName();
				QString rgbPath = dir.absoluteFilePath(baseNameIso+".png");
				QFileInfo rgbFileInfo(rgbPath);
				if (isoFileInfo.exists() && isoFileInfo.isReadable() && rgbFileInfo.exists() && rgbFileInfo.isReadable()) {
					fileList.push_back(std::pair<QString, QString>(isoPath, rgbPath));
				}
			}
		}
	}

	FixedRGBLayersFromDataset* outObj = nullptr;
	if (fileList.size()>0) {
		outObj = new FixedRGBLayersFromDataset(fileList, name, workingSet, dataset, parent);
	}
	return outObj;
}

//FixedRGBLayersFromDataset* FixedRGBLayersFromDataset::createDataFromDatasetWithList(
//		QList<std::pair<QString, QString>> layers, QString name, WorkingSetManager *workingSet,
//		Seismic3DAbstractDataset* dataset, QObject *parent) {
//
//	// size
//	std::size_t Nlayers = layers.size();
//	std::Size_t layerSize = 4 * dataset->width() * dataset->depth();
//	std::size_t totalSize = Nlayers * layerSize * sizeof(float);
//
//	std::vector<float> cpuBuffer;
//	cpuBuffer.resize(layerSize);
//	float* cudaBuffer;
//	checkCudaErrors(cudaMalloc(&cudaBuffer, totalSize));
//
//	for (std::size_t i=0; i<layers.size(); i++) {
//		// read iso and rgb
//		FILE* file = fopen(layers.first.toStdString().c_str(), "r");
//		GDALDataset* poDataset;
//		bool valid = file != nullptr;
//
//		if (valid) {
//			m_buffers.append(std::vector<float>());
//			m_buffers.last().resize(layerSize);
//			QString name = QFileInfo(layers.second).baseName();
//			m_layers.push_back(QList<std::pair<QString, float*>>(name, m_buffers.last().data()));
//
//			fread(m_buffers.last().data()+3*dataset->width()*dataset->depth(), sizeof(float), dataset->width() * dataset->depth(), file);
//			fclose(file);
//
//			GDALDataset* poDataset = (GDALDataset*) GDALOpen(imageFilePath.toStdString().c_str(),
//							GA_ReadOnly);
//			isValid = poDataset == nullptr;
//		}
//		std::size_t width, height;
//		if (isValid) {
//			width = poDataset->GetRasterXSize();
//			height = poDataset->GetRasterYSize();
//
//			isValid = height==dataset->depth() && width==dataset->width();
//		}
//		if (isValid) {
//			GDALRasterBand* hBand = (GDALRasterBand*) GDALGetRasterBand(poDataset, 1);
//			ImageFormats::QColorFormat colorFormat = GDALLoader::getColorFormatType(poDataset);
//			ImageFormats::QSampleType sampleType = GDALLoader::getSampleType(hBand);
//			GDALDataType type = GDALGetRasterDataType(hBand);
//			int offset = GDALGetDataTypeSizeBytes(type);
//			int numBands = oDataset->GetRasterCount();
//
//			std::vector<char> tmpBuf;
//			tmpBuf.resize(width*height*numBands*offset);
//
//			isValid = poDataset->RasterIO(GF_Read, 0, 0, width, height, tmpBuf.data(), width, height,
//							type, numBands, nullptr, numBands * offset,
//							numBands * offset * width, offset)==CPLErr::CE_None;
//
//			GDALClose(poDataset);
//
//
//			if (isValid) {
//				copyGDALBufToFloatBufPlanar(tmpBuf.data(), m_buffers.last().data(), width, height, numBands, offset, colorFormat, sampleType, hBand);
//			}
//		} else if (poDataset!=nullptr) {
//			GDALClose(poDataset);
//		}
//
//		// copy to gpu
//		if (isValid) {
////			checkCudaErrors(cudaMemcpy(cudaBuffer + layerSize*i, cpuBuffers.data(), layerSize * sizeof(float), cudaMemcpyHostToDevice));
//		} else {
//			m_buffers.removeLast();
//			m_layers.removeLast();
//		}
//	}
//
//	// create indexes + get iso mean for all layers
//	QList<std::pair<long, double>> listIdxAndMean;
//	for (long i=0; i<m_layers.size(); i++) {
//		double mean = 0;
//		std::size_t w = dataset->width();
//		std::size_t d = dataset->depth();
//		std::size_t N = w * d;
//		float* isoTab = m_layers.second + 3*N;
//		for (std::size_t i=0; i<w; i++) {
//			double tmpMean = 0;
//			for (std::size_t j=0; j<d; j++) {
//				tmpMean += isoTab[i*d+j];
//			}
//			tmpMean /= d;
//			mean += tmpMean;
//		}
//		mean /= w;
//
//		listIdxAndMean.push_back(std::pair<long, double>(i, mean));
//	}
//
//	// reorder indexes
//	std::sort(listIdxAndMean.begin(), listIdxAndMean.end(), [](const <std::pair<long, double>& a, const <std::pair<long, double>& b) {
//		return a.second < b.second;
//	});
//	for (std::pair<long, double> e : listIdxAndMean) {
//		m_layersKeys.push_back(e.first);
//	}
//
//	// select all
//	m_selectedLayersKeys = m_layersKeys;
//}
