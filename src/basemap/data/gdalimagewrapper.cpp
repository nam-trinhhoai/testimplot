#include "gdalimagewrapper.h"

#include <QFileInfo>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include "gdalloader.h"
#include "gdal.h"
#include "gdal_priv.h"

GDALImageWrapper::GDALImageWrapper(QObject *parent) :
		QObject(parent) {
	m_sampleType = ImageFormats::QSampleType::UINT8;
	m_colorFormat = ImageFormats::QColorFormat::RGB_INTERLEAVED;
	poDataset = nullptr;
}

GDALImageWrapper::~GDALImageWrapper() {
	close();
}

void GDALImageWrapper::close() {
	if (poDataset)
		GDALClose(poDataset);

	poDataset = nullptr;
}

int GDALImageWrapper::width() const {
	return m_width;
}
int GDALImageWrapper::height() const {
	return m_height;
}

void GDALImageWrapper::worldToImage(double worldX, double worldY,
		double &imageX, double &imageY) {
	GDALApplyGeoTransform(m_invGeoTransform, worldX, worldY, &imageX, &imageY);
}
void GDALImageWrapper::imageToWorld(double imageX, double imageY,
		double &worldX, double &worldY) {
	GDALApplyGeoTransform(m_geoTransform, imageX, imageY, &worldX, &worldY);
}

QMatrix4x4 GDALImageWrapper::imageToWorldTransformation() {
	return QMatrix4x4((float) m_geoTransform[1], (float) m_geoTransform[2],
			0.0f, (float) m_geoTransform[0], (float) m_geoTransform[4],
			(float) m_geoTransform[5], 0.0f, (float) m_geoTransform[3], 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
}

bool GDALImageWrapper::open(const QString &imageFilePath) {
	close();
	bool done = false;
	do {
		// check file
		QFileInfo fileInfo(imageFilePath);
		if (!fileInfo.exists()) {
			break;
		}

		// probing file format
		QFile file(imageFilePath);
		if (!file.open(QIODevice::ReadOnly)) {
			break;
		}
		poDataset = (GDALDataset*) GDALOpen(imageFilePath.toStdString().c_str(),
				GA_ReadOnly);
		if (poDataset == nullptr) {
			break;
		}

//		double angle = -20 * 3.14 / 360;

		poDataset->GetGeoTransform(m_geoTransform);
//		m_geoTransform[1]=std::cos(angle);
//		m_geoTransform[2]=-std::sin(angle);
//
//		m_geoTransform[4]=std::sin(angle);
//		m_geoTransform[5]=std::cos(angle);

		int result = GDALInvGeoTransform(m_geoTransform, m_invGeoTransform);
		if (!result) {
			std::cerr << "Invertible transfo" << std::endl;
		}

		hBand = (GDALRasterBand*) GDALGetRasterBand(poDataset, 1);
		m_colorFormat = GDALLoader::getColorFormatType(poDataset);
		m_sampleType = GDALLoader::getSampleType(hBand);
		m_width = poDataset->GetRasterXSize();
		m_height = poDataset->GetRasterYSize();

		done = true;
	} while (0);

	if (!done) {
		close();
	}

	return done;
}

int GDALImageWrapper::numBands() const {
	return poDataset->GetRasterCount();
}

bool GDALImageWrapper::readData(int i0, int j0, int width, int height,
		void *dest, int numBands) {
	GDALDataType type = GDALGetRasterDataType(hBand);
	int offset = GDALGetDataTypeSizeBytes(type);
	std::cout<<this->width()<<"\t"<<this->height()<<" Asked:"<<i0<<"\t"<<j0<<"\t"<<width<<"\t"<<height<<std::endl;


	if(poDataset->RasterIO(GF_Read, i0, j0, width, height, dest, width, height,
			GDALGetRasterDataType(hBand), numBands, nullptr, numBands * offset,
			numBands * offset * width, offset)!=CPLErr::CE_None)
	{
		std::cerr<<"Failed to read data"<<std::endl;
		return false;
	}
	return true;
}

double GDALImageWrapper::noDataValue(bool &hasNoData) const {
	int hasNoDataValue;
	double val=hBand->GetNoDataValue(&hasNoDataValue);
	hasNoData = hasNoDataValue;
	return val;
}

QVector2D GDALImageWrapper::computeRange() const {
	GDALRasterBand *hBand = (GDALRasterBand*) GDALGetRasterBand(poDataset, 1);
	double minmax[2];
	hBand->ComputeRasterMinMax(TRUE, minmax);
	return QVector2D (minmax[0],minmax[1]);
}

QHistogram GDALImageWrapper::computeHistogram(const QVector2D &range,
		int nBuckets) const {
	QHistogram histo;
	histo.setRange(range);

	//The default case load the whole data into the memory and sub sample
	GUIntBig hh[nBuckets];
	GDALRasterBand *hBand = (GDALRasterBand*) GDALGetRasterBand(poDataset, 1);
	hBand->GetHistogram(range.x(), range.y(), nBuckets, hh, 0, 0,
			GDALDummyProgress, nullptr);
	for (int i = 0; i < nBuckets; i++)
		histo[i] = hh[i];

	return histo;
}

