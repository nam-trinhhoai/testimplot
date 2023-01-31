#ifndef GDALImageWrapper_H_
#define GDALImageWrapper_H_

#include <QObject>
#include <QVector2D>
#include <QMatrix4x4>
#include "qhistogram.h"

#include "imageformats.h"

class QOpenGLTexture;
class GDALDataset;
class GDALRasterBand;

class GDALImageWrapper: public QObject {
Q_OBJECT
public:
	GDALImageWrapper(QObject *parent = 0);
	~GDALImageWrapper();

	int width() const;
	int height() const;

	virtual bool open(const QString &imageFilePath);
	virtual void close();
	double noDataValue(bool &hasNoData) const;

	virtual QVector2D computeRange() const;
	virtual QHistogram computeHistogram(const QVector2D &range,
			int nBuckets) const;

	//Coordinate system transformations
	virtual void worldToImage(double worldX, double worldY, double &imageX,
			double &imageY);
	virtual void imageToWorld(double imageX, double imageY, double &worldX,
			double &worldY);
	QRectF worldExtent();

	virtual QMatrix4x4 imageToWorldTransformation();

	ImageFormats::QColorFormat colorFormat() const {
		return m_colorFormat;
	};

	ImageFormats::QSampleType sampleType() const {
		return m_sampleType;
	};

	int numBands() const;
	bool readData(int i0, int j0, int width, int height, void *dest,
			int numBands);

private:
	ImageFormats::QColorFormat m_colorFormat;
	ImageFormats::QSampleType m_sampleType;

	GDALDataset *poDataset;
	GDALRasterBand *hBand;
	int m_width;
	int m_height;

	double m_geoTransform[6];
	double m_invGeoTransform[6];
};

#endif /* QTLARGEIMAGEVIEWER_QGLSIMPLEIMAGE_H_ */
