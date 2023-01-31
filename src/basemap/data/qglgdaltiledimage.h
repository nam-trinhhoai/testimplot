#ifndef QGLGDALTiledImage_H_
#define QGLGDALTiledImage_H_

#include <QVector2D>
#include <QMatrix4x4>

#include "qglabstracttiledimage.h"

class GDALImageWrapper;

class QGLGDALTiledImage : public QGLAbstractTiledImage
{
	Q_OBJECT
	Q_PROPERTY(int width READ width CONSTANT)
	Q_PROPERTY(int height READ height CONSTANT)
public:
	QGLGDALTiledImage(QObject *parent=0);
	~QGLGDALTiledImage();

	virtual bool open(const QString& imageFilePath);
	virtual void close();

	//IPaletteHolder
	virtual QHistogram computeHistogram(const QVector2D &range, int nBuckets) override;

	//IGeorefImage
	int width() const override;
	int height() const override;
	virtual void worldToImage(double worldX, double worldY,double &imageX, double &imageY) const override;
	virtual void imageToWorld(double imageX, double imageY,double &worldX, double &worldY) const override;
	virtual QMatrix4x4 imageToWorldTransformation() const override;

	virtual bool  valueAt(int i, int j,double & value) const override;
	void valuesAlongJ(int j, bool* valid,double* values)const override;
	void valuesAlongI(int i, bool* valid,double* values)const override;

	virtual bool fillTileData(int i0, int j0, int width, int height, void *dest) const override;
private:
	QVector2D computeRange() const override;
private:
	GDALImageWrapper  *m_internalImage;
	QHistogram m_cachedHisto;
};

#endif /* QTLARGEIMAGEVIEWER_QGLSIMPLEIMAGE_H_ */
