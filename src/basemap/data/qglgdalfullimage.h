#ifndef QGLGDALFullImage_H_
#define QGLGDALFullImage_H_

#include <QVector2D>
#include <QMatrix4x4>
#include "qhistogram.h"
#include "lookuptable.h"
#include "qglabstractfullimage.h"

class GDALImageWrapper;
class QOpenGLTexture;

//A full image read from GDAL: as a proof of concept. Should be abstracted to a more higher level interface
class QGLGDALFullImage : public QGLAbstractFullImage
{
	Q_OBJECT
	Q_PROPERTY(int width READ width CONSTANT)
	Q_PROPERTY(int height READ height CONSTANT)
public:
	QGLGDALFullImage(QObject *parent=0);
	~QGLGDALFullImage();

	virtual bool open(const QString& imageFilePath);
	virtual void close();

	virtual void bindTexture(unsigned int unit) override;
	virtual void releaseTexture(unsigned int unit) override;

	//IPaletteHolder
	virtual QHistogram computeHistogram(const QVector2D &range, int nBuckets) override;

	//IGeorefImage
	int width() const override;
	int height() const override;
	virtual void worldToImage(double worldX, double worldY,double &imageX, double &imageY) const override;
	virtual void imageToWorld(double imageX, double imageY,double &worldX, double &worldY) const override;
	virtual QMatrix4x4 imageToWorldTransformation() const override;

	bool valueAt(int i, int j, double & value) const override;
	void valuesAlongJ(int j, bool* valid,double* values)const override;
	void valuesAlongI(int i, bool* valid,double* values)const override;
protected:
	QVector2D computeRange() const override;
private:
	GDALImageWrapper  *m_internalImage;
	QByteArray m_buffer;
	QOpenGLTexture *m_texture = nullptr;

	QHistogram m_cachedHisto;
};


#endif /* QTLARGEIMAGEVIEWER_QGLSIMPLEIMAGE_H_ */
