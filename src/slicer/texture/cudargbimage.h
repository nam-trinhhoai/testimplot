#ifndef CUDARGBImage_H
#define CUDARGBImage_H

#include <QWidget>
#include "qhistogram.h"
#include "ipaletteholder.h"
#include "lookuptable.h"
#include <QVector2D>

#include "cudaimagepaletteholder.h"

class CUDARGBImage: public QObject {
	Q_OBJECT
	Q_PROPERTY(float opacity READ opacity WRITE setOpacity NOTIFY opacityChanged)
public:
	CUDARGBImage(int width, int height,ImageFormats::QSampleType type=ImageFormats::QSampleType::INT16, const IGeorefImage * const transfoProvider=nullptr,QObject *parent = 0);
	virtual ~CUDARGBImage();

	int width() const;
	int height() const;

	float opacity() const;

	QVector2D rangeRatio(int i);

	void swapCudaPointer();
	void lockPointer();
	void unlockPointer();

	QVector<IPaletteHolder*> holders() const;
	CUDAImagePaletteHolder * get(int i){return m_GPUImages[i];}

public slots:
	void setRange(unsigned int i,const QVector2D & range);
	void setOpacity(float value);
signals:
	void opacityChanged(float val);
	void rangeChanged(unsigned int i, const QVector2D & range);
private:
	QVector<CUDAImagePaletteHolder*> m_GPUImages;
	float m_opacity;

};

#endif /* QTCUDAIMAGEVIEWER_SRC_IMAGEITEM_GPUIMAGEPALETTEHOLDER_H_ */
