#include "cudargbimage.h"
#include <QOpenGLTexture>

CUDARGBImage::CUDARGBImage(int width, int height,
		ImageFormats::QSampleType type, const IGeorefImage * const transfoProvider,QObject *parent) :
		QObject(parent) {

	for (int i = 0; i < 3; i++)
		m_GPUImages.push_back(
				new CUDAImagePaletteHolder(width, height, type, transfoProvider,this));

	m_opacity = 1.0;
}

QVector2D CUDARGBImage::rangeRatio(int i) {
	return m_GPUImages[i]->rangeRatio();
}

CUDARGBImage::~CUDARGBImage() {

}

int CUDARGBImage::width() const {
	return m_GPUImages[0]->width();
}
int CUDARGBImage::height() const {
	return m_GPUImages[0]->height();
}

float CUDARGBImage::opacity() const {
	return m_opacity;
}

void CUDARGBImage::setOpacity(float value) {
	m_opacity = value;
	emit opacityChanged(value);
}

void CUDARGBImage::setRange(unsigned int i, const QVector2D &range) {
	m_GPUImages[i]->setRange(range);
	emit rangeChanged(i,range);
}

QVector<IPaletteHolder*> CUDARGBImage::holders() const {
	QVector<IPaletteHolder*> result;
	for (CUDAImagePaletteHolder *el : m_GPUImages)
		result.push_back(el);

	return result;
}

void CUDARGBImage::lockPointer() {
	for (CUDAImagePaletteHolder *el : m_GPUImages)
		el->lockPointer();
}
void CUDARGBImage::unlockPointer() {
	for (CUDAImagePaletteHolder *el : m_GPUImages)
		el->unlockPointer();
}


void CUDARGBImage::swapCudaPointer() {
	for (CUDAImagePaletteHolder *el : m_GPUImages)
		el->swapCudaPointer();
}

