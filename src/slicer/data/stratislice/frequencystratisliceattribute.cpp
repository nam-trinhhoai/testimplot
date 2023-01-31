#include "frequencystratisliceattribute.h"

#include "cudaimagepaletteholder.h"
#include "affine2dtransformation.h"
#include "seismic3dabstractdataset.h"
#include "stratislice.h"
#include "stratislicefrequencyattributegraphicrepfactory.h"

FrequencyStratiSliceAttribute::FrequencyStratiSliceAttribute(
		WorkingSetManager *workingSet, StratiSlice *stratislice,
		QObject *parent) :
		AbstractStratiSliceAttribute(workingSet, stratislice, parent) {

	m_repFactory = new StratiSliceFrequencyAttributeGraphicRepFactory(this);

	m_currentImg = new CUDAImagePaletteHolder(
					stratiSlice()->width(), stratiSlice()->depth(),
					ImageFormats::QSampleType::FLOAT32,
					stratiSlice()->seismic()->ijToXYTransfo(), this);
	resetFrequencies();
}

void FrequencyStratiSliceAttribute::setExtractionWindow(uint w) {
	AbstractStratiSliceAttribute::setExtractionWindow(w);
	resetFrequencies();
}

int FrequencyStratiSliceAttribute::index() const {
	return m_currentIndex;
}

void FrequencyStratiSliceAttribute::setIndex(int value) {
	m_currentIndex = value;
	updateExposedImage();
	emit indexChanged();
}

void FrequencyStratiSliceAttribute::updateExposedImage()
{
	m_images[m_currentIndex]->lockPointer();
	m_currentImg->updateTexture(m_images[m_currentIndex]->backingPointer(),false);
	m_images[m_currentIndex]->unlockPointer();
}

void FrequencyStratiSliceAttribute::loadSlice(unsigned int z) {
	AbstractStratiSliceAttribute::loadSlice(z);
	AbstractStratiSliceAttribute::loadFrequencySlice(m_isoSurfaceHolder, m_images, m_extractionWindow,
			z);
	updateExposedImage();
}

int FrequencyStratiSliceAttribute::frequencyCount()
{
	return (extractionWindow() / 2 + 1);
}

void FrequencyStratiSliceAttribute::resetFrequencies() {
	m_currentIndex = 0;
	int freqCount = frequencyCount();

	for (CUDAImagePaletteHolder *img : m_images)
		delete img;
	m_images.clear();
	for (int i = 0; i < freqCount; i++) {
		CUDAImagePaletteHolder *img = new CUDAImagePaletteHolder(
				stratiSlice()->width(), stratiSlice()->depth(),
				ImageFormats::QSampleType::FLOAT32,
				stratiSlice()->seismic()->ijToXYTransfo(), this);
		m_images.push_back(img);
	}
}

QString FrequencyStratiSliceAttribute::name() const {
	return "Frequency attribute";
}

IGraphicRepFactory* FrequencyStratiSliceAttribute::graphicRepFactory() {
	return m_repFactory;
}

FrequencyStratiSliceAttribute::~FrequencyStratiSliceAttribute() {
}

