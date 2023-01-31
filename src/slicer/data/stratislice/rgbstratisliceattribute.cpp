#include "rgbstratisliceattribute.h"
#include "cudargbimage.h"
#include "cudaimagepaletteholder.h"
#include "affine2dtransformation.h"
#include "seismic3dabstractdataset.h"
#include "stratislice.h"
#include "stratislicergbattributegraphicrepfactory.h"

RGBStratiSliceAttribute::RGBStratiSliceAttribute(WorkingSetManager *workingSet,
		StratiSlice *stratislice, QObject *parent) :
		AbstractStratiSliceAttribute(workingSet, stratislice, parent) {
	m_image = new CUDARGBImage(stratislice->width(), stratislice->depth(),
			ImageFormats::QSampleType::FLOAT32, stratislice->seismic()->ijToXYTransfo(),
			this);

	m_repFactory = new StratiSliceRGBAttributeGraphicRepFactory(this);
	resetFrequencies();
}


void RGBStratiSliceAttribute::setExtractionWindow(uint w) {
	resetFrequencies();
	AbstractStratiSliceAttribute::setExtractionWindow(w);
}

int RGBStratiSliceAttribute::redIndex() const {
	return m_freqIndex[0];
}
int RGBStratiSliceAttribute::greenIndex() const {
	return m_freqIndex[1];
}
int RGBStratiSliceAttribute::blueIndex() const {
	return m_freqIndex[2];
}

void RGBStratiSliceAttribute::setRedIndex(int value) {
	m_freqIndex[0] = value;
	loadSlice(m_currentSlice);
	emit frequencyChanged();
}
void RGBStratiSliceAttribute::setGreenIndex(int value) {
	m_freqIndex[1] = value;
	loadSlice(m_currentSlice);
	emit frequencyChanged();
}
void RGBStratiSliceAttribute::setBlueIndex(int value) {
	m_freqIndex[2] = value;
	loadSlice(m_currentSlice);
	emit frequencyChanged();
}

void RGBStratiSliceAttribute::resetFrequencies() {
	m_freqIndex[0] = 0;
	m_freqIndex[1] = 1;
	m_freqIndex[2] = 2;
}

void RGBStratiSliceAttribute::loadSlice(unsigned int z) {
	AbstractStratiSliceAttribute::loadSlice(z);
	AbstractStratiSliceAttribute::loadRGBSlice(m_isoSurfaceHolder, m_image, m_extractionWindow,
			z, m_freqIndex[0], m_freqIndex[1], m_freqIndex[2]);
}

QString RGBStratiSliceAttribute::name() const {
	return "RGB blending";
}

IGraphicRepFactory* RGBStratiSliceAttribute::graphicRepFactory() {
	return m_repFactory;
}

RGBStratiSliceAttribute::~RGBStratiSliceAttribute() {
}

