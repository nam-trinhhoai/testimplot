#include "amplitudestratisliceattribute.h"
#include "seismic3dabstractdataset.h"
#include "stratisliceamplitudeattributegraphicrepfactory.h"
#include "cudaimagepaletteholder.h"
#include "cuda_volume.h"
#include "seismic3dcudadataset.h"
#include "seismic3ddataset.h"
#include  "cudargttile.h"
#include <iostream>
#include <QElapsedTimer>
#include "cuda_common_helpers.h"
#include "datasetbloccache.h"
#include "affine2dtransformation.h"
#include "stratislice.h"



AmplitudeStratiSliceAttribute::AmplitudeStratiSliceAttribute(WorkingSetManager *workingSet,
		StratiSlice *slice, QObject *parent) :
		AbstractStratiSliceAttribute(workingSet, slice, parent) {
	m_image = new CUDAImagePaletteHolder(slice->width(), slice->depth(),
			ImageFormats::QSampleType::INT16, slice->seismic()->ijToXYTransfo(),
			this);

	m_repFactory = new StratiSliceAmplitudeAttributeGraphicRepFactory(this);
}

void AmplitudeStratiSliceAttribute::loadSlice(unsigned int z) {
	AbstractStratiSliceAttribute::loadSlice(z);
	AbstractStratiSliceAttribute::loadSlice(m_isoSurfaceHolder, m_image, m_extractionWindow, z);
}

QString AmplitudeStratiSliceAttribute::name() const {
	return "RMS Amplitude attribute";
}

IGraphicRepFactory* AmplitudeStratiSliceAttribute::graphicRepFactory() {
	return m_repFactory;
}

AmplitudeStratiSliceAttribute::~AmplitudeStratiSliceAttribute() {
}

