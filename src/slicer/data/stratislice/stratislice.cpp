#include "stratislice.h"

#include "seismic3dabstractdataset.h"
#include "rgbstratisliceattribute.h"
#include "amplitudestratisliceattribute.h"
#include "frequencystratisliceattribute.h"
#include "stratislicegrahicrepfactory.h"

StratiSlice::StratiSlice(WorkingSetManager *workingSet,
		Seismic3DAbstractDataset *seismic, Seismic3DAbstractDataset *rgt,
		QObject *parent) :
		IData(workingSet, parent) {
	m_seismic = seismic;
	m_rgt = rgt;


	m_repFactory=new StratiSliceGraphicRepFactory(this);

	m_rgbAttribute=new RGBStratiSliceAttribute(workingSet,this,this);
	m_amplitudeAttribute=new AmplitudeStratiSliceAttribute(workingSet,this,this);
	m_frequencyAttribute=new FrequencyStratiSliceAttribute(workingSet,this,this);
}


unsigned int StratiSlice::width() const {
	return m_seismic->width();
}
unsigned int StratiSlice::height() const {
	return m_seismic->height();
}

unsigned int StratiSlice::depth() const {
	return m_seismic->depth();
}

QString StratiSlice::name() const {
	return "RGT on " + m_seismic->name();
}

QUuid StratiSlice::dataID() const {
	return m_seismic->dataID();
}

QVector2D StratiSlice::rgtMinMax() {
	return m_rgt->minMax(0); // because this is only called with dataset with dimV==1
}

StratiSlice::~StratiSlice() {
}

