#include "seismic3ddataset.h"
#include "fixedlayerfromdataset.h"
#include "affinetransformation.h"
#include "iabstractisochrone.h"

#include <vector>

#include <QDebug>

template<typename InputCubeType>
AbstractKohonenProcess* AbstractKohonenProcess::CreateKohonenProcessObjectKernel<InputCubeType>::run(Seismic3DDataset* dataset, int channel) {
	return new KohonenProcess<InputCubeType>(dataset, channel);
}

template<typename CubeType>
KohonenProcess<CubeType>::KohonenProcess(Seismic3DDataset* dataset, int channel) :
	AbstractKohonenProcess(dataset, channel) {
	m_dimI = m_dataset->width();
	m_dimJ = m_dataset->depth();


	m_dimx = m_dataset->height();
	m_buffer.resize(m_dimx);

	m_pasech = m_dataset->sampleTransformation()->a();
	m_tdeb = m_dataset->sampleTransformation()->b();
}

template<typename CubeType>
KohonenProcess<CubeType>::~KohonenProcess() {

}

template<typename CubeType>
bool KohonenProcess<CubeType>::isExampleValid(long tabSize, long i, long j) const {
	bool ok;
	float isoVal = m_isochrone->getValue(i, j, &ok);
	long no_ech = (isoVal - m_tdeb)/ m_pasech;

	bool returnVal = !isNonValue(isoVal) && ok && no_ech+tabSize<=m_dimx && i>=0 && i<m_dimI && j>=0 && j<m_dimJ;

	return returnVal;
}

template<typename CubeType>
bool KohonenProcess<CubeType>::getExample(float* tab, long tabSize, long i, long j) const {
	bool ok;
	float isoVal = m_isochrone->getValue(i, j, &ok);
	long no_ech = (isoVal - m_tdeb)/ m_pasech;

	bool returnVal = !isNonValue(isoVal) && ok && no_ech+tabSize<=m_dimx && i>=0 && i<m_dimI && j>=0 && j<m_dimJ;
	if(returnVal) {
		m_dataset->readTraceBlockAndSwap(m_buffer.data(), i, i+1, j);
		for(int k=0; k < tabSize; k++) {
			tab[k] = (float) m_buffer[(no_ech +k)*m_dataset->dimV()+m_channel] ;
		}
	}
	return returnVal;
}
