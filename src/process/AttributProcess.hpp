/*
 *
 *
 *  Created on: 5 janv. 2018
 *      Author: Georges
 */

#include "AttributProcess.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>

#include <vector>
#include <memory>
#include <unordered_map>
#include <math.h>
#include <cmath>

#include "RGT_Spectrum_Memory.cuh"

#include "seismic3ddataset.h"
#include "palette/imageformats.h"
#include "RgtLayerProcessUtil.h"

using namespace std;

template <typename InputCubeType>
AttributProcess<InputCubeType>::AttributProcess(
		Seismic3DDataset *cubeS, int channelS,
		Seismic3DDataset *cubeT, int channelT) :
	ToAnalyse5Process( cubeS, channelS, cubeT, channelT) {
	init();
}

template <typename InputCubeType>
AttributProcess<InputCubeType>::~AttributProcess() {
	if ( m_module != nullptr) {
		delete m_module;
		m_module = nullptr;
	}
}

template <typename InputCubeType>
void AttributProcess<InputCubeType>::init() {

	//io::Cube * cube = nullptr;
}

template <typename InputCubeType>
bool AttributProcess<InputCubeType>::isCompatible(
		e_SliceDirection lastDirection, int lastSlice) {
	return true;//( lastSlice == m_lastSlice && lastDirection == m_lastDirection );
}

template<typename InputCubeType> void AttributProcess<InputCubeType>::saveIn(
		Seismic3DAbstractDataset* modulesDataset) {
//
//	std::unique_ptr<Cube> outputCube = process::openOrCreateCube<short>(modulesDataset);
//	io::InputOutputCube<short>* cubeOut = dynamic_cast<InputOutputCube<short>*>(outputCube.get());
//	QMutexLocker lock(&m_cacheMutex);
//
//	int nb = 0;
//	for (int iz = 0; iz < m_nbOutputSlices; iz++) {
//		size_t indexInModule = iz * m_dimW * m_dimH;
//		cubeOut->writeSubVolume( 0, 0, iz, &m_module[indexInModule], m_dimW, m_dimH, 1) ;
//	}
}
