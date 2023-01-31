/*

 *
 *  Created on: 25 Mars 2020
 *      Author: Georges
 */

#include "LayerProcess.h"

#include <vector>
#include <math.h>
#include <cmath>
#include <complex>
#include <memory>
#include <QRect>
#include <QMutex>
#include <QMutexLocker>

#include <seismic3ddataset.h>


LayerProcess::LayerProcess(Seismic3DDataset* cubeS, int channelS,
		Seismic3DDataset* cubeT, int channelT):QObject() {

		m_cubeS = cubeS;
		m_channelS = channelS;
		m_cubeT = cubeT;
		m_channelT = channelT;

		m_dimW = m_cubeS->width();
		m_dimH = m_cubeS->depth();
}

LayerProcess::~LayerProcess() {
	if (m_module!=nullptr) {
		delete m_module;
		m_module = nullptr;
	}
};
