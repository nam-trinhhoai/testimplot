/*
 * MorletProcess.cpp
 *
 *  Created on: 2 mars 2020
 *      Author: l0222891
 */


#include "MorletProcess.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>

#include <vector>
#include <memory>
#include <unordered_map>
#include <math.h>
#include <cmath>

#include "RGTMorletMemory.cuh"

#include <datasets/DatasetGroup.h>
#include <DatasetCubeUtil.h>
#include "SampleTypeBinder.h"
#include "util/SeismicUtil.h"

typedef float2 Complex;

template <typename InputCubeType>
struct computeModules {
	static short* run(const ToAnalyse3Process* pr, int freqMin, int freqMax, int freqStep,
			std::vector<process::RgtSeed> seeds, bool polarity, bool useSnap, bool useMedian,
			int lwx, int distancePower, int snapWindow) {

		const MorletProcess<InputCubeType>* spectrumProcess =
				dynamic_cast<const MorletProcess<InputCubeType>*> (pr);

		int type = polarity ? 1 : -1;

//		QMutexLocker lock(&spectrumProcess->m_mutex);

		const io::InputOutputCube<InputCubeType>* iocubeS =
				dynamic_cast<const io::InputOutputCube<InputCubeType>*>(spectrumProcess->getCubeS());
		io::CubeDimension dims = spectrumProcess->getCubeS()->getDim();
		int dimx = dims.getI();
		int dimy = dims.getJ();
		io::CubeStep steps = spectrumProcess->getCubeS()->getSteps();
		double sampleRate = steps.getI() / 1000;
		CubeSeismicAddon seismicAddon = spectrumProcess->getCubeS()->getSeismicAddon();
		const io::InputOutputCube<short>* iocubeT =
				dynamic_cast<const io::InputOutputCube<short>*>(spectrumProcess->getCubeT());
		double firstSample = seismicAddon.getFirstSample();
		double sampleStep = seismicAddon.getSampleStep();

	    int halfWindow = 60;
	    int filterKernelSize = 2*halfWindow + 1;
	    int n_cycles = 3 ;

		float hat_pow = 10;
		short* module;

		int nbKernel = (freqMax - freqMin) / freqStep;
		int nbOutputSlices = 2 + nbKernel;
		std::size_t sizeLayer = dimy * dims.getK();
		std::size_t sizeInline = dimx * dimy;
		module = new short[ sizeLayer * (nbOutputSlices) ];
		memset(module, 0, sizeLayer * (nbOutputSlices)*sizeof(short));

		Complex *h_signal = new Complex[ sizeLayer * filterKernelSize];

		// sort seed vector
		std::sort(seeds.begin(), seeds.end(), [](process::RgtSeed a, process::RgtSeed b){
			return a.rgtValue<b.rgtValue;
		});

		#pragma omp parallel for schedule (dynamic)
		for (std::size_t iz = 0; iz < dims.getK(); iz++ ) {
			printf(" numplan %d/%d\n",iz,dims.getK()) ;
			std::vector<double> dist;
			dist.resize(seeds.size());

			short* rgtData = new short[sizeInline];
			InputCubeType* rawData = new InputCubeType[sizeInline];

			iocubeS->readSubVolume(0, 0, iz, dimx, dimy, 1, rawData);
			iocubeT->readSubVolume(0, 0, iz, dimx, dimy, 1, rgtData);

			std::vector<int> points;
			points.resize(dimy);

			for (std::size_t iy=0; iy<dimy; iy++) {
				double som=0.0 ;

				int ix;
				bool seedFound = false;
				for(int i=0; (i < seeds.size()) && !seedFound; i++) {
					long val = ((iy - seeds[i].y)*(iy-seeds[i].y) + (iz -seeds[i].z)*(iz-seeds[i].z));
					if (val!=0) {
						dist[i] = 1.0 / std::pow(val,distancePower) ;
						som += dist[i] ;
					} else {
						ix = seeds[i].x;
						seedFound = true;
					}
				}

				if(!seedFound) {
					ix = 0 ;
					double weightedIso = 0;
					for(int i=0; i < seeds.size() ; i++) {
						while ( (rgtData[iy*dimx + ix])  < seeds[i].rgtValue  && ix < dimx ) {
							ix ++ ;
						}
						weightedIso += ix*dist[i] ;
					}
					ix = std::round(weightedIso/som) ;
					if (ix>=dimx) {
						ix = dimx - 1;
					} else if (ix<0) {
						ix = 0;
					}
				}
				if (useSnap) {
					int newx = process::bl_indpol(ix, rawData+iy*dims.getI(), dims.getI(), type, snapWindow);
					ix = (newx==process::SLOPE::RAIDE)? ix : newx;
				}
				points[iy] = ix;
			}

			if (useMedian) {
				process::UtFiltreMedianeX(points.data(), points.size(), 1, lwx);
			}

			for (std::size_t iy=0; iy<dimy; iy++) {
				int ix = points[iy];
				module[iz*dimy + iy] = firstSample + ix * sampleStep;
				module[sizeLayer + iz*dimy + iy] = rawData[iy*dimx + ix] ;

				for (int j=0; j<filterKernelSize; j++) {
	                long ind = (iz*dimy + iy)*filterKernelSize + j;
	                if ( dimx > ix + j - filterKernelSize/2 >= 0)
	                	h_signal[ind].x =  rawData[iy*dimx + ix + j - filterKernelSize/2];
	                else
	                	h_signal[ind].x = 0;
	                h_signal[ind].y =  0.0 ;

				}
			}
			delete rawData;
			delete rgtData;
		}

		RGTMorletMemory(h_signal, module, (size_t) dimy, (size_t) dims.getK(),
				filterKernelSize,
				freqMin, freqMax, freqStep, sampleRate, n_cycles);

		// Print pour debug
		int i =0;
		for(int freq = freqMin; freq < freqMax; freq += freqStep) {
			printf("Freq %d: ", freq);
			for ( int j = 0; j < 10; j++)
				printf(" %d", module[sizeLayer * i + sizeLayer / 2 -5 + j]);
			printf("\n");
			i++;
		}

		delete h_signal;
		printf(" End of MorletProcess::computeModules\n") ;

		return module;
	}
};

ToAnalyse3Process::ToAnalyse3Process(std::unique_ptr<io::Cube>&& cubeS, std::unique_ptr<io::Cube>&& cubeT) :
		LayerProcess(std::move(cubeS), std::move(cubeT)) {

}

void ToAnalyse3Process::compute() {
	if (! m_isComputed && m_computeMutex.try_lock()) {

		//m_nbOutputSlices = 2 + (m_freqMax - m_freqMin) / m_freqStep;

		io::SampleTypeBinder binder0(m_cubeS->getNativeType());
		short* buf = binder0.bind<computeModules> (this, m_freqMin, m_freqMax, m_freqStep,
				m_seeds, m_polarity, m_useSnap, m_useMedian, m_lwx_medianFilter, m_distancePower,
				m_snapWindow);

		{
			QMutexLocker lockCache(&m_cacheMutex);
			if (m_module!=nullptr) {
				delete m_module;
			}
			m_module = buf;
			m_nbOutputSlices = 2 + (m_freqMax - m_freqMin) / m_freqStep;
			m_isComputed = true;
			m_computeMutex.unlock();
		}

		emit LayerProcess::processCacheIsReset();
	}
}

const short* ToAnalyse3Process::getModuleData(std::size_t spectrumSlice) const {
	QMutexLocker cacheLock(&m_cacheMutex);
	if (spectrumSlice >= m_nbOutputSlices) {
		printf( "ToAnalyse3Process::getModuleData Error Slice outside range %d / % d\n", spectrumSlice,
				m_nbOutputSlices);
		spectrumSlice = 0;
	}
	return m_module + (spectrumSlice * m_dimW * m_dimH);
}

