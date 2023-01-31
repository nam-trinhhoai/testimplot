/*
 * MorletProcess.cpp
 *
 *  Created on: 2 mars 2020
 *      Author: l0222891
 */


#include "AnisotropyProcess.h"

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_blas.h>

#include <omp.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <math.h>
#include <cmath>
#include <stdio.h>


template <typename DataType>
struct GetValueKernel {
	static double run(char* buffer, int channel) {
		DataType* buf = static_cast<DataType*>(static_cast<void*>(buffer));
		double val =  buf[channel];
		return val;
	}
};

template<typename DataType>
struct AnisotropyProcessReadKernel {
	static void run(Seismic3DDataset* dataset, char* buffer, int zDeb, int zLimit, bool swap) {
		dataset->readInlineBlock<DataType>(static_cast<DataType*>(static_cast<void*>(buffer)), zDeb, zLimit, swap);
	}
};

template <typename RgtType>
std::vector<std::vector<float>>* AnisotropyAbstractProcess::ComputeModulesKernel<RgtType>::run( AnisotropyAbstractProcess* pr,
		const std::vector<float>& constrainLayer,
		std::vector<ReferenceDuo>& reference,
		const std::vector<RgtSeed> &seeds,
		long dtauReference,
		bool useSnap, bool useMedian, int distancePower,
		const QList<std::tuple<Seismic3DDataset*, int, float>>& attributsDatasets,int windowSize,
		LayerSpectrumDialog *layerspectrumdialog) {
	SampleTypeBinder binder(pr->m_cubeS->sampleType());
	return binder.bind<ComputeModulesKernelLevel2>(pr,
			constrainLayer,
			reference,
			seeds,
			dtauReference,
			useSnap, useMedian, distancePower, attributsDatasets,
			windowSize, layerspectrumdialog);
}

template <typename RgtType>
template <typename SeismicType>
std::vector<std::vector<float>>* AnisotropyAbstractProcess::ComputeModulesKernel<RgtType>::ComputeModulesKernelLevel2<SeismicType>::run( AnisotropyAbstractProcess* pr,
		const std::vector<float>& constrainLayer,
		std::vector<ReferenceDuo>& reference,
		std::vector<RgtSeed> seeds,
		long dtauReference,
		bool useSnap, bool useMedian, int distancePower, const QList<std::tuple<Seismic3DDataset*, int, float>>& attributsDatasets,
		int windowSize,
		LayerSpectrumDialog *layerspectrumdialog) {

	fprintf(stderr, "ok\n");

	// override window size for now because the current version of the process only support window_size 1
	windowSize = 1;

	double firstSample = pr->m_cubeS->sampleTransformation()->b(); //TODO seismicAddon.getFirstSample();
	double sampleStep = pr->m_cubeS->sampleTransformation()->a(); //TODO seismicAddon.getSampleStep();
	int numberOfSamples = pr->m_cubeS->height(); // samples
	int numberOfTraces = pr->m_cubeS->width(); // traces
	int numberOfInlines = pr->m_cubeS->depth(); // Inline

	std::vector<std::vector<float>>* module;
	std::size_t sizeLayer = static_cast<std::size_t>(numberOfTraces) * numberOfInlines;
	std::size_t sizeInline = static_cast<std::size_t>(numberOfTraces) * numberOfSamples;

	module = new std::vector<std::vector<float>>(); //[ sizeLayer * (2 + windowSize/2) ];
	std::vector<float> initShortVec;
	module->resize(6, initShortVec);
	for (std::size_t attrIndex = 0; attrIndex<module->size(); attrIndex++) {
		(*module)[attrIndex].resize(sizeLayer, 0);
	}
	bool isConstrainSet = constrainLayer.size()==sizeLayer;

	bool isReferenceSet = reference.size()!=0;
	std::vector<int> referenceValues;

	if (isReferenceSet)
	{
		referenceValues.resize(reference.size());
		std::vector<RgtType> trace;
		trace.resize(numberOfSamples * pr->m_cubeT->dimV());
		pr->m_cubeT->readSubTraceAndSwap(trace.data(), 0, numberOfSamples, numberOfTraces/2, numberOfInlines/2);
		for (std::size_t index_ref=0; index_ref<reference.size(); index_ref++)
		{
			int pos_tau = (reference[index_ref].iso[numberOfTraces/2 + numberOfTraces * (numberOfInlines/2)] - firstSample) / sampleStep;
			referenceValues[index_ref] = trace[pos_tau * pr->m_cubeT->dimV()+pr->m_channelT];
		}

		std::vector<RgtType> sectionBuffer;
		sectionBuffer.resize(sizeInline * pr->m_cubeT->dimV());
		for (long index=0; index<seeds.size(); index++)
		{
			RgtSeed& seed = seeds[index];
			// read and apply
			bool isReadNeeded = false;
			for (std::size_t index_ref=0; index_ref<reference.size() && !isReadNeeded; index_ref++)
			{
				isReadNeeded = reference[index_ref].rgt[seed.y+seed.z*numberOfTraces] == -9999.0;
			}
			if (isReadNeeded)
			{
				pr->m_cubeT->readTraceBlockAndSwap(sectionBuffer.data(), 0, numberOfTraces, seeds[index].z);
				for (std::size_t iy=0; iy<numberOfTraces; iy++)
				{
					for (std::size_t indexRef=0; indexRef<reference.size(); indexRef++)
					{
						int indexTrace = (reference[indexRef].iso[iy + seed.z*numberOfTraces] - firstSample) / sampleStep;
						reference[indexRef].rgt[iy + seed.z*numberOfTraces] = sectionBuffer[(numberOfSamples*iy+indexTrace)*pr->m_cubeT->dimV()+pr->m_channelT];
					}
				}
			}
			seed.rgtValue = getNewRgtValueFromReference(seed.y, seed.z, seed.x, seed.rgtValue, firstSample, sampleStep, numberOfTraces, reference, referenceValues);
		}
	}

	std::sort(seeds.begin(), seeds.end(), [](RgtSeed a, RgtSeed b){ return a.rgtValue<b.rgtValue; });

	std::vector<std::vector<SeismicType>> ioCubesS;
	std::vector<std::vector<RgtType>> ioCubesT;
	std::vector<SeismicType> initS;
	std::vector<RgtType> initT;
	ioCubesS.resize(20, initS);
	ioCubesT.resize(20, initT);

	std::vector<std::vector<std::vector<char>>> attributsBuffers;
	std::vector<std::vector<char>> initAttributes;
	std::vector<char> initBufferChar;
	attributsBuffers.resize(20, initAttributes);

	for (int i=0; i<ioCubesS.size(); i++)
	{
		ioCubesS[i].resize(numberOfSamples*numberOfTraces* pr->m_cubeS->dimV());
		ioCubesT[i].resize(numberOfSamples*numberOfTraces* pr->m_cubeT->dimV());
		attributsBuffers[i].resize(attributsDatasets.count(), initBufferChar);

		for (int j=0; j<attributsDatasets.count(); j++) {
			attributsBuffers[i][j].resize(numberOfSamples*numberOfTraces* std::get<0>(attributsDatasets[j])->dimV() * std::get<0>(attributsDatasets[j])->sampleType().byte_size());
		}
	}

	omp_set_num_threads(20);

	#pragma omp parallel for schedule (dynamic)
	for (int iz = 0; iz < numberOfInlines; iz++ )
	{
		int threadId = omp_get_thread_num();
		printf(" numplan: %d/%d\n", iz, numberOfInlines);
		if ( layerspectrumdialog != nullptr ) layerspectrumdialog->set_progressbar_values((double)iz, (double)numberOfInlines);
		std::vector<double> dist;
		dist.resize(seeds.size());

		std::vector<SeismicType>& sliceS = ioCubesS[threadId];
		std::vector<RgtType>& sliceT = ioCubesT[threadId];
		pr->m_cubeS->readInlineBlock(sliceS.data(), iz, iz+1, false);
		pr->m_cubeT->readInlineBlock(sliceT.data(), iz, iz+1, false);

		std::vector<std::vector<char>>& attributsSlices = attributsBuffers[threadId];

		for (int attrIdx=0; attrIdx<attributsSlices.size(); attrIdx++) {
			SampleTypeBinder binder(std::get<0>(attributsDatasets[attrIdx])->sampleType());
			binder.bind<AnisotropyProcessReadKernel>(std::get<0>(attributsDatasets[attrIdx]), attributsSlices[attrIdx].data(), iz, iz+1, false);
		}

		SeismicType* rawData = sliceS.data();
		RgtType* rgtData = sliceT.data();

		for (std::size_t iy=0; iy<numberOfTraces; iy++)
		{
			for (std::size_t indexRef=0; indexRef<reference.size(); indexRef++)
			{
				int indexTrace = (reference[indexRef].iso[iy + iz*numberOfTraces] - firstSample) / sampleStep;
				reference[indexRef].rgt[iy + iz*numberOfTraces] = rgtData[(numberOfSamples*iy+indexTrace)*pr->m_cubeT->dimV()+pr->m_channelT];
			}
		}

		std::vector<float> points;
		points.resize(numberOfTraces);

		for (std::size_t iy=0; iy<numberOfTraces; iy++)
		{
			double som=0.0 ;
			int ix;
			bool seedFound = false;
			if (isConstrainSet && constrainLayer[iz*numberOfTraces+iy]!=-9999)
			{
				long ixOri = (constrainLayer[iz*numberOfTraces+iy] - firstSample) / sampleStep;
				ix = ixOri;
				if (dtauReference>0)
				{
					ix = std::max(ix, 0);
					while (ix<numberOfSamples && rgtData[(ix+iy*numberOfSamples)*pr->m_cubeT->dimV()+pr->m_channelT]<
							rgtData[(ixOri+iy*numberOfSamples)*pr->m_cubeT->dimV()+pr->m_channelT]+dtauReference)
					{
						ix++;
					}
					ix = std::min(ix, numberOfSamples-1);
				}
				else if(dtauReference<0)
				{
					ix = std::min(ix, numberOfSamples);
					long oldX = ix;
					while (ix>=0 && rgtData[(ix+iy*numberOfSamples)*pr->m_cubeT->dimV()+pr->m_channelT]>
							rgtData[(ixOri+iy*numberOfSamples)*pr->m_cubeT->dimV()+pr->m_channelT]+dtauReference)
					{
						ix--;
					}
					if (ix<numberOfSamples-1 && rgtData[(ix+iy*numberOfSamples)*pr->m_cubeT->dimV()+pr->m_channelT]<
							rgtData[(ixOri+iy*numberOfSamples)*pr->m_cubeT->dimV()+pr->m_channelT]+dtauReference)
					{
						ix++;
					}
				}
				seedFound = true;
			}
			else
			{
				for(int i=0; (i < seeds.size()) && !seedFound; i++)
				{
					long val = ((iy - seeds[i].y)*(iy-seeds[i].y) + (iz -seeds[i].z)*(iz-seeds[i].z));
					if (val!=0)
					{
						dist[i] = 1.0 / std::pow(val,distancePower) ;
						som += dist[i] ;
					}
					else
					{
						ix = seeds[i].x;
						seedFound = true;
					}
				}
			}

			float floatIx = ix;
			if(!seedFound)
			{
				ix = 0 ;
				double weightedIso = 0;
				if (isReferenceSet)
				{
					for(int i=0; i < seeds.size() ; i++)
					{
						while ( ix<numberOfSamples && getNewRgtValueFromReference(iy, iz, ix, rgtData[(iy*numberOfSamples + ix)*pr->m_cubeT->dimV()+pr->m_channelT], firstSample, sampleStep, numberOfTraces, reference, referenceValues)  < seeds[i].rgtValue )
						{
							ix ++ ;
						}
						if (ix>=numberOfSamples)
						{
							ix = numberOfSamples - 1;
						}
						weightedIso += ix*dist[i] ;
					}
				}
				else
				{
					for(int i=0; i < seeds.size() ; i++)
					{
						while ( ix<numberOfSamples && (rgtData[(iy*numberOfSamples + ix)*pr->m_cubeT->dimV()+pr->m_channelT])  < seeds[i].rgtValue )
						{
							ix ++ ;
						}
						if (ix>=numberOfSamples)
						{
							ix = numberOfSamples - 1;
						}
						double ixDouble = ix;
						if (rgtData[(iy*numberOfSamples + ix)*pr->m_cubeT->dimV()+pr->m_channelT]==seeds[i].rgtValue || ix==0) {
							ixDouble = ix;
						} else {
							double ix_floor_rgt = rgtData[(iy*numberOfSamples + ix-1)*pr->m_cubeT->dimV()+pr->m_channelT];
							double ix_rgt = rgtData[(iy*numberOfSamples + ix)*pr->m_cubeT->dimV()+pr->m_channelT];
							ixDouble = ix-1 + (seeds[i].rgtValue - ix_floor_rgt) / (ix_rgt - ix_floor_rgt);

							if (ixDouble>ix) {
								ixDouble = ix;
							} else if (ixDouble<ix-1) {
								ixDouble = ix-1;
							}

						}
						weightedIso += ixDouble*dist[i] ;
					}
				}
				floatIx = weightedIso/som;
				if (floatIx>=numberOfSamples - 1) {
					floatIx = numberOfSamples - 1;
				} else if (floatIx<0) {
					floatIx = 0;
				}
				ix = std::round(floatIx);
			}

			if (useSnap)
			{

			}
			points[iy] = floatIx;
		}
		if (useMedian)
		{

		}

		for (std::size_t iy=0; iy<numberOfTraces; iy++) {
			float floatIx = points[iy];
			if (floatIx<0) {
				floatIx = 0;
			} else if (floatIx>=numberOfSamples-1) {
				floatIx = numberOfSamples - 1;
			}
			int ix = std::round(floatIx);
			(*module)[0][iz*numberOfTraces + iy] = floatIx; //firstSample + ix * sampleStep;
			(*module)[1][iz*numberOfTraces + iy] = rawData[(iy*numberOfSamples + ix)*pr->m_cubeS->dimV()+pr->m_channelS] ;
		}

		for (std::size_t iy=0; iy<numberOfTraces; iy++)
		{
			float floatIx = points[iy];
			if (floatIx<0) {
				floatIx = 0;
			} else if (floatIx>=numberOfSamples-1) {
				floatIx = numberOfSamples - 1;
			}
			int ix = std::round(floatIx);

			int deb = std::max(ix-windowSize/2, 0);
			int fin = std::min(ix-windowSize/2 + windowSize, numberOfSamples);
			int realWindowSize = std::max(windowSize, fin - deb);

			double sumx=0, sumy=0, sumxx=0, sumyy=0, sumxy=0;
			double xbar, ybar, varx, vary, covxy, sumvars, diffvars;
			double D,T, L1,L2, v1x, v1y;
			int npix=0;

			for (int attrIdx = 0; attrIdx<attributsDatasets.count(); attrIdx++) {
				SampleTypeBinder binder(std::get<0>(attributsDatasets[attrIdx])->sampleType());

				int dimV = std::get<0>(attributsDatasets[attrIdx])->dimV();
				int typeSize = std::get<0>(attributsDatasets[attrIdx])->sampleType().byte_size();
				double azimuth = std::get<2>(attributsDatasets[attrIdx])*M_PI/180.0 ;
				double attrVal = binder.bind<GetValueKernel>(attributsSlices[attrIdx].data() + (deb+iy*numberOfSamples)*dimV*typeSize, std::get<1>(attributsDatasets[attrIdx]));
				double x = attrVal * std::cos(azimuth);
				double y = attrVal * std::sin(azimuth);

				sumx+=x;
				sumy+=y;
				sumxx+=x*x;
				sumyy+=y*y;
				sumxy+=x*y;
				npix++;
			}

			// baricenter
			xbar = ((double)sumx)/npix;
			ybar = ((double)sumy)/npix;
			// variances and covariance
			varx = sumxx/npix - xbar*xbar;
			vary = sumyy/npix - ybar*ybar;
			covxy = sumxy/npix - xbar*ybar;

			T = varx + vary; /* trace matrice covarince */
			D = varx*vary - covxy*covxy;
			// eigenvalues
			L1 = T/2 + std::sqrt(T*T/4 -D);
			L2 = T/2 - std::sqrt(T*T/4 -D);

		    if(sumxy > 10e-7) {
		        v1x = L1-vary;
		        v1y = covxy;
		    } else {
		        v1x = 1;
		        v1y = 0;
		    }

			double sqrtL1 = std::sqrt(L1);
			double sqrtL2 = std::sqrt(L2);
			(*module)[2][iy + numberOfTraces * iz] = (sqrtL2/sqrtL1) * 10000;
			(*module)[3][iy + numberOfTraces * iz] = ((std::atan2(v1y, v1x)+M_PI/2.0) * 180.0 / M_PI) * 10;
			(*module)[4][iy + numberOfTraces * iz] = (sqrtL1<32536.0f) ? sqrtL1 : 32536.0f;
			(*module)[5][iy + numberOfTraces * iz] = (sqrtL2<32536.0f) ? sqrtL2 : 32536.0f;

//					double meanVal = binder.bind<ComputeMeanKernel>(attributsSlices[attrIdx].data() + (deb+iy*numberOfSamples)*dimV*typeSize, realWindowSize,
//							dimV, attributsDatasets[attrIdx].second);

//					(*module)[attrIdx+2][iy + numberOfTraces * iz] = meanVal;

//					for (int j=0; j<windowSize; j++)
//					{
//						//double wt = pow(1 - fabs(j-windowSize/2)/(windowSize/2), hat_pow) ;
//						float rawDataVal = 0;
//						if (0 <= ix+j-windowSize/2 && ix+j-windowSize/2 < numberOfSamples)
//						{
//							rawDataVal = attributsSlices[attrIdx][(iy*numberOfSamples + ix + j-windowSize/2)*
//																  attributsDatasets[attrIdx].first->dimV()+attributsDatasets[attrIdx].second];
//						}
//
//						hostInputData[(iz*numberOfTraces + iy)*windowSize + j] = (cufftReal)rawDataVal;
//					}
		}
	}

	// gradient_multiscale_gpu_v2_run(hostInputData, windowSize, numberOfTraces, numberOfInlines, 7, 7, 5, out);
	fprintf(stderr, "%d\n", module->size());
	for (int i=0; i<module->size(); i++)
	{
		(*module)[i].resize(numberOfInlines*numberOfTraces);
		for (int j=0; j<(*module)[i].size(); j++)
		{
			// (*module)[i][j] = out[j+numberOfTraces * numberOfInlines*i];
		}
	}

	printf(" End of LayerSpectrumProcess::computeModules\n") ;
	return module;
}

AnisotropyAbstractProcess::AnisotropyAbstractProcess(Seismic3DDataset *cubeS, int channelS, Seismic3DDataset *cubeT, int channelT) :
		LayerProcess(cubeS, channelS, cubeT, channelT) {

}

void AnisotropyAbstractProcess::compute(LayerSpectrumDialog *layerspectrumdialog) {
	if (! m_isComputed && m_computeMutex.try_lock() && m_attributDatasets.count()!=0) {
		SampleTypeBinder binder(m_cubeS->sampleType());
		std::vector<std::vector<float>>* buf = binder.bind<ComputeModulesKernel>(this,
				m_constrainIso,
				m_reference,
				m_seeds,
				m_dtauReference,
				m_useSnap, m_useMedian, m_distancePower, m_attributDatasets,
				m_windozSize, layerspectrumdialog);

			{
				fprintf(stderr, "<<<<<<<<<<<<<<<<<<<<<<< %s %d\n", __FILE__, __LINE__);
				QMutexLocker lock(&m_cacheMutex);
				if (m_module!=nullptr) {
					delete m_module;
				}
				m_module = buf;
				m_nbOutputSlices = 6;
				m_isComputed = true;
				m_computeMutex.unlock();
			}

			emit LayerProcess::processCacheIsReset();
		}
}

const float* AnisotropyAbstractProcess::getModuleData(std::size_t spectrumSlice) const {
	/*
	QMutexLocker lock(&m_cacheMutex);
	if (spectrumSlice >= m_nbOutputSlices) {
		printf( "ToAnalyse4Process::getModuleData Error Slice outside range %d / % d\n", spectrumSlice,
				m_nbOutputSlices);
		spectrumSlice = 0;
	}
	return nullptr;
	*/
	QMutexLocker lock(&m_cacheMutex);
	if (!m_isComputed && m_module==nullptr) {
		return nullptr;
	}

	if (spectrumSlice >= m_nbOutputSlices) {
		printf( "ToAnalyse3Process::getModuleData Error Slice outside range %d / % d", spectrumSlice,
				m_nbOutputSlices);
		spectrumSlice =  m_nbOutputSlices-1;
	}
	const float* outTab;
	if (m_module==nullptr) {
		outTab = nullptr;
	} else {
		outTab = (*m_module)[spectrumSlice].data();// + (spectrumSlice * m_dimW * m_dimH);
		}
	return outTab;
}


