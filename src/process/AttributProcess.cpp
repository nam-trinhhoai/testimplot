/*
 * MorletProcess.cpp
 *
 *  Created on: 2 mars 2020
 *      Author: l0222891
 */


#include "AttributProcess.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>

#include <omp.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <math.h>
#include <cmath>
#include <stdio.h>

#include "RGTMorletMemory.cuh"
#include "gradient_multiscale/gradient_multiscale.cuh"

// #include <datasets/DatasetGroup.h>
// #include <DatasetCubeUtil.h>
// #include "SampleTypeBinder.h"


typedef float2 Complex;

/*
template <typename InputCubeType>
struct computeModules {
	static float* run(const ToAnalyse4Process* pr, int windowSize, int w, int shift,
			std::vector<RgtSeed> seeds, int distancePower) {

		return nullptr;
	}
};
*/

template <typename DataType>
struct ComputeMeanKernel {
	static double run(char* buffer, long N, int dimV, int channel) {
		DataType* buf = static_cast<DataType*>(static_cast<void*>(buffer));
		double sum = 0;
		for (long i=0; i<N; i++) {
			sum += buf[i*dimV+channel];
		}
		return sum / N;
	}
};

template<typename DataType>
struct AttributeProcessReadKernel {
	static void run(Seismic3DDataset* dataset, char* buffer, int zDeb, int zLimit, bool swap) {
		dataset->readInlineBlock<DataType>(static_cast<DataType*>(static_cast<void*>(buffer)), zDeb, zLimit, swap);
	}
};

template <typename RgtType>
std::vector<std::vector<float>>* ToAnalyse5Process::ComputeModulesKernel<RgtType>::run( ToAnalyse5Process* pr,
		const std::vector<float>& constrainLayer,
		std::vector<ReferenceDuo>& reference,
		const std::vector<RgtSeed> &seeds,
		long dtauReference,
		bool useSnap, bool useMedian, int distancePower,
		const QList<std::pair<Seismic3DDataset*, int>>& attributsDatasets,int windowSize,
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
std::vector<std::vector<float>>* ToAnalyse5Process::ComputeModulesKernel<RgtType>::ComputeModulesKernelLevel2<SeismicType>::run( ToAnalyse5Process* pr,
		const std::vector<float>& constrainLayer,
		std::vector<ReferenceDuo>& reference,
		std::vector<RgtSeed> seeds,
		long dtauReference,
		bool useSnap, bool useMedian, int distancePower, const QList<std::pair<Seismic3DDataset*, int>>& attributsDatasets,
		int windowSize,
		LayerSpectrumDialog *layerspectrumdialog) {

	fprintf(stderr, "ok\n");

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
	module->resize(2 + attributsDatasets.count(), initShortVec);
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
			attributsBuffers[i][j].resize(numberOfSamples*numberOfTraces* attributsDatasets[j].first->dimV() * attributsDatasets[j].first->sampleType().byte_size());
		}
	}

	//long blocSizeZ = RGTMemorySpectrum_CPUGPUMemory_getBlocSize((size_t) numberOfSamples, (size_t) numberOfTraces, (size_t) numberOfInlines, windowSize);
	//SeismicType *hostInputData = new SeismicType[ numberOfTraces * blocSizeZ * windowSize];

	omp_set_num_threads(20);
	//int geologicalTime = seeds[0].rgtValue;
//	for (int izDeb=0; izDeb<numberOfInlines; izDeb+=blocSizeZ)
//	{
//		int izMax = izDeb + blocSizeZ;
//		if (izMax>numberOfInlines)
//		{
//			izMax = numberOfInlines;
//		}

		// #pragma omp parallel for schedule (dynamic)
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
				SampleTypeBinder binder(attributsDatasets[attrIdx].first->sampleType());
				binder.bind<AttributeProcessReadKernel>(attributsDatasets[attrIdx].first, attributsSlices[attrIdx].data(), iz, iz+1, false);
			}

			SeismicType* rawData = sliceS.data();
			RgtType* rgtData = sliceT.data();

			/*for (std::size_t index=0; index<sliceS.size(); index++)
			{
				unsigned char* tab = (unsigned char*)(rawData+index);
				unsigned char tmp = tab[0];
				tab[0] = tab[1];
				tab[1] = tmp;
				tab = (unsigned char*)(rgtData+index);
				tmp = tab[0];
				tab[0] = tab[1];
				tab[1] = tmp;
			}*/

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

			for (int attrIdx=0; attrIdx<attributsDatasets.count(); attrIdx++) {
				SampleTypeBinder binder(attributsDatasets[attrIdx].first->sampleType());
				for (std::size_t iy=0; iy<numberOfTraces; iy++)
				{
					float floatIx = points[iy];
					if (floatIx<0) {
						floatIx = 0;
					} else if (floatIx>=numberOfSamples-1) {
						floatIx = numberOfSamples - 1;
					}
					int ix = std::round(floatIx);

					int dimV = attributsDatasets[attrIdx].first->dimV();
					int typeSize = attributsDatasets[attrIdx].first->sampleType().byte_size();
					int deb = std::max(ix-windowSize/2, 0);
					int fin = std::min(ix-windowSize/2 + windowSize, numberOfSamples);
					int realWindowSize = std::max(windowSize, fin - deb);
					double meanVal = binder.bind<ComputeMeanKernel>(attributsSlices[attrIdx].data() + (deb+iy*numberOfSamples)*dimV*typeSize, realWindowSize,
							dimV, attributsDatasets[attrIdx].second);
					(*module)[attrIdx+2][iy + numberOfTraces * iz] = meanVal;

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
//	}
//	delete hostInputData;
	printf(" End of LayerSpectrumProcess::computeModules\n") ;
	return module;
}

ToAnalyse5Process::ToAnalyse5Process(Seismic3DDataset *cubeS, int channelS, Seismic3DDataset *cubeT, int channelT) :
		LayerProcess(cubeS, channelS, cubeT, channelT) {

}

void ToAnalyse5Process::compute(LayerSpectrumDialog *layerspectrumdialog) {
	if (! m_isComputed && m_computeMutex.try_lock()) {
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
				m_nbOutputSlices = 2 + m_attributDatasets.count();
				m_isComputed = true;
				m_computeMutex.unlock();
			}

			emit LayerProcess::processCacheIsReset();
		}
}

const float* ToAnalyse5Process::getModuleData(std::size_t spectrumSlice) const {
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
		spectrumSlice =  m_nbOutputSlices/2;
	}
	const float* outTab;
	if (m_module==nullptr) {
		outTab = nullptr;
	} else {
		outTab = (*m_module)[spectrumSlice].data();// + (spectrumSlice * m_dimW * m_dimH);
		}
	return outTab;
}


