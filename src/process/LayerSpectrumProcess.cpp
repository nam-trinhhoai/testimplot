/*
 * LayerSpectrumProcess.cpp
 *
 *  Created on: 2 mars 2020
 *      Author: l0222891
 */


#include "LayerSpectrumProcess.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>

#include <vector>
#include <list>
#include <memory>
#include <unordered_map>
#include <math.h>
#include <cmath>
#include <omp.h>
#include <slicer/data/sliceutils.h>
#include "slicer/data/affine2dtransformation.h"
#include "process/RgtLayerProcessUtil.h"
#include "affinetransformation.h"
#include "cudaimagepaletteholder.h"

#include "RGT_Spectrum_Memory.cuh"

//#include <DatasetCubeUtil.h>
//#include "SampleTypeBinder.h"
#include "palette/imageformats.h"
#include "RgtLayerProcessUtil.h"
#include "cwt_file.h"
#include "ioutil.h"
#include "sampletypebinder.h"

std::vector<std::vector<float>>* ToAnalyse2Process::computeModulesCwt( ToAnalyse2Process* pr, int windowSize,  std::vector<RgtSeed> seeds, float hat_pow, bool polarity,
			bool useSnap, bool useMedian, int lwx, int distancePower, int snapWindow, const std::vector<float>& constrainLayer,
			long dtauReference, std::string cwtSeismicPath, std::string cwtRgtPath, std::vector<ReferenceDuo>& reference,
			LayerSpectrumDialog *layerspectrumdialog) {
//		QMutexLocker lock(&spectrumProcess->m_mutex);

		int type = polarity ? 1 : -1;

//		const io::InputOutputCube<InputCubeType>* iocubeS =
//				dynamic_cast<const io::InputOutputCube<InputCubeType>*>(spectrumProcess->getCubeS());
//		io::CubeDimension dims = spectrumProcess->getCubeS()->getDim();
//		CubeSeismicAddon seismicAddon = spectrumProcess->getCubeS()->getSeismicAddon();
//		const io::InputOutputCube<float>* iocubeT =
//				dynamic_cast<const io::InputOutputCube<float>*>(spectrumProcess->getCubeT());
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
		module->resize(2 + windowSize/2, initShortVec);
		for (std::size_t attrIndex = 0; attrIndex<module->size(); attrIndex++) {
			(*module)[attrIndex].resize(sizeLayer, 0);
		}
//		memset(module, 0, sizeLayer * (2+ windowSize/2)*sizeof(float));

		bool isConstrainSet = constrainLayer.size()==sizeLayer;

		std::vector<std::vector<float>> ioCubesS, ioCubesT;
		std::vector<float> init;
		ioCubesS.resize(20, init);
		ioCubesT.resize(20, init);

		std::list<CWT_FILE> filesS, filesT;
		for (int i=0; i<20; i++) {
			CWT_FILE initFile, initFile2;
			filesS.push_back(std::move(initFile));

			filesT.push_back(std::move(initFile2));
		}

		for (int i=0; i<ioCubesS.size(); i++) {
			ioCubesS[i].resize(numberOfSamples*numberOfTraces);
			ioCubesT[i].resize(numberOfSamples*numberOfTraces);

			// init cwt files
			std::list<CWT_FILE>::iterator itS = filesS.begin();
			std::list<CWT_FILE>::iterator itT = filesT.begin();
			std::advance(itS, i);
			std::advance(itT, i);

			itS->openForRead(cwtSeismicPath.c_str(), CWT_FILE::FLOAT);
			itT->openForRead(cwtRgtPath.c_str(), CWT_FILE::FLOAT);
		}

        bool isReferenceSet = false;//reference.size()!=0;
        std::vector<int> referenceValues;

        if (isReferenceSet) {
        	std::list<CWT_FILE>::iterator itT = filesT.begin();
			referenceValues.resize(reference.size());
			std::vector<float> trace;
			trace.resize(numberOfSamples);
//			ioCubesT[0]->readSubVolume(0, dims.getJ()/2, dims.getK()/2, dims.getI(), 1, 1, trace.data());
			itT->inlineRead(numberOfInlines/2, ioCubesT[0].data());
			memcpy(trace.data(), ioCubesT[0].data()+numberOfSamples*(numberOfTraces/2), numberOfSamples*sizeof(float));

			for (std::size_t index_ref=0; index_ref<reference.size(); index_ref++) {
				int pos_tau = (reference[index_ref].iso[numberOfTraces/2 + numberOfTraces * (numberOfInlines/2)] - firstSample) / sampleStep;
				referenceValues[index_ref] = trace[pos_tau];
			}

			std::vector<float> sectionBuffer;
			sectionBuffer.resize(sizeInline);
			for (long index=0; index<seeds.size(); index++) {
				RgtSeed& seed = seeds[index];
				// read and apply
				bool isReadNeeded = false;
				for (std::size_t index_ref=0; index_ref<reference.size() && !isReadNeeded; index_ref++) {
					isReadNeeded = reference[index_ref].rgt[seed.y+seed.z*numberOfTraces] == -9999.0;
				}
				if (isReadNeeded) {
//					ioCubesT[0]->readSubVolume(0, 0, seeds[index].z, dims.getI(), dims.getJ(), 1, sectionBuffer.data());
					itT->inlineRead(seeds[index].z, sectionBuffer.data());
					for (std::size_t iy=0; iy<numberOfTraces; iy++) {
						for (std::size_t indexRef=0; indexRef<reference.size(); indexRef++) {
							int indexTrace = (reference[indexRef].iso[iy + seed.z*numberOfTraces] - firstSample) / sampleStep;
							reference[indexRef].rgt[iy + seed.z*numberOfTraces] = sectionBuffer[numberOfSamples*iy+indexTrace];
						}
					}
				}

				seed.rgtValue = getNewRgtValueFromReference(seed.y, seed.z, seed.x, seed.rgtValue, firstSample, sampleStep, numberOfTraces, reference, referenceValues);
			}
        }


		// sort seed vector
		std::sort(seeds.begin(), seeds.end(), [](RgtSeed a, RgtSeed b){
			return a.rgtValue<b.rgtValue;
		});

		// long blocSizeZ = RGTMemorySpectrum_getBlocSize((size_t) numberOfSamples, (size_t) numberOfTraces, (size_t) numberOfInlines, windowSize);
		long blocSizeZ = RGTMemorySpectrum_CPUGPUMemory_getBlocSize((size_t) numberOfSamples, (size_t) numberOfTraces, (size_t) numberOfInlines, windowSize);

		cufftReal *hostInputData = new cufftReal[ numberOfTraces * blocSizeZ * windowSize];
//		memset(hostInputData, 0, sizeLayer * windowSize*sizeof(cufftReal));

//		qDebug() << "Process hostInputData init";

		omp_set_num_threads(20);
		for (int izDeb=0; izDeb<numberOfInlines; izDeb+=blocSizeZ) {
			int izMax = izDeb + blocSizeZ;
			if (izMax>numberOfInlines) {
				izMax = numberOfInlines;
			}

			#pragma omp parallel for schedule (dynamic)
			for (int iz = izDeb; iz < izMax; iz++ ) {
				int threadId = omp_get_thread_num();
				printf(" numplan:: %d/%d\n", iz, numberOfInlines) ;
				if ( layerspectrumdialog != nullptr )
				{
					layerspectrumdialog->set_progressbar_values((double)iz, (double)(numberOfInlines-1));
				}
				else
				{
					fprintf(stderr, "null: %s %d\n", __FILE__, __LINE__);
				}

				std::vector<double> dist;
				dist.resize(seeds.size());

				std::vector<float>& sliceS = ioCubesS[threadId];
				std::vector<float>& sliceT = ioCubesT[threadId];

				std::list<CWT_FILE>::iterator itS = filesS.begin();
				std::list<CWT_FILE>::iterator itT = filesT.begin();
				std::advance(itS, threadId);
				std::advance(itT, threadId);
				itS->inlineRead(iz, sliceS.data());
				itT->inlineRead(iz, sliceT.data());
				//pr->m_cubeS->readInlineBlock(sliceS.data(), iz, iz+1);
				//pr->m_cubeT->readInlineBlock(sliceT.data(), iz, iz+1);

				float* rawData = sliceS.data();
				float* rgtData = sliceT.data();

				if (isReferenceSet) {
					for (std::size_t iy=0; iy<numberOfTraces; iy++) {
						for (std::size_t indexRef=0; indexRef<reference.size(); indexRef++) {
							int indexTrace = (reference[indexRef].iso[iy + iz*numberOfTraces] - firstSample) / sampleStep;
							reference[indexRef].rgt[iy + iz*numberOfTraces] = rgtData[numberOfSamples*iy+indexTrace];
						}
					}
				}

				// swap
				/*for (std::size_t index=0; index<sliceS.size(); index++) {
					unsigned char* tab = (unsigned char*)(rawData+index);
					unsigned char tmp = tab[0];
					tab[0] = tab[1];
					tab[1] = tmp;

					tab = (unsigned char*)(rgtData+index);
					tmp = tab[0];
					tab[0] = tab[1];
					tab[1] = tmp;
				}*/

				std::vector<float> points;
				points.resize(numberOfTraces);

				for (std::size_t iy=0; iy<numberOfTraces; iy++) {
					double som=0.0 ;

					int ix;
					bool seedFound = false;
					if (isConstrainSet && constrainLayer[iz*numberOfTraces+iy]!=-9999) {
						long ixOri = (constrainLayer[iz*numberOfTraces+iy] - firstSample) / sampleStep;
						ix = ixOri;
						if (dtauReference>0) {
							ix = std::max(ix, 0);
							while (ix<numberOfSamples && rgtData[ix+iy*numberOfSamples]<rgtData[ixOri+iy*numberOfSamples]+
									dtauReference) {
								ix++;
							}
							ix = std::min(ix, numberOfSamples-1);
						} else if(dtauReference<0) {
							ix = std::min(ix, numberOfSamples);
							long oldX = ix;
							while (ix>=0 && rgtData[ix+iy*numberOfSamples]>rgtData[ixOri+iy*numberOfSamples]+dtauReference) {
								ix--;
							}
							if (ix<numberOfSamples-1 && rgtData[ix+iy*numberOfSamples]<rgtData[ixOri+iy*numberOfSamples]+dtauReference) {
								ix++;
							}

						}
						//ix = (constrainLayer[iz*numberOfTraces+iy] - firstSample) / sampleStep;
						seedFound = true;
					} else {
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
					}

					float floatIx = ix;
					if(!seedFound) {
						ix = 0 ;
						double weightedIso = 0;
						if (isReferenceSet) {
							for(int i=0; i < seeds.size() ; i++) {
								while ( ix<numberOfSamples && getNewRgtValueFromReference(iy, iz, ix, rgtData[iy*numberOfSamples + ix], firstSample, sampleStep, numberOfTraces, reference, referenceValues)  < seeds[i].rgtValue ) {
									ix ++ ;
								}
								if (ix>=numberOfSamples) {
									ix = numberOfSamples - 1;
								}
								weightedIso += ix*dist[i] ;
							}
						} else {
							for(int i=0; i < seeds.size() ; i++) {
								while ( ix<numberOfSamples && (rgtData[iy*numberOfSamples + ix])  < seeds[i].rgtValue ) {
									ix ++ ;
								}
								if (ix>=numberOfSamples) {
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
					if (useSnap) {
	//TODO					int newx = process::bl_indpol(ix, rawData+iy*height, height, type, snapWindow);
	//TODO					ix = (newx==process::SLOPE::RAIDE)? ix : newx;
					}
					points[iy] = floatIx;
				}

				if (useMedian) {
	//TODO				process::UtFiltreMedianeX(points.data(), points.size(), 1, lwx);
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
					(*module)[1][iz*numberOfTraces + iy] = rawData[iy*numberOfSamples + ix] ;

					double A = 0.54, B = 0.46;
					double omega = 2.0 * M_PI / (windowSize - 1);
					for (int j=0; j<windowSize; j++) {
						double wt = pow(1 - fabs(j-windowSize/2)/(windowSize/2), hat_pow) ;
						double hammingCoef = A -B * cos(omega * j);
						float rawDataVal = 0;
						if (0 <= ix+j-windowSize/2 && ix+j-windowSize/2 < numberOfSamples) {
							rawDataVal = rawData[iy*numberOfSamples + ix + j-windowSize/2];
						}

						hostInputData[((iz-izDeb)*numberOfTraces + iy)*windowSize + j] =
								(cufftReal)( hammingCoef * wt * rawDataVal);
					}
				}
			}

			std:vector<short*> pseudoModule;
			pseudoModule.resize(module->size());
			for (std::size_t attrIndex=0; attrIndex<pseudoModule.size(); attrIndex++) {
				pseudoModule[attrIndex] = new short[numberOfTraces * (izMax - izDeb)]();//(*module)[attrIndex].data() + izDeb*numberOfTraces;
			}
			int result =  RGTMemorySpectrumBis ( hostInputData, pseudoModule.data(),
					(size_t) numberOfSamples, (size_t) numberOfTraces, (size_t) izMax - izDeb,
					windowSize);

			for (std::size_t attrIndex=0; attrIndex<pseudoModule.size(); attrIndex++) {
				for (std::size_t copyIdx=0; copyIdx<numberOfTraces * (izMax - izDeb); copyIdx++) {
					(*module)[attrIndex][izDeb*numberOfTraces+copyIdx] = pseudoModule[attrIndex][copyIdx];
				}
				delete[] pseudoModule[attrIndex];
			}
		}

		for (int i=0; i<ioCubesS.size(); i++) {
			//TODO delete ioCubesS[i];
			//TODO delete ioCubesT[i];
		}

		delete[] hostInputData;

		printf(" End of LayerSpectrumProcess::computeModules\n") ;
		return module;
}

template <typename RgtType>
std::vector<std::vector<float>>* ToAnalyse2Process::ComputeModulesDefaultKernel<RgtType>::run(
		ToAnalyse2Process* pr, int windowSize, const std::vector<RgtSeed>& seeds, float hat_pow, bool polarity,
		bool useSnap, bool useMedian, int lwx, int distancePower, int snapWindow, const std::vector<float>& constrainLayer,
		long dtauReference, std::vector<ReferenceDuo>& reference,
		LayerSpectrumDialog *layerspectrumdialog) {
	SampleTypeBinder binder(pr->m_cubeS->sampleType());
	return binder.bind<ComputeModulesDefaultKernelLevel2>(pr, windowSize, seeds, hat_pow, polarity,
			useSnap, useMedian, lwx, distancePower, snapWindow, constrainLayer,
			dtauReference, reference, layerspectrumdialog);
}

template <typename RgtType>
template <typename SeismicType>
std::vector<std::vector<float>>* ToAnalyse2Process::ComputeModulesDefaultKernel<RgtType>::ComputeModulesDefaultKernelLevel2<SeismicType>::run(
		ToAnalyse2Process* pr, int windowSize,  std::vector<RgtSeed> seeds, float hat_pow, bool polarity,
			bool useSnap, bool useMedian, int lwx, int distancePower, int snapWindow, const std::vector<float>& constrainLayer,
			long dtauReference, std::vector<ReferenceDuo>& reference,
			LayerSpectrumDialog *layerspectrumdialog) {
//		QMutexLocker lock(&spectrumProcess->m_mutex);

		int type = polarity ? 1 : -1;

//		const io::InputOutputCube<InputCubeType>* iocubeS =
//				dynamic_cast<const io::InputOutputCube<InputCubeType>*>(spectrumProcess->getCubeS());
//		io::CubeDimension dims = spectrumProcess->getCubeS()->getDim();
//		CubeSeismicAddon seismicAddon = spectrumProcess->getCubeS()->getSeismicAddon();
//		const io::InputOutputCube<float>* iocubeT =
//				dynamic_cast<const io::InputOutputCube<float>*>(spectrumProcess->getCubeT());
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
		module->resize(2 + windowSize/2, initShortVec);
		for (std::size_t attrIndex = 0; attrIndex<module->size(); attrIndex++) {
			(*module)[attrIndex].resize(sizeLayer, 0);
		}
//		module = new float[ sizeLayer * (2 + windowSize/2) ];
//		memset(module, 0, sizeLayer * (2+ windowSize/2)*sizeof(float));

		bool isConstrainSet = constrainLayer.size()==sizeLayer;

		bool isReferenceSet = false;//reference.size()!=0;
		std::vector<int> referenceValues;

		if (isReferenceSet) {
			referenceValues.resize(reference.size());
			std::vector<RgtType> trace;
			trace.resize(numberOfSamples*pr->m_cubeT->dimV());
//			ioCubesT[0]->readSubVolume(0, dims.getJ()/2, dims.getK()/2, dims.getI(), 1, 1, trace.data());
			pr->m_cubeT->readSubTraceAndSwap(trace.data(), 0, numberOfSamples, numberOfTraces/2, numberOfInlines/2);

			for (std::size_t index_ref=0; index_ref<reference.size(); index_ref++) {
				int pos_tau = (reference[index_ref].iso[numberOfTraces/2 + numberOfTraces * (numberOfInlines/2)] - firstSample) / sampleStep;
				referenceValues[index_ref] = trace[pos_tau * pr->m_cubeT->dimV() + pr->m_channelT];
			}

			std::vector<RgtType> sectionBuffer;
			sectionBuffer.resize(sizeInline*pr->m_cubeT->dimV());
			for (long index=0; index<seeds.size(); index++) {
				RgtSeed& seed = seeds[index];
				// read and apply
				bool isReadNeeded = false;
				for (std::size_t index_ref=0; index_ref<reference.size() && !isReadNeeded; index_ref++) {
					isReadNeeded = reference[index_ref].rgt[seed.y+seed.z*numberOfTraces] == -9999.0;
				}
				if (isReadNeeded) {
//					ioCubesT[0]->readSubVolume(0, 0, seeds[index].z, dims.getI(), dims.getJ(), 1, sectionBuffer.data());
					pr->m_cubeT->readTraceBlockAndSwap(sectionBuffer.data(), 0, numberOfTraces, seeds[index].z);
					for (std::size_t iy=0; iy<numberOfTraces; iy++) {
						for (std::size_t indexRef=0; indexRef<reference.size(); indexRef++) {
							int indexTrace = (reference[indexRef].iso[iy + seed.z*numberOfTraces] - firstSample) / sampleStep;
							reference[indexRef].rgt[iy + seed.z*numberOfTraces] = sectionBuffer[(numberOfSamples*iy+indexTrace) * pr->m_cubeT->dimV() + pr->m_channelT];
						}
					}
				}

				seed.rgtValue = getNewRgtValueFromReference(seed.y, seed.z, seed.x, seed.rgtValue, firstSample, sampleStep, numberOfTraces, reference, referenceValues);
			}
		}

		// sort seed vector
		std::sort(seeds.begin(), seeds.end(), [](RgtSeed a, RgtSeed b){
			return a.rgtValue<b.rgtValue;
		});

		std::vector<std::vector<SeismicType>> ioCubesS;
		std::vector<std::vector<RgtType>> ioCubesT;
		std::vector<SeismicType> initS;
		std::vector<RgtType> initT;
		ioCubesS.resize(20, initS);
		ioCubesT.resize(20, initT);

		for (int i=0; i<ioCubesS.size(); i++) {
			ioCubesS[i].resize(numberOfSamples*numberOfTraces*pr->m_cubeS->dimV());
			ioCubesT[i].resize(numberOfSamples*numberOfTraces*pr->m_cubeT->dimV());
		}

		// long blocSizeZ = RGTMemorySpectrum_getBlocSize((size_t) numberOfSamples, (size_t) numberOfTraces, (size_t) numberOfInlines, windowSize);
		long blocSizeZ = RGTMemorySpectrum_CPUGPUMemory_getBlocSize((size_t) numberOfSamples, (size_t) numberOfTraces, (size_t) numberOfInlines, windowSize);

		cufftReal *hostInputData = new cufftReal[ numberOfTraces * blocSizeZ * windowSize];

		omp_set_num_threads(20);
		for (int izDeb=0; izDeb<numberOfInlines; izDeb+=blocSizeZ) {
			int izMax = izDeb + blocSizeZ;
			if (izMax>numberOfInlines) {
				izMax = numberOfInlines;
			}

			#pragma omp parallel for schedule (dynamic)
			for (int iz = izDeb; iz < izMax; iz++ ) {
				int threadId = omp_get_thread_num();
				printf(" numplan: %d/%d\n", iz, numberOfInlines);
				if ( layerspectrumdialog != nullptr ) layerspectrumdialog->set_progressbar_values((double)iz, (double)numberOfInlines);

				std::vector<double> dist;
				dist.resize(seeds.size());

				std::vector<SeismicType>& sliceS = ioCubesS[threadId];
				std::vector<RgtType>& sliceT = ioCubesT[threadId];
				pr->m_cubeS->readInlineBlock(sliceS.data(), iz, iz+1, false);
				pr->m_cubeT->readInlineBlock(sliceT.data(), iz, iz+1, false);

				SeismicType* rawData = sliceS.data();
				RgtType* rgtData = sliceT.data();

				// swap done in the reader
				/*for (std::size_t index=0; index<sliceS.size(); index++) {
					unsigned char* tab = (unsigned char*)(rawData+index);
					unsigned char tmp = tab[0];
					tab[0] = tab[1];
					tab[1] = tmp;

					tab = (unsigned char*)(rgtData+index);
					tmp = tab[0];
					tab[0] = tab[1];
					tab[1] = tmp;
				}*/

				if (isReferenceSet) {
					for (std::size_t iy=0; iy<numberOfTraces; iy++) {
						for (std::size_t indexRef=0; indexRef<reference.size(); indexRef++) {
							int indexTrace = (reference[indexRef].iso[iy + iz*numberOfTraces] - firstSample) / sampleStep;
							reference[indexRef].rgt[iy + iz*numberOfTraces] = rgtData[(numberOfSamples*iy+indexTrace) *pr->m_cubeT->dimV() + pr->m_channelT];
						}
					}
				}


				std::vector<float> points;
				points.resize(numberOfTraces);

				for (std::size_t iy=0; iy<numberOfTraces; iy++) {
					double som=0.0 ;

					int ix;
					bool seedFound = false;
					if (isConstrainSet && constrainLayer[iz*numberOfTraces+iy]!=-9999) {
						long ixOri = (constrainLayer[iz*numberOfTraces+iy] - firstSample) / sampleStep;
						ix = ixOri;
						if (dtauReference>0) {
							ix = std::max(ix, 0);
							while (ix<numberOfSamples && rgtData[(ix+iy*numberOfSamples)*pr->m_cubeT->dimV()+pr->m_channelT]<
									rgtData[(ixOri+iy*numberOfSamples)*pr->m_cubeT->dimV()+pr->m_channelT]+dtauReference) {
								ix++;
							}
							ix = std::min(ix, numberOfSamples-1);
						} else if(dtauReference<0) {
							ix = std::min(ix, numberOfSamples);
							long oldX = ix;
							while (ix>=0 && rgtData[(ix+iy*numberOfSamples)*pr->m_cubeT->dimV()+pr->m_channelT]>
									rgtData[(ixOri+iy*numberOfSamples)*pr->m_cubeT->dimV()+pr->m_channelT]+dtauReference) {
								ix--;
							}
							if (ix<numberOfSamples-1 && rgtData[(ix+iy*numberOfSamples)*pr->m_cubeT->dimV()+pr->m_channelT]<
									rgtData[(ixOri+iy*numberOfSamples)*pr->m_cubeT->dimV()+pr->m_channelT]+dtauReference) {
								ix++;
							}

						}
						//ix = (constrainLayer[iz*numberOfTraces+iy] - firstSample) / sampleStep;
						seedFound = true;
					} else {
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
					}

					float floatIx = ix;
					if(!seedFound) {
						ix = 0 ;
						double weightedIso = 0;
						if (isReferenceSet) {
							for(int i=0; i < seeds.size() ; i++) {
								while ( ix<numberOfSamples && getNewRgtValueFromReference(iy, iz, ix, rgtData[(iy*numberOfSamples + ix)*pr->m_cubeT->dimV()+pr->m_channelT],
										firstSample, sampleStep, numberOfTraces, reference, referenceValues)  < seeds[i].rgtValue ) {
									ix ++ ;
								}
								if (ix>=numberOfSamples) {
									ix = numberOfSamples - 1;
								}
								weightedIso += ix*dist[i] ;
							}
						} else {
							for(int i=0; i < seeds.size() ; i++) {
								while ( ix<numberOfSamples && (rgtData[(iy*numberOfSamples + ix)*pr->m_cubeT->dimV()+pr->m_channelT])  < seeds[i].rgtValue ) {
									ix ++ ;
								}
								if (ix>=numberOfSamples) {
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
						if (som==0.0) {
							som = 1;
						}
						floatIx = weightedIso/som;
						if (floatIx>=numberOfSamples - 1) {
							floatIx = numberOfSamples - 1;
						} else if (floatIx<0) {
							floatIx = 0;
						}
						ix = std::round(floatIx);
					}
					if (useSnap) {
							// care about rawData structure (type + dimV)
	//TODO					int newx = process::bl_indpol(ix, rawData+iy*height, height, type, snapWindow);
	//TODO					ix = (newx==process::SLOPE::RAIDE)? ix : newx;
					}
					points[iy] = floatIx;
				}

				if (useMedian) {
	//TODO				process::UtFiltreMedianeX(points.data(), points.size(), 1, lwx);
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

					double A = 0.54, B = 0.46;
					double omega = 2.0 * M_PI / (windowSize - 1);
					for (int j=0; j<windowSize; j++) {
						double wt = pow(1 - fabs(j-windowSize/2)/(windowSize/2), hat_pow) ;
						double hammingCoef = A -B * cos(omega * j);
						float rawDataVal = 0;
						if (0 <= ix+j-windowSize/2 && ix+j-windowSize/2 < numberOfSamples) {
							rawDataVal = rawData[(iy*numberOfSamples + ix + j-windowSize/2)*pr->m_cubeS->dimV()+pr->m_channelS];
						}

						hostInputData[((iz-izDeb)*numberOfTraces + iy)*windowSize + j] =
								(cufftReal)( hammingCoef * wt * rawDataVal);
					}
				}
			}

			std:vector<short*> pseudoModule;
			pseudoModule.resize(module->size());
			for (std::size_t attrIndex=0; attrIndex<pseudoModule.size(); attrIndex++) {
				pseudoModule[attrIndex] = new short[numberOfTraces * (izMax - izDeb)]();//(*module)[attrIndex].data() + izDeb*numberOfTraces;
			}

			int result =  RGTMemorySpectrumBis ( hostInputData, pseudoModule.data(),
					(size_t) numberOfSamples, (size_t) numberOfTraces, (size_t) izMax - izDeb,
					windowSize);
			for (std::size_t attrIndex=0; attrIndex<pseudoModule.size(); attrIndex++) {
				if (attrIndex>1) {
					for (std::size_t copyIdx=0; copyIdx<numberOfTraces * (izMax - izDeb); copyIdx++) {
						(*module)[attrIndex][izDeb*numberOfTraces+copyIdx] = pseudoModule[attrIndex][copyIdx];
					}
				}
				delete[] pseudoModule[attrIndex];
			}
		}

		for (int i=0; i<ioCubesS.size(); i++) {
			//TODO delete ioCubesS[i];
			//TODO delete ioCubesT[i];
		}

		delete[] hostInputData;

		printf(" End of LayerSpectrumProcess::computeModules\n") ;
		return module;
}

std::vector<std::vector<float>>* ToAnalyse2Process::computeModules(ToAnalyse2Process* pr, int windowSize,  std::vector<RgtSeed> seeds,
		float hat_pow, bool polarity, bool useSnap, bool useMedian, int lwx, int distancePower, int snapWindow,
		const std::vector<float>& constrainLayer, long dtauReference,
		std::vector<ReferenceDuo>& reference, LayerSpectrumDialog *layerspectrumdialog) {
	bool isCwtValid = false;
	std::string cwtSeismicPath, cwtRgtPath;
	cwtSeismicPath = m_cubeS->path(); //remove_extension(m_cubeS->path()) + ".cwt";
	cwtRgtPath = m_cubeT->path(); //remove_extension(m_cubeT->path()) + ".cwt";
	isCwtValid = cwtRgtPath.substr(cwtRgtPath.find_last_of(".") + 1)=="cwt" &&
			cwtSeismicPath.substr(cwtSeismicPath.find_last_of(".") + 1)=="cwt";

	if (isCwtValid && m_cubeS->dimV()==1 && m_cubeT->dimV()==1) { // because cwt is defined for dimV=1, nothing specified if dimV>1
		return computeModulesCwt(pr, windowSize, seeds, hat_pow, polarity, useSnap, useMedian, lwx,
				distancePower, snapWindow, constrainLayer, dtauReference, cwtSeismicPath, cwtRgtPath, reference,
				layerspectrumdialog);
	} else {
		SampleTypeBinder binder(m_cubeT->sampleType());
		return binder.bind<ComputeModulesDefaultKernel>(pr, windowSize, seeds, hat_pow, polarity, useSnap, useMedian, lwx,
				distancePower, snapWindow, constrainLayer, dtauReference, reference, layerspectrumdialog);
	}
}

ToAnalyse2Process::ToAnalyse2Process(
		Seismic3DDataset* cubeS, int channelS,
		Seismic3DDataset* cubeT, int channelT) :
	LayerProcess(cubeS, channelS, cubeT, channelT) {
	m_module = nullptr;
}

// JDTODO
void ToAnalyse2Process::compute(LayerSpectrumDialog *layerspectrumdialog) {
	fprintf(stderr, "<<<<<<<<<<<<<<<<<<<<<<< %s %d\n", __FILE__, __LINE__);
	if (! m_isComputed && m_computeMutex.try_lock()) {
		std::vector<std::vector<float>>* buf = computeModules(this, m_windozSize, m_seeds, m_hatPower, m_polarity, m_useSnap,
				m_useMedian, m_lwx_medianFilter, m_distancePower, m_snapWindow, m_constrainIso, m_dtauReference,
				m_reference, layerspectrumdialog);

		{
			fprintf(stderr, "<<<<<<<<<<<<<<<<<<<<<<< %s %d\n", __FILE__, __LINE__);
			QMutexLocker lock(&m_cacheMutex);
			if (m_module!=nullptr) {
				delete m_module;
			}
			m_module = buf;
			m_nbOutputSlices = 2 + m_windozSize / 2;
			m_isComputed = true;
			m_computeMutex.unlock();
		}

		emit LayerProcess::processCacheIsReset();
	}
}

const float* ToAnalyse2Process::getModuleData(std::size_t spectrumSlice) const {
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
