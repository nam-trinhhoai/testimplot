/*
 * MorletProcess.cpp
 *
 *  Created on: 2 mars 2020
 *      Author: l0222891
 */


#include "GradientMultiScaleProcess.h"

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
#include <ihm.h>
#include "gradient_multiscale/gradient_multiscale.cuh"

// #include <datasets/DatasetGroup.h>
// #include <DatasetCubeUtil.h>
// #include "SampleTypeBinder.h"


typedef float2 Complex;

/*
template <typename InputCubeType>
struct computeModules {
	static short* run(const ToAnalyse4Process* pr, int windowSize, int w, int shift,
			std::vector<RgtSeed> seeds, int distancePower) {

		return nullptr;
	}
};
*/

template<typename RgtType>
std::vector<std::vector<float>>* ToAnalyse4Process::ComputeModulesKernel<RgtType>::run( ToAnalyse4Process* pr,
		const std::vector<float>& constrainLayer,
		std::vector<ReferenceDuo>& reference,
		std::vector<RgtSeed> seeds,
		long dtauReference,
		bool useSnap, bool useMedian, int distancePower,
		float hat_pow,
		int type,
		int gcc_offset, int w, int shift,
		LayerSpectrumDialog *layerspectrumdialog) {
	SampleTypeBinder binder(pr->m_cubeS->sampleType());
	return binder.bind<ComputeModulesKernelLevel2>(pr, constrainLayer, reference, seeds, dtauReference, useSnap, useMedian, distancePower, hat_pow,
			type, gcc_offset, w, shift, layerspectrumdialog);
}

template<typename RgtType>
template<typename SeismicType>
std::vector<std::vector<float>>* ToAnalyse4Process::ComputeModulesKernel<RgtType>::ComputeModulesKernelLevel2<SeismicType>::run( ToAnalyse4Process* pr,
		const std::vector<float>& constrainLayer,
		std::vector<ReferenceDuo>& reference,
		std::vector<RgtSeed> seeds,
		long dtauReference,
		bool useSnap, bool useMedian, int distancePower,
		float hat_pow,
		int type,
		int gcc_offset, int w, int shift,
		LayerSpectrumDialog *layerspectrumdialog) {

	/*
	int nbKernel = windowSize;
	int nbOutputSlices = 2 + nbKernel;
	module = new short[ dimy * dims.getK() * (nbOutputSlices) ];
	memset(module, 0, dimy* dims.getK() * (nbOutputSlices)*sizeof(float));
	*/
/*
	int windowSize = 7;

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


	return module;
	*/




	//		QMutexLocker lock(&spectrumProcess->m_mutex);

			// int type = polarity ? 1 : -1;

	//		const io::InputOutputCube<InputCubeType>* iocubeS =
	//				dynamic_cast<const io::InputOutputCube<InputCubeType>*>(spectrumProcess->getCubeS());
	//		io::CubeDimension dims = spectrumProcess->getCubeS()->getDim();
	//		CubeSeismicAddon seismicAddon = spectrumProcess->getCubeS()->getSeismicAddon();
	//		const io::InputOutputCube<short>* iocubeT =
	//				dynamic_cast<const io::InputOutputCube<short>*>(spectrumProcess->getCubeT());
//			int windowSize = 7;

			long window_size = (w + shift) * 2 + 1;

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
			module->resize(2 + gcc_offset*2, initShortVec);
			for (std::size_t attrIndex = 0; attrIndex<module->size(); attrIndex++) {
				(*module)[attrIndex].resize(sizeLayer, 0);
			}

	//		module = new short[ sizeLayer * (2 + windowSize/2) ];
	//		memset(module, 0, sizeLayer * (2+ windowSize/2)*sizeof(short));

			bool isConstrainSet = constrainLayer.size()==sizeLayer;

			bool isReferenceSet = reference.size()!=0;
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
								reference[indexRef].rgt[iy + seed.z*numberOfTraces] = sectionBuffer[(numberOfSamples*iy+indexTrace)* pr->m_cubeT->dimV() + pr->m_channelT];
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

			std::vector<std::vector<RgtType>> ioCubesT;
			std::vector<std::vector<SeismicType>> ioCubesS;
			std::vector<RgtType> initT;
			std::vector<SeismicType> initS;
			ioCubesS.resize(20, initS);
			ioCubesT.resize(20, initT);

			for (int i=0; i<ioCubesS.size(); i++) {
				ioCubesS[i].resize(numberOfSamples*numberOfTraces*pr->m_cubeS->dimV());
				ioCubesT[i].resize(numberOfSamples*numberOfTraces*pr->m_cubeT->dimV());
			}

			// long blocSizeZ = RGTMemorySpectrum_getBlocSize((size_t) numberOfSamples, (size_t) numberOfTraces, (size_t) numberOfInlines, windowSize);
			long blocSizeZ = RGTMemorySpectrum_CPUGPUMemory_getBlocSize((size_t) numberOfSamples, (size_t) numberOfTraces, (size_t) numberOfInlines, window_size);

			// cufftReal *hostInputData = new cufftReal[ numberOfTraces * blocSizeZ * windowSize];
			short *hostInputData = new short[ numberOfTraces * blocSizeZ * window_size];

			long blocSizeZForLoop = blocSizeZ - (window_size);// because window_size is always even
			if (blocSizeZForLoop<=0) {
				blocSizeZForLoop = 1;
			}

			float *gccMean = (float*)calloc(gcc_offset, sizeof(float));
			bool gccComputeMean = true;
			omp_set_num_threads(20);
			//int geologicalTime = seeds[0].rgtValue;
			for (int izDeb=0; izDeb<numberOfInlines; izDeb+=blocSizeZForLoop) {
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
					for (std::size_t iy=0; iy<numberOfTraces; iy++) {
						for (std::size_t indexRef=0; indexRef<reference.size(); indexRef++) {
							int indexTrace = (reference[indexRef].iso[iy + iz*numberOfTraces] - firstSample) / sampleStep;
							reference[indexRef].rgt[iy + iz*numberOfTraces] = rgtData[(numberOfSamples*iy+indexTrace)* pr->m_cubeT->dimV() + pr->m_channelT];
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
							while (ix<numberOfSamples && rgtData[(ix+iy*numberOfSamples)* pr->m_cubeT->dimV() + pr->m_channelT]<
									rgtData[(ixOri+iy*numberOfSamples)* pr->m_cubeT->dimV() + pr->m_channelT]+
									dtauReference) {
													ix++;
							}
							ix = std::min(ix, numberOfSamples-1);
						} else if(dtauReference<0) {
							ix = std::min(ix, numberOfSamples);
							long oldX = ix;
							while (ix>=0 && rgtData[(ix+iy*numberOfSamples)* pr->m_cubeT->dimV() + pr->m_channelT]>
									rgtData[(ixOri+iy*numberOfSamples)* pr->m_cubeT->dimV() + pr->m_channelT]+dtauReference) {
								ix--;
							}
							if (ix<numberOfSamples-1 && rgtData[(ix+iy*numberOfSamples)* pr->m_cubeT->dimV() + pr->m_channelT]<
									rgtData[(ixOri+iy*numberOfSamples)* pr->m_cubeT->dimV() + pr->m_channelT]+dtauReference) {
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
									while ( ix<numberOfSamples && getNewRgtValueFromReference(iy, iz, ix, rgtData[(iy*numberOfSamples + ix)* pr->m_cubeT->dimV() + pr->m_channelT], firstSample, sampleStep, numberOfTraces, reference, referenceValues)  < seeds[i].rgtValue ) {
										ix ++ ;
									}
									if (ix>=numberOfSamples) {
										ix = numberOfSamples - 1;
									}
									weightedIso += ix*dist[i] ;
								}
							} else {
								for(int i=0; i < seeds.size() ; i++) {
									while ( ix<numberOfSamples && (rgtData[(iy*numberOfSamples + ix)* pr->m_cubeT->dimV() + pr->m_channelT])  < seeds[i].rgtValue ) {
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
										(*module)[1][iz*numberOfTraces + iy] = rawData[(iy*numberOfSamples + ix)* pr->m_cubeS->dimV() + pr->m_channelS] ;

										for (int j=0; j<window_size; j++) {
											short rawDataVal = 0;
											if (0 <= ix+j-window_size/2 && ix+j-window_size/2 < numberOfSamples) {
												rawDataVal = rawData[(iy*numberOfSamples + ix + j-window_size/2)* pr->m_cubeS->dimV() + pr->m_channelS];
											}

											hostInputData[((iz-izDeb)*numberOfTraces + iy)*window_size + j] = rawDataVal;
										}
									}
								}


					// gradient_multiscale_gpu_run(rawData, rgtData, filterKernelSize, dimy, dims.getK(), windowSize, geologicalTime, w, shift, module+static_cast<std::size_t>(dimy) * dims.getK()*2);
				// gradient_multiscale_gpu_run(rawDataVal, rgtData, filterKernelSize, dimy, dims.getK(), windowSize, geologicalTime, w, shift, module+static_cast<std::size_t>(dimy) * dims.getK()*2);
				// void gradient_multiscale_gpu_run(hostInputData, short* rgt, long dimx, dimy, long dimz, int window_size, short tau, int w, int shift, short* tabatt)


				long size0 = numberOfTraces * (izMax-izDeb) * gcc_offset;
				short *out = (short*)calloc(size0, sizeof(short));
				short *outSum = (short*)calloc(size0, sizeof(short));

				/*
				FILE *pFile = fopen("/data/PLI/jacques/in.raw", "w");
				fwrite(hostInputData, sizeof(short), (long)window_size*numberOfTraces*(izMax-izDeb), pFile);
				fclose(pFile);
				*/

				gradient_multiscale_gpu_v2_run(hostInputData, window_size, numberOfTraces, izMax-izDeb, izDeb, numberOfInlines, gcc_offset, w, shift, 1000.0f, gccComputeMean, gccMean, out, outSum);
				gccComputeMean = false;
				// gradient_multiscale_gpu_v2_run(hostInputData, window_size, numberOfTraces, izMax-izDeb, , 3, 3, 1000.0f, out, outSum);

				/*
				pFile = fopen("/data/PLI/jacques/tmp.raw", "w");
				fwrite(out, sizeof(short), size0, pFile);
				fclose(pFile);
				*/

				/*
				fprintf(stderr, "size0: %d\n", module->size());
				for (int i=0;i<module->size(); i++)
					fprintf(stderr, "%d %d\n", i, (*module)[i].size());
				// exit(0);
				 * */
				// (*module).resize(34);
				long noDataStep = 0;
				if (izDeb>0) {
					noDataStep = numberOfTraces * gcc_offset;
				}

				fprintf(stderr, "%d\n", module->size());
				for (int i=0; i<(module->size()-2)/2; i++)
				{
					(*module)[i].resize(numberOfInlines*numberOfTraces);
					for (int j=noDataStep; j<numberOfTraces*(izMax-izDeb-gcc_offset); j++)
					{
						if (out[j+numberOfTraces * (izMax-izDeb)*i]!=0) {
							(*module)[i+2][j+izDeb*numberOfTraces] = out[j+numberOfTraces * (izMax-izDeb)*i]; //gccmean[j]; //out[j+numberOfTraces * numberOfInlines*i];
						}
					}
				}
				fprintf(stderr, "%d\n", module->size());
				for (int i=0; i<(module->size()-2)/2; i++)
				{
					(*module)[i].resize(numberOfInlines*numberOfTraces);
					for (int j=noDataStep; j<numberOfTraces*(izMax-izDeb-gcc_offset); j++)
					{
						if (outSum[j+numberOfTraces * (izMax-izDeb)*i]!=0) {
							(*module)[i+2+(module->size()-2)/2][j+izDeb*numberOfTraces] = outSum[j+numberOfTraces * (izMax-izDeb)*i]; //gccmean[j]; //out[j+numberOfTraces * numberOfInlines*i];
						}
					}
				}
				free(out);
				free(outSum);
			}

			if ( gccMean ) free(gccMean);


			delete hostInputData;

			printf(" End of LayerSpectrumProcess::computeModules\n") ;
			return module;
}

ToAnalyse4Process::ToAnalyse4Process(Seismic3DDataset *cubeS, int channelS, Seismic3DDataset *cubeT, int channelT) :
		LayerProcess(cubeS, channelS, cubeT, channelT) {

}

void ToAnalyse4Process::compute(LayerSpectrumDialog *layerspectrumdialog) {
	if (! m_isComputed && m_computeMutex.try_lock()) {
		SampleTypeBinder binder(m_cubeT->sampleType());
		std::vector<std::vector<float>>* buf = binder.bind<ComputeModulesKernel> (this,
				m_constrainIso,
				m_reference,
				m_seeds,
				m_dtauReference,
				m_useSnap, m_useMedian, m_distancePower,
				5.0f,
				m_type_gcc_or_mean,
				m_gccOffset, m_w, m_shift,
				layerspectrumdialog);

			{
				fprintf(stderr, "<<<<<<<<<<<<<<<<<<<<<<< %s %d\n", __FILE__, __LINE__);
				QMutexLocker lock(&m_cacheMutex);
				if (m_module!=nullptr) {
					delete m_module;
				}
				m_module = buf;
				m_nbOutputSlices = 2 + m_gccOffset*2;
				m_isComputed = true;
				m_computeMutex.unlock();
			}

			emit LayerProcess::processCacheIsReset();
		}
}

const float* ToAnalyse4Process::getModuleData(std::size_t spectrumSlice) const {
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
		long idx = spectrumSlice;
		// if ( spectrumSlice >= 6) idx = 6;
		// if ( m_type_gcc_or_mean == 0 )  { idx = spectrumSlice%m_nbOutputSlices; } else { idx = 0; }
		outTab = (*m_module)[idx].data();// + (spectrumSlice * m_dimW * m_dimH);
		}
	return outTab;
}


