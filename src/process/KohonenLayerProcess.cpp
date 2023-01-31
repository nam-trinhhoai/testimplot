/*
 *
 *  Created on: 24 janv. 2021
 *      Author: Armand
 */



#include "KohonenLayerProcess.h"

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

#include "KohonenProcess.h"
#include "ijkhorizon.h"
//#include <DatasetCubeUtil.h>
//#include "SampleTypeBinder.h"
#include "palette/imageformats.h"
#include "RgtLayerProcessUtil.h"
#include "ioutil.h"
#include "sampletypebinder.h"


std::vector<std::vector<float>>* KohonenLayerProcess::computeModules(std::vector<RgtSeed> seeds, bool polarity,
		bool useSnap, bool useMedian, int lwx, int distancePower,
		int snapWindow,const std::vector<float>& constrainLayer, long dtauReference,
		std::vector<ReferenceDuo>& reference, LayerSpectrumDialog *layerspectrumdialog) {
	std::unique_ptr<AbstractKohonenProcess> process;
	process.reset(AbstractKohonenProcess::getObjectFromDataset(m_cubeS, {m_channelS}));

	IAbstractIsochrone* isochrone = new SeedLayerIsochrone(m_cubeT, m_channelT, seeds, polarity, useSnap, useMedian, lwx, distancePower,
			snapWindow, constrainLayer, dtauReference, reference);

	process->setExtractionIsochrone(isochrone);

	std::vector<std::vector<float>>* output = new std::vector<std::vector<float>>();
	std::vector<float> init;
	init.resize(m_cubeT->width()*m_cubeT->depth(), 0);
	output->resize(3, init);

	std::vector<float> attribute, labelBuffer;
	attribute.resize(m_cubeT->width()*m_cubeT->depth());
	labelBuffer.resize(m_cubeT->width()*m_cubeT->depth());

	process->setOutputOnIsochroneAttribute(attribute.data());
	process->setOutputHorizonBuffer(labelBuffer.data());

	//process->setOutputHorizonProperties(outputLayer, "Tmap");

	bool success = process->compute(m_tmapExampleSize, m_tmapSize, m_tmapExampleStep);

	float tdeb = m_cubeT->sampleTransformation()->b();
	float pasech = m_cubeT->sampleTransformation()->a();

	if (success) {
		float* isoTab = isochrone->getTab();

		#pragma omp parallel for
		for (long i=0; i<m_cubeT->width()*m_cubeT->depth(); i++) {
			(*output)[0][i] = (isoTab[i] - tdeb) / pasech;
			(*output)[1][i] = attribute[i];
			(*output)[2][i] = labelBuffer[i];
		}
	}

	return output;
}

KohonenLayerProcess::KohonenLayerProcess(
		Seismic3DDataset* cubeS, int channelS,
		Seismic3DDataset* cubeT, int channelT) :
	LayerProcess(cubeS, channelS, cubeT, channelT) {
	m_module = nullptr;
}

// JDTODO
void KohonenLayerProcess::compute(LayerSpectrumDialog *layerspectrumdialog) {
	fprintf(stderr, "<<<<<<<<<<<<<<<<<<<<<<< %s %d\n", __FILE__, __LINE__);
	if (! m_isComputed && m_computeMutex.try_lock()) {
		std::vector<std::vector<float>>* buf = computeModules (m_seeds, m_polarity, m_useSnap,
				m_useMedian, m_lwx_medianFilter, m_distancePower, m_snapWindow, m_constrainIso, m_dtauReference,
				m_reference, layerspectrumdialog);

		{
			fprintf(stderr, "<<<<<<<<<<<<<<<<<<<<<<< %s %d\n", __FILE__, __LINE__);
			QMutexLocker lock(&m_cacheMutex);
			if (m_module!=nullptr) {
				delete m_module;
			}
			m_module = buf;
			m_nbOutputSlices = 3;
			m_isComputed = true;
			m_computeMutex.unlock();
		}

		emit LayerProcess::processCacheIsReset();
	}
}

const float* KohonenLayerProcess::getModuleData(std::size_t spectrumSlice) const {
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


SeedLayerIsochrone::SeedLayerIsochrone(Seismic3DDataset* cubeT, int channel, const std::vector<RgtSeed>& seeds, bool polarity,
			bool useSnap, bool useMedian, int lwx, int distancePower,
			int snapWindow,const std::vector<float>& constrainLayer, long dtauReference,
			std::vector<ReferenceDuo>& reference) : m_cubeT(cubeT), m_seeds(seeds), m_polarity(polarity),
			m_useSnap(useSnap), m_useMedian(useMedian), m_lwx(lwx), m_distancePower(distancePower),
			m_snapWindow(snapWindow), m_constrainLayer(constrainLayer), m_dtauReference(dtauReference),
			m_reference(reference) {
	m_numTraces = cubeT->width();
	m_numProfils = cubeT->depth();
	m_readInlines.resize(m_numProfils, false);
	m_fullMapBuffer.resize(m_numTraces*m_numProfils);
	m_channel = channel;

	m_N = m_numProfils;
}

SeedLayerIsochrone::~SeedLayerIsochrone() {

}

int SeedLayerIsochrone::getNumTraces() const {
	return m_numTraces;
}

int SeedLayerIsochrone::getNumProfils() const {
	return m_numProfils;
}

float SeedLayerIsochrone::getValue(long i, long j, bool* ok) {
	SampleTypeBinder binder(m_cubeT->sampleType());
	float val;
	(*ok) = i>=0 && i<m_numTraces && j>=0 && j<m_numProfils;
	if (*ok) {
		if (m_readInlines[j]) {
			val = m_fullMapBuffer[i + j * m_numTraces];
		} else {
			binder.bind<GetIsochroneInlineKernel>(this, j, m_fullMapBuffer.data()+j*m_numTraces);
			val = m_fullMapBuffer[i+j*m_numTraces];
			m_readInlines[j] = true;
			m_N--;
		}
	}
	return val;
}

float* SeedLayerIsochrone::getTab() {
	SampleTypeBinder binder(m_cubeT->sampleType());
	if (m_N>0) {
		for (long j=0; j<m_numProfils; j++) {
			if (!m_readInlines[j]) {
				binder.bind<GetIsochroneInlineKernel>(this, j, m_fullMapBuffer.data() + m_numTraces * j);
				m_readInlines[j] = true;
				m_N--;
			}
		}
	}
	return m_fullMapBuffer.data();
}

template<typename InputCubeType>
void SeedLayerIsochrone::GetIsochroneInlineKernel<InputCubeType>::run(SeedLayerIsochrone* obj, long z, float* tab) {

	int type = obj->m_polarity ? 1 : -1;

	double firstSample = obj->m_cubeT->sampleTransformation()->b(); //TODO seismicAddon.getFirstSample();
	double sampleStep = obj->m_cubeT->sampleTransformation()->a(); //TODO seismicAddon.getSampleStep();
	int numberOfSamples = obj->m_cubeT->height(); // samples
	int numberOfTraces = obj->m_cubeT->width(); // traces
	int numberOfInlines = obj->m_cubeT->depth(); // Inline

	std::size_t sizeLayer = static_cast<std::size_t>(numberOfTraces) * numberOfInlines;
	std::size_t sizeInline = static_cast<std::size_t>(numberOfTraces) * numberOfSamples;


	bool isConstrainSet = obj->m_constrainLayer.size()==sizeLayer;

	bool isReferenceSet = false;//reference.size()!=0;
	std::vector<int> referenceValues;

	if (isReferenceSet) {
		referenceValues.resize(obj->m_reference.size());
		std::vector<InputCubeType> trace;
		trace.resize(numberOfSamples*obj->m_cubeT->dimV());
		obj->m_cubeT->readSubTraceAndSwap(trace.data(), 0, numberOfSamples, numberOfTraces/2, numberOfInlines/2);

		for (std::size_t index_ref=0; index_ref<obj->m_reference.size(); index_ref++) {
			int pos_tau = (obj->m_reference[index_ref].iso[numberOfTraces/2 + numberOfTraces * (numberOfInlines/2)] - firstSample) / sampleStep;
			referenceValues[index_ref] = trace[pos_tau * obj->m_cubeT->dimV() + obj->m_channel];
		}

		std::vector<InputCubeType> sectionBuffer;
		sectionBuffer.resize(sizeInline*obj->m_cubeT->dimV());
		for (long index=0; index<obj->m_seeds.size(); index++) {
			RgtSeed& seed = obj->m_seeds[index];
			// read and apply
			bool isReadNeeded = false;
			for (std::size_t index_ref=0; index_ref<obj->m_reference.size() && !isReadNeeded; index_ref++) {
				isReadNeeded = obj->m_reference[index_ref].rgt[seed.y+seed.z*numberOfTraces] == -9999.0;
			}
			if (isReadNeeded) {
//					ioCubesT[0]->readSubVolume(0, 0, seeds[index].z, dims.getI(), dims.getJ(), 1, sectionBuffer.data());
				obj->m_cubeT->readTraceBlockAndSwap(sectionBuffer.data(), 0, numberOfTraces, obj->m_seeds[index].z);
				for (std::size_t iy=0; iy<numberOfTraces; iy++) {
					for (std::size_t indexRef=0; indexRef<obj->m_reference.size(); indexRef++) {
						int indexTrace = (obj->m_reference[indexRef].iso[iy + seed.z*numberOfTraces] - firstSample) / sampleStep;
						obj->m_reference[indexRef].rgt[iy + seed.z*numberOfTraces] = sectionBuffer[(numberOfSamples*iy+indexTrace)*obj->m_cubeT->dimV()+obj->m_channel];
					}
				}
			}

			seed.rgtValue = getNewRgtValueFromReference(seed.y, seed.z, seed.x, seed.rgtValue, firstSample, sampleStep, numberOfTraces, obj->m_reference, referenceValues);
		}
	}

	// sort seed vector
	std::sort(obj->m_seeds.begin(), obj->m_seeds.end(), [](RgtSeed a, RgtSeed b){
		return a.rgtValue<b.rgtValue;
	});

	std::vector<InputCubeType> ioCubesT;
	ioCubesT.resize(numberOfSamples * numberOfTraces * obj->m_cubeT->dimV());


	std::vector<double> dist;
	dist.resize(obj->m_seeds.size());

	std::vector<InputCubeType>& sliceT = ioCubesT;
	obj->m_cubeT->readInlineBlock(sliceT.data(), z, z+1, false);

	InputCubeType* rgtData = sliceT.data();

	// swap
	/*for (std::size_t index=0; index<sliceT.size(); index++) {
		unsigned char* tab = (unsigned char*)(rgtData+index);
		unsigned char tmp = tab[0];
		tab[0] = tab[1];
		tab[1] = tmp;
	}*/

	if (isReferenceSet) {
		for (std::size_t iy=0; iy<numberOfTraces; iy++) {
			for (std::size_t indexRef=0; indexRef<obj->m_reference.size(); indexRef++) {
				int indexTrace = (obj->m_reference[indexRef].iso[iy + z*numberOfTraces] - firstSample) / sampleStep;
				obj->m_reference[indexRef].rgt[iy + z*numberOfTraces] = rgtData[(numberOfSamples*iy+indexTrace)*obj->m_cubeT->dimV()+obj->m_channel];
			}
		}
	}


	std::vector<float> points;
	points.resize(numberOfTraces);

	for (std::size_t iy=0; iy<numberOfTraces; iy++) {
		double som=0.0 ;

		int ix;
		bool seedFound = false;
		if (isConstrainSet && obj->m_constrainLayer[z*numberOfTraces+iy]!=-9999) {
			long ixOri = (obj->m_constrainLayer[z*numberOfTraces+iy] - firstSample) / sampleStep;
			ix = ixOri;
			if (obj->m_dtauReference>0) {
				ix = std::max(ix, 0);
				while (ix<numberOfSamples && rgtData[(ix+iy*numberOfSamples)*obj->m_cubeT->dimV()+obj->m_channel]<
						rgtData[(ixOri+iy*numberOfSamples)*obj->m_cubeT->dimV()+obj->m_channel]+
						obj->m_dtauReference) {
					ix++;
				}
				ix = std::min(ix, numberOfSamples-1);
			} else if(obj->m_dtauReference<0) {
				ix = std::min(ix, numberOfSamples);
				long oldX = ix;
				while (ix>=0 && rgtData[(ix+iy*numberOfSamples)*obj->m_cubeT->dimV()+obj->m_channel]>
						rgtData[(ixOri+iy*numberOfSamples)*obj->m_cubeT->dimV()+obj->m_channel]+obj->m_dtauReference) {
					ix--;
				}
				if (ix<numberOfSamples-1 && rgtData[(ix+iy*numberOfSamples)*obj->m_cubeT->dimV()+obj->m_channel]<
						rgtData[(ixOri+iy*numberOfSamples)*obj->m_cubeT->dimV()+obj->m_channel]+obj->m_dtauReference) {
					ix++;
				}

			}
			//ix = (constrainLayer[iz*numberOfTraces+iy] - firstSample) / sampleStep;
			seedFound = true;
		} else {
			for(int i=0; (i < obj->m_seeds.size()) && !seedFound; i++) {
				long val = ((iy - obj->m_seeds[i].y)*(iy-obj->m_seeds[i].y) + (z -obj->m_seeds[i].z)*(z-obj->m_seeds[i].z));
				if (val!=0) {
					dist[i] = 1.0 / std::pow(val,obj->m_distancePower) ;
					som += dist[i] ;
				} else {
					ix = obj->m_seeds[i].x;
					seedFound = true;
				}
			}
		}

		float floatIx = ix;
		if(!seedFound) {
			ix = 0 ;
			double weightedIso = 0;
			if (isReferenceSet) {
				for(int i=0; i < obj->m_seeds.size() ; i++) {
					while ( ix<numberOfSamples && getNewRgtValueFromReference(iy, z, ix, rgtData[(iy*numberOfSamples + ix)*obj->m_cubeT->dimV()+obj->m_channel], firstSample, sampleStep, numberOfTraces, obj->m_reference, referenceValues)  < obj->m_seeds[i].rgtValue ) {
						ix ++ ;
					}
					if (ix>=numberOfSamples) {
						ix = numberOfSamples - 1;
					}
					weightedIso += ix*dist[i] ;
				}
			} else {
				for(int i=0; i < obj->m_seeds.size() ; i++) {
					while ( ix<numberOfSamples && (rgtData[(iy*numberOfSamples + ix)*obj->m_cubeT->dimV()+obj->m_channel])  < obj->m_seeds[i].rgtValue ) {
						ix ++ ;
					}
					if (ix>=numberOfSamples) {
						ix = numberOfSamples - 1;
					}
					double ixDouble = ix;
					if (rgtData[(iy*numberOfSamples + ix)*obj->m_cubeT->dimV()+obj->m_channel]==obj->m_seeds[i].rgtValue || ix==0) {
						ixDouble = ix;
					} else {
						double ix_floor_rgt = rgtData[(iy*numberOfSamples + ix-1)*obj->m_cubeT->dimV()+obj->m_channel];
						double ix_rgt = rgtData[(iy*numberOfSamples + ix-1)*obj->m_cubeT->dimV()+obj->m_channel];
						ixDouble = ix-1 + (obj->m_seeds[i].rgtValue - ix_floor_rgt) / (ix_rgt - ix_floor_rgt);

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
		if (obj->m_useSnap) {
//TODO					int newx = process::bl_indpol(ix, rawData+iy*height, height, type, snapWindow);
//TODO					ix = (newx==process::SLOPE::RAIDE)? ix : newx;
		}
		points[iy] = floatIx;
	}

	if (obj->m_useMedian) {
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
		//(*module)[0][iz*numberOfTraces + iy] = ix; //firstSample + ix * sampleStep;
		//(*module)[1][iz*numberOfTraces + iy] = rawData[iy*numberOfSamples + ix] ;

		tab[iy] = firstSample + floatIx * sampleStep;
	}
}

