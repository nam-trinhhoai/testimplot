#include "cwt_file.h"

#include <omp.h>
#include <memory>


template<typename RgtType, typename SeismicType>
void layerRGTInterpolatorMultiSeed(const std::vector<float>& inputLayer, long dtauReference, std::vector<float>& outputLayerIso,
		std::vector<float>& outputLayerSeismic, std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers,
		std::vector<RgtSeed> seeds, const Seismic3DDataset* rgt, int channelT, const Seismic3DDataset* seismic, int channelS, bool useSnap,
		bool useMedian, int lwx, int distancePower, int snapWindow, int polarity, float tdeb, float pasech) {
	bool isCwtValid = false;
	std::string cwtSeismicPath, cwtRgtPath;
	cwtSeismicPath = seismic->path(); //remove_extension(seismic->path()) + ".cwt";
	cwtRgtPath = rgt->path(); //remove_extension(rgt->path()) + ".cwt";
	isCwtValid = cwtRgtPath.substr(cwtRgtPath.find_last_of(".") + 1)=="cwt" &&
			cwtSeismicPath.substr(cwtSeismicPath.find_last_of(".") + 1)=="cwt";
	if (isCwtValid && rgt->dimV()==1 && seismic->dimV()==1) {
		layerRGTInterpolatorMultiSeedCwt<RgtType, SeismicType>(inputLayer, dtauReference, outputLayerIso, outputLayerSeismic, referenceLayers, seeds,
				rgt, cwtRgtPath, seismic, cwtSeismicPath, useSnap, useMedian, lwx, distancePower, snapWindow, polarity, tdeb, pasech);
	} else {
		layerRGTInterpolatorMultiSeedDefault<RgtType, SeismicType>(inputLayer, dtauReference, outputLayerIso, outputLayerSeismic, referenceLayers, seeds,
				rgt, channelT, seismic, channelS, useSnap, useMedian, lwx, distancePower, snapWindow, polarity, tdeb, pasech);
	}
}

template<typename RgtType, typename SeismicType>
void layerRGTInterpolatorMultiSeedDefault(const std::vector<float>& inputLayer, long dtauReference, std::vector<float>& outputLayerIso,
		std::vector<float>& outputLayerSeismic, std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers,
		std::vector<RgtSeed> seeds, const Seismic3DDataset* rgt, int channelT, const Seismic3DDataset* seismic, int channelS, bool useSnap,
		bool useMedian, int lwx, int distancePower, int snapWindow, int polarity, float tdeb, float pasech) {
//	const io::InputOutputCube<InputType>* rgtCube = dynamic_cast<const io::InputOutputCube<InputType>*>(rgt);
//	const io::InputOutputCube<InputType>* seismicCube = dynamic_cast<const io::InputOutputCube<InputType>*>(seismic);

	long dimx = rgt->height();
	long dimy = rgt->width();
	long dimz = rgt->depth();

	int i=0;
	int type = polarity; //(seeds[idFirstSeed].seismicValue>=0) ? 1 : -1;

	//std::vector<float> referenceLayerVector;
	outputLayerIso = inputLayer;
	outputLayerSeismic.resize(dimy*dimz);

	bool isReferenceLayerSet = referenceLayers.size()!=0;

	std::vector<int> referenceValues;
	std::vector<ReferenceDuo> referenceVec;

//	std::vector<std::shared_ptr<io::InputOutputCube<InputType>>> ioCubesS, ioCubesT;
//	ioCubesS.resize(20, nullptr);
//	ioCubesT.resize(20, nullptr);

//	for (int i=0; i<ioCubesS.size(); i++) {
//		ioCubesS[i].reset(io::openCube<InputType>(seismicCube->getSourceId()));
//		ioCubesT[i].reset(io::openCube<InputType>(rgtCube->getSourceId()));
//	}

	if (isReferenceLayerSet) {
		QString isoName("isochrone");
		QString rgtName("rgt");
		referenceValues.resize(referenceLayers.size());
		referenceVec.resize(referenceLayers.size());
		for (std::size_t indexRef=0; indexRef<referenceLayers.size(); indexRef++) {
			referenceVec[indexRef].rgt.resize(dimy*dimz);
			referenceVec[indexRef].iso.resize(dimy*dimz);
			referenceLayers[indexRef]->readProperty(referenceVec[indexRef].iso.data(), isoName);
			referenceLayers[indexRef]->readProperty(referenceVec[indexRef].rgt.data(), rgtName);
		}
		std::vector<RgtType> trace;
		trace.resize(dimx * rgt->dimV());
//		ioCubesT[0]->readSubVolume(0, dimy/2, dimz/2, dimx, 1, 1, trace.data());
		rgt->readSubTraceAndSwap(trace.data(), 0, dimx, dimy/2, dimz/2);
		for (std::size_t index_ref=0; index_ref<referenceLayers.size(); index_ref++) {
			int pos_tau = (referenceVec[index_ref].iso[dimy/2 + dimy * (dimz/2)] - tdeb) / pasech;
			referenceValues[index_ref] = trace[pos_tau * rgt->dimV() + channelT];
		}

		std::vector<RgtType> sectionBuffer;
		sectionBuffer.resize(dimx*dimy*rgt->dimV());
		for (long index=0; index<seeds.size(); index++) {
			RgtSeed& seed = seeds[index];
			// read and apply
			bool isReadNeeded = false;
			for (std::size_t index_ref=0; index_ref<referenceLayers.size() && !isReadNeeded; index_ref++) {
				isReadNeeded = referenceVec[index_ref].rgt[seed.y+seed.z*dimy] == -9999.0;
			}
			if (isReadNeeded) {
				rgt->readTraceBlockAndSwap(sectionBuffer.data(), 0, dimy, seeds[index].z);
				for (std::size_t iy=0; iy<dimy; iy++) {
					for (std::size_t indexRef=0; indexRef<referenceLayers.size(); indexRef++) {
						int indexTrace = (referenceVec[indexRef].iso[iy + seed.z*dimy] - tdeb) / pasech;
						referenceVec[indexRef].rgt[iy + seed.z*dimy] = sectionBuffer[(dimx*iy+indexTrace)*rgt->dimV()+channelT];
					}
				}
			}

			seed.rgtValue = getNewRgtValueFromReference(seed.y, seed.z, seed.x, seed.rgtValue, tdeb, pasech, dimy, referenceVec, referenceValues);
		}
	}


	std::sort(seeds.begin(), seeds.end(), [](RgtSeed a, RgtSeed b){
		return a.rgtValue<b.rgtValue;
	});

	omp_set_num_threads(20);

	RgtType** rgtData_0 = new RgtType*[20];
	SeismicType** rawData_0 = new SeismicType*[20];
	for (int i=0; i<20; i++)
	{
		rgtData_0[i] = new RgtType[dimx*dimy*rgt->dimV()];
		rawData_0[i] = new SeismicType[dimx*dimy*rgt->dimV()];
	}


#pragma omp parallel for schedule (dynamic)
	for (int z=0; z<dimz; z++) {
		int threadId = omp_get_thread_num();
//		std::shared_ptr<io::InputOutputCube<InputType>> _iocubeS = ioCubesS[threadId];
//		std::shared_ptr<io::InputOutputCube<InputType>> _iocubeT = ioCubesT[threadId];
		printf(" numplan %d/%d\n",z,dimz) ;
		std::vector<int> points;
		points.resize(dimy);

		std::vector<double> dist;
		dist.resize(seeds.size());
		RgtType* rgtData = rgtData_0[threadId];
		SeismicType* rawData = rawData_0[threadId];
		rgt->readTraceBlockAndSwap(rgtData, 0, dimy, z);
		seismic->readTraceBlockAndSwap(rawData, 0, dimy, z);
		for (std::size_t iy=0; iy<dimy; iy++) {
			for (std::size_t indexRef=0; indexRef<referenceVec.size(); indexRef++) {
				int indexTrace = (referenceVec[indexRef].iso[iy + z*dimy] - tdeb) / pasech;
				referenceVec[indexRef].rgt[iy + z*dimy] = rgtData[(dimx*iy+indexTrace)*rgt->dimV()+channelT];
			}
		}

		for (int y=0; y<dimy; y++) {
			double som=0.0 ;

			long ix;
			bool seedFound = false;
			if (outputLayerIso[y+z*dimy]!=-9999) {
				//points[y] = (outputLayerIso[y+z*dimy] - tdeb)/pasech;
				long ixOri = (outputLayerIso[y+z*dimy] - tdeb)/pasech;
				ix = ixOri;
				if (dtauReference>0) {
					ix = std::max(ix, 0l);
					while (ix<dimx && rgtData[(ix+y*dimx)*rgt->dimV()+channelT]<rgtData[(ixOri+y*dimx)*rgt->dimV()+channelT]+
							dtauReference) {
						ix++;
					}
					ix = std::min(ix, dimx-1);
				} else if(dtauReference<0) {
					ix = std::min(ix, dimx);
					long oldX = ix;
					while (ix>=0 && rgtData[(ix+y*dimx)*rgt->dimV()+channelT]>rgtData[(ixOri+y*dimx)*rgt->dimV()+channelT]+dtauReference) {
						ix--;
					}
					if (ix<dimx-1 && rgtData[(ix+y*dimx)*rgt->dimV()+channelT]<rgtData[(ixOri+y*dimx)*rgt->dimV()+channelT]+dtauReference) {
						ix++;
					}

				}
				seedFound = true;
			} else {
				for(int i=0; (i < seeds.size()) && !seedFound; i++) {
					long val = ((y - seeds[i].y)*(y-seeds[i].y) + (z -seeds[i].z)*(z-seeds[i].z));
					if (val!=0) {
						dist[i] = 1.0 / std::pow(val,distancePower) ;
						som += dist[i] ;
					} else {
						ix = seeds[i].x;
						seedFound = true;
					}
				}
			}

			if(!seedFound) {
				ix = 0 ;
				double weightedIso = 0;
				if (isReferenceLayerSet) {
					for(int i=0; i < seeds.size() ; i++) {
						while ( getNewRgtValueFromReference(y, z, ix, rgtData[(y*dimx + ix)*rgt->dimV()+channelT], tdeb, pasech, dimy, referenceVec, referenceValues)  < seeds[i].rgtValue  && ix < dimx ) {
							ix ++ ;
						}
						if (ix>=dimx) {
							ix = dimx-1;
						}
						weightedIso += ix*dist[i] ;
					}
				} else {
					for(int i=0; i < seeds.size() ; i++) {
						while ( (rgtData[(y*dimx + ix)*rgt->dimV()+channelT])  < seeds[i].rgtValue  && ix < dimx ) {
							ix ++ ;
						}
						if (ix>=dimx) {
							ix = dimx-1;
						}
						weightedIso += ix*dist[i] ;
					}
				}
				points[y] = weightedIso/som ;
			} else {
				points[y] = ix;
			}
		}

		if (useSnap) {
			// snap
			std::vector<SeismicType> trace;
			trace.resize(dimx);
			for (int y=0; y<dimy; y++) {
				for (long idx=y*dimx; idx<(y+1)*dimx; idx++) {
					trace[idx-y*dimx] = rawData[idx*seismic->dimV()+channelS];
				}
				int x = points[y];
				int newx = bl_indpol(x, trace.data(), dimx, type, snapWindow);
				points[y] = (newx==SLOPE::RAIDE)? x : newx;
			}
		}

		if (useMedian) {
			// apply median
			UtFiltreMedianeX(points.data(), points.size(), 1, lwx);
		}

		for (int y=0; y<dimy; y++) {
			outputLayerIso[z * dimy + y] = points[y]*pasech + tdeb;
			outputLayerSeismic[z*dimy+y] = rawData[(y*dimx+points[y])*seismic->dimV()+channelS];
		}
	}
	for (int i=0; i<20; i++)
	{
		delete rgtData_0[i];
		delete rawData_0[i];
	}
	delete rawData_0;
	delete rgtData_0;
	//outputLayer->writeProperty(referenceLayerVector.data(), isoName);
}

template<typename RgtType, typename SeismicType>
void layerRGTInterpolatorMultiSeedCwt(const std::vector<float>& inputLayer, long dtauReference, std::vector<float>& outputLayerIso,
		std::vector<float>& outputLayerSeismic, std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers,
		std::vector<RgtSeed> seeds, const Seismic3DDataset* rgt, std::string rgtCwtPath, const Seismic3DDataset* seismic,
		std::string seismicCwtPath, bool useSnap, bool useMedian, int lwx, int distancePower, int snapWindow, int polarity,
		float tdeb, float pasech) {
//	const io::InputOutputCube<InputType>* rgtCube = dynamic_cast<const io::InputOutputCube<InputType>*>(rgt);
//	const io::InputOutputCube<InputType>* seismicCube = dynamic_cast<const io::InputOutputCube<InputType>*>(seismic);

	long dimx = rgt->height();
	long dimy = rgt->width();
	long dimz = rgt->depth();

	int i=0;
	int type = polarity; //(seeds[idFirstSeed].seismicValue>=0) ? 1 : -1;

	//std::vector<float> referenceLayerVector;
	outputLayerIso = inputLayer;
	outputLayerSeismic.resize(dimy*dimz);

	bool isReferenceLayerSet = referenceLayers.size()!=0;

	std::vector<int> referenceValues;
	std::vector<ReferenceDuo> referenceVec;

//	std::vector<std::shared_ptr<io::InputOutputCube<float>>> ioCubesS, ioCubesT;
//	ioCubesS.resize(20, nullptr);
//	ioCubesT.resize(20, nullptr);
//
//	for (int i=0; i<ioCubesS.size(); i++) {
//		ioCubesS[i].reset(io::openCube<float>(seismicCube->getSourceId()));
//		ioCubesT[i].reset(io::openCube<float>(rgtCube->getSourceId()));
//	}

	std::list<CWT_FILE> filesS, filesT;
	for (int i=0; i<20; i++) {
		CWT_FILE initFile, initFile2;
		filesS.push_back(std::move(initFile));

		filesT.push_back(std::move(initFile2));

		// init cwt files
		std::list<CWT_FILE>::iterator itS = filesS.begin();
		std::list<CWT_FILE>::iterator itT = filesT.begin();
		std::advance(itS, i);
		std::advance(itT, i);

		itS->openForRead(seismicCwtPath.c_str(), CWT_FILE::FLOAT);
		itT->openForRead(rgtCwtPath.c_str(), CWT_FILE::FLOAT);
	}

	if (isReferenceLayerSet) {
		QString isoName("isochrone");
		QString rgtName("rgt");
		referenceValues.resize(referenceLayers.size());
		referenceVec.resize(referenceLayers.size());
		for (std::size_t indexRef=0; indexRef<referenceLayers.size(); indexRef++) {
			referenceVec[indexRef].rgt.resize(dimy*dimz);
			referenceVec[indexRef].iso.resize(dimy*dimz);
			referenceLayers[indexRef]->readProperty(referenceVec[indexRef].iso.data(), isoName);
			referenceLayers[indexRef]->readProperty(referenceVec[indexRef].rgt.data(), rgtName);
		}
		std::vector<RgtType> trace;
		trace.resize(dimx);
		rgt->readSubTraceAndSwap(trace.data(), 0, dimx, dimy/2, dimz/2);
		for (std::size_t index_ref=0; index_ref<referenceLayers.size(); index_ref++) {
			int pos_tau = (referenceVec[index_ref].iso[dimy/2 + dimy * (dimz/2)] - tdeb) / pasech;
			referenceValues[index_ref] = trace[pos_tau];
		}

		std::vector<float> sectionBuffer;
		sectionBuffer.resize(dimx*dimy);
		for (long index=0; index<seeds.size(); index++) {
			RgtSeed& seed = seeds[index];
			// read and apply
			bool isReadNeeded = false;
			for (std::size_t index_ref=0; index_ref<referenceLayers.size() && !isReadNeeded; index_ref++) {
				isReadNeeded = referenceVec[index_ref].rgt[seed.y+seed.z*dimy] == -9999.0;
			}
			if (isReadNeeded) {
				std::list<CWT_FILE>::iterator itT = filesT.begin();
				itT->inlineRead(seeds[index].z, sectionBuffer.data());

//				rgtCube->readSubVolume(0, 0, seeds[index].z, dimx, dimy, 1, sectionBuffer.data());
				for (std::size_t iy=0; iy<dimy; iy++) {
					for (std::size_t indexRef=0; indexRef<referenceLayers.size(); indexRef++) {
						int indexTrace = (referenceVec[indexRef].iso[iy + seed.z*dimy] - tdeb) / pasech;
						referenceVec[indexRef].rgt[iy + seed.z*dimy] = sectionBuffer[dimx*iy+indexTrace];
					}
				}
			}

			seed.rgtValue = getNewRgtValueFromReference(seed.y, seed.z, seed.x, seed.rgtValue, tdeb, pasech, dimy, referenceVec, referenceValues);
		}
	}


	std::sort(seeds.begin(), seeds.end(), [](RgtSeed a, RgtSeed b){
		return a.rgtValue<b.rgtValue;
	});

	omp_set_num_threads(20);

	float** rgtData_0 = new float*[20];
	float** rawData_0 = new float*[20];
	for (int i=0; i<20; i++)
	{
		rgtData_0[i] = new float[dimx*dimy];
		rawData_0[i] = new float[dimx*dimy];
	}


#pragma omp parallel for schedule (dynamic)
	for (int z=0; z<dimz; z++) {
		int threadId = omp_get_thread_num();
//		std::shared_ptr<io::InputOutputCube<float>> _iocubeS = ioCubesS[threadId];
//		std::shared_ptr<io::InputOutputCube<float>> _iocubeT = ioCubesT[threadId];
		printf(" numplan %d/%d\n",z,dimz) ;
		std::vector<int> points;
		points.resize(dimy);

		std::vector<double> dist;
		dist.resize(seeds.size());
		float* rgtData = rgtData_0[threadId];
		float* rawData = rawData_0[threadId];
//		_iocubeT->readSubVolume(0, 0, z, dimx, dimy, 1, rgtData);
//		_iocubeS->readSubVolume(0, 0, z, dimx, dimy, 1, rawData);

		std::list<CWT_FILE>::iterator itS = filesS.begin();
		std::list<CWT_FILE>::iterator itT = filesT.begin();
		std::advance(itS, threadId);
		std::advance(itT, threadId);
		itS->inlineRead(z, rawData);
		itT->inlineRead(z, rgtData);
		for (std::size_t iy=0; iy<dimy; iy++) {
			for (std::size_t indexRef=0; indexRef<referenceVec.size(); indexRef++) {
				int indexTrace = (referenceVec[indexRef].iso[iy + z*dimy] - tdeb) / pasech;
				referenceVec[indexRef].rgt[iy + z*dimy] = rgtData[dimx*iy+indexTrace];
			}
		}

		for (int y=0; y<dimy; y++) {
			double som=0.0 ;

			long ix;
			bool seedFound = false;
			if (outputLayerIso[y+z*dimy]!=-9999) {
				//points[y] = (outputLayerIso[y+z*dimy] - tdeb)/pasech;
				long ixOri = (outputLayerIso[y+z*dimy] - tdeb)/pasech;
				ix = ixOri;
				if (dtauReference>0) {
					ix = std::max(ix, 0l);
					while (ix<dimx && rgtData[ix+y*dimx]<rgtData[ixOri+y*dimx]+
							dtauReference) {
						ix++;
					}
					ix = std::min(ix, dimx-1);
				} else if(dtauReference<0) {
					ix = std::min(ix, dimx);
					long oldX = ix;
					while (ix>=0 && rgtData[ix+y*dimx]>rgtData[ixOri+y*dimx]+dtauReference) {
						ix--;
					}
					if (ix<dimx-1 && rgtData[ix+y*dimx]<rgtData[ixOri+y*dimx]+dtauReference) {
						ix++;
					}

				}
				seedFound = true;
				//continue; // skip
			} else {
				for(int i=0; (i < seeds.size()) && !seedFound; i++) {
					long val = ((y - seeds[i].y)*(y-seeds[i].y) + (z -seeds[i].z)*(z-seeds[i].z));
					if (val!=0) {
						dist[i] = 1.0 / std::pow(val,distancePower) ;
						som += dist[i] ;
					} else {
						ix = seeds[i].x;
						seedFound = true;
					}
				}
			}

			if(!seedFound) {
				ix = 0 ;
				double weightedIso = 0;
				if (isReferenceLayerSet) {
					for(int i=0; i < seeds.size() ; i++) {
						while ( getNewRgtValueFromReference(y, z, ix, rgtData[y*dimx + ix], tdeb, pasech, dimy, referenceVec, referenceValues)  < seeds[i].rgtValue  && ix < dimx ) {
							ix ++ ;
						}
						if (ix>=dimx) {
							ix = dimx-1;
						}
						weightedIso += ix*dist[i] ;
					}
				} else {
					for(int i=0; i < seeds.size() ; i++) {
						while ( (rgtData[y*dimx + ix])  < seeds[i].rgtValue  && ix < dimx ) {
							ix ++ ;
						}
						if (ix>=dimx) {
							ix = dimx-1;
						}
						weightedIso += ix*dist[i] ;
					}
				}
				points[y] = weightedIso/som ;
			} else {
				points[y] = ix;
			}
		}

		if (useSnap) {
			// snap
			for (int y=0; y<dimy; y++) {
				int x = points[y];
				int newx = bl_indpol(x, rawData+y*dimx, dimx, type, snapWindow);
				points[y] = (newx==SLOPE::RAIDE)? x : newx;
			}
		}

		if (useMedian) {
			// apply median
			UtFiltreMedianeX(points.data(), points.size(), 1, lwx);
		}

		for (int y=0; y<dimy; y++) {
			outputLayerIso[z * dimy + y] = points[y]*pasech + tdeb;
			outputLayerSeismic[z*dimy+y] = rawData[y*dimx+points[y]];
		}
	}
	for (int i=0; i<20; i++)
	{
		delete rgtData_0[i];
		delete rawData_0[i];
	}
	delete rawData_0;
	delete rgtData_0;
	//outputLayer->writeProperty(referenceLayerVector.data(), isoName);
}

