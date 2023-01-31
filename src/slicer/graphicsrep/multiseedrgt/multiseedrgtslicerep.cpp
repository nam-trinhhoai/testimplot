#include "multiseedrgtslicerep.h"
#include "multiseedrgtslicelayer.h"
#include "seismic3ddataset.h"
#include "multiseedrgt.h"
#include "slicerep.h"
#include "imouseimagedataprovider.h"
#include "affine2dtransformation.h"
#include "cudaimagepaletteholder.h"
#include "fixedlayerfromdataset.h"
#include "abstractsectionview.h"

#include <omp.h>
#include <cmath>
#include <tuple>
#include <QMutexLocker>

MultiSeedRgtSliceRep::MultiSeedRgtSliceRep(MultiSeedRgt *data,
		const QPair<QVector2D,AffineTransformation> & sliceRangeAndTransfo, SliceDirection dir,
				AbstractInnerView *parent)  : AbstractGraphicRep(parent),
				m_eventFilterClass({Qt::Key_S}, this)
{

}

MultiSeedRgtSliceRep::~MultiSeedRgtSliceRep() {
	m_parent->removeEventFilter(&m_eventFilterClass);
	if (m_layer!=nullptr) {
		delete m_layer;
	}
}

void MultiSeedRgtSliceRep::notifyDataControlerMousePressed(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys) {
		fprintf(stderr, "---> pressed\n");
}

std::size_t MultiSeedRgtSliceRep::addPoint(QPoint point) {
	fprintf(stderr, "---> pressed\n");
	return 0;
}

std::size_t MultiSeedRgtSliceRep::addPoint(int x, int y) { // add seed on current section
	return addPoint(QPoint(x, y));
}


/*
std::pair<SliceRep*, SliceRep*> MultiSeedRgtSliceRep::findSliceRepsFromSectionInnerViewAndData(
		MultiSeedRgt *data, AbstractSectionView *parent) {
	SliceRep* seismic = nullptr;
	SliceRep* rgt = nullptr;

	const QList<AbstractGraphicRep*>& reps = parent->getVisibleReps();
	std::size_t index = 0;
	while (index<reps.size() && (seismic==nullptr || rgt==nullptr)) {
		SliceRep* slice = dynamic_cast<SliceRep*>(reps[index]);
		if (slice!=nullptr && slice->data()==data->seismic()) {
			seismic = slice;
		} else if (slice!=nullptr && slice->data()==data->rgt()) {
			rgt = slice;
		}
		index++;
	}

	std::pair<SliceRep*, SliceRep*> out;
	out.first = seismic;
	out.second = rgt;


	return out;
}

MultiSeedRgtSliceRep::MultiSeedRgtSliceRep(MultiSeedRgt *data,
		const QPair<QVector2D,AffineTransformation> & sliceRangeAndTransfo, SliceDirection dir,
				AbstractInnerView *parent)  : AbstractGraphicRep(parent),
				m_eventFilterClass({Qt::Key_S}, this)
{

}



MultiSeedRgtSliceRep::~MultiSeedRgtSliceRep() {
	m_parent->removeEventFilter(&m_eventFilterClass);
	if (m_layer!=nullptr) {
		delete m_layer;
	}
}

void MultiSeedRgtSliceRep::disconnectMoveSlot(QObject* requestingObj) {
	// disconnect(m_data, &MultiSeedHorizon::dtauPolygonMainCacheChanged, this, &MultiSeedSliceRep::applyDeltaTau);
	// disconnect(m_data, &MultiSeedHorizon::pointMoved, this, &MultiSeedSliceRep::pointMovedSynchro);
}

void MultiSeedRgtSliceRep::connectMoveSlotAndUpdate(QObject* requestingObj) {
	// connect(m_data, &MultiSeedHorizon::dtauPolygonMainCacheChanged, this, &MultiSeedSliceRep::applyDeltaTau);
	// connect(m_data, &MultiSeedHorizon::pointMoved, this, &MultiSeedSliceRep::pointMovedSynchro);
	// if (requestingObj!=this) { // emiting rep is supposed to take care of it itself
	//	m_polygonMainCache.clear();
	//	clearCurveSpeedupCache();
	//	updateMainHorizon();
	// }
}

void MultiSeedRgtSliceRep::updateMainHorizonNewConstrain() {
	// clearCurveSpeedupCache();
	// m_data->setDTauReference(0);
	// updateMainHorizon();
}

void MultiSeedRgtSliceRep::updateMainHorizonNoCache() {
	// clearCurveSpeedupCache();
	// updateMainHorizon();
}

int MultiSeedRgtSliceRep::currentSliceWorldPosition() const {
    // double val;
    // m_sliceRangeAndTransfo.second.direct((double)m_currentSlice,val);
    // return (int)val;
	return 0;
}

int MultiSeedRgtSliceRep::currentSliceIJPosition() const {
	return m_currentSlice;
}

const QPolygon& MultiSeedRgtSliceRep::getMainPolygon() const {
	//QMutexLocker lock(&m_polygonMutex);
	return m_polygonMain;
}

const QPolygon& MultiSeedRgtSliceRep::getTopPolygon() const {
	//QMutexLocker lock(&m_polygonMutex);
	return m_polygonTop;
}

const QPolygon& MultiSeedRgtSliceRep::getBottomPolygon() const {
	//QMutexLocker lock(&m_polygonMutex);
	return m_polygonBottom;
}

// jd
const std::vector<std::vector<QPolygon>>& MultiSeedRgtSliceRep::getReferencePolygon() const {
	//QMutexLocker lock(&m_polygonMutex);
	return m_polygonReference;
}


//AbstractGraphicRep
QWidget* MultiSeedRgtSliceRep::propertyPanel() {
	return nullptr;
}

GraphicLayer* MultiSeedRgtSliceRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	// if (m_layer==nullptr) {
	//	m_layer = new MultiSeedSliceLayer(this, scene, defaultZDepth, parent);
	// }
	return m_layer;
}

std::size_t MultiSeedRgtSliceRep::addPoint(QPoint point) {
	if (m_data->getBehaviorMode()==MOUSETRACKING) {
		return MultiSeedHorizon::INVALID_ID;
	}

	initSliceReps();

	if (m_data->getDTauPolygonMainCache()!=0) {
		m_data->applyDtauToSeeds(); //applyDTauToData();
		m_polygonMainCache.clear();
	}

	// care point is transposed
	long dimI = m_data->seismic()->height();
	long dimJ;
	if (m_dir==SliceDirection::Inline) { // Z
		dimJ = m_data->seismic()->width();
	} else if(m_dir==SliceDirection::XLine) { // Y
		dimJ = m_data->seismic()->depth();
	} else {
		return MultiSeedHorizon::INVALID_ID;
	}
	if ((point.x()<0 || point.x()>=dimJ) && (point.y()<0 && point.y()>=dimI)) {
		return MultiSeedHorizon::INVALID_ID;
	}
	RgtSeed seed;
	double worldX, worldY;
	switch (m_dir) {
	case SliceDirection::Inline:
		seed.x = point.y();
		seed.y = point.x();
		seed.z = m_currentSlice;
		m_data->seismic()->ijToInlineXlineTransfoForInline()->imageToWorld(point.x(), point.y(), worldX, worldY);
		break;
	case SliceDirection::XLine:
		seed.x = point.y();
		seed.y = m_currentSlice;
		seed.z = point.x();
		m_data->seismic()->ijToInlineXlineTransfoForXline()->imageToWorld(point.x(), point.y(), worldX, worldY);
		break;
	}

	// get rgt
	// buffer is transposed and point too
	seed.rgtValue = getValueFromSliceRepForSeed(m_rgt, point.x(), point.y(), m_data->channelRgt());
	seed.seismicValue = getValueFromSliceRepForSeed(m_seismic, point.x(), point.y(), m_data->channelSeismic());
	return m_data->addPoint(seed);
}

std::size_t MultiSeedRgtSliceRep::addPoint(int x, int y) { // add seed on current section
	return addPoint(QPoint(x, y));
}

std::size_t MultiSeedRgtSliceRep::addPointAndSelect(QPoint point) {
	std::size_t id = addPoint(point);
	if (id!=MultiSeedHorizon::INVALID_ID) {
		m_data->selectPoint(id);
	}
	return id;
}

std::size_t MultiSeedRgtSliceRep::addPointAndSelect(int x, int y) {
	return addPointAndSelect(QPoint(x, y));
}

void MultiSeedRgtSliceRep::moveSelectedPoint(QPointF point) {
	initSliceReps();

	IMouseImageDataProvider::MouseInfo info;
	if (m_rgt->mouseData(point.x(), point.y(), info) && info.values.size() < 1) {
		return;
	}

	double imageX, imageY;
	switch (m_dir) {
	case SliceDirection::Inline:
		m_data->seismic()->ijToInlineXlineTransfoForInline()->worldToImage(point.x(), point.y(), imageX, imageY);
		break;
	case SliceDirection::XLine:
		m_data->seismic()->ijToInlineXlineTransfoForXline()->worldToImage(point.x(), point.y(), imageX, imageY);
		break;
	}

	QPoint pt(imageX, imageY);// = m_rgtVisual->sceneToVisual(point);
	moveSelectedPoint(pt);
}

void MultiSeedRgtSliceRep::moveSelectedPoint(QPoint point) { // move point and update
	// care point is transposed
	if (m_data->getSelectedId()==MultiSeedHorizon::INVALID_ID) {
		return;
	}
	initSliceReps();
	long dimI = m_data->seismic()->height();
	long dimJ;
	if (m_dir==SliceDirection::Inline) { // Z
		dimJ = m_data->seismic()->width();
	} else if(m_dir==SliceDirection::XLine) { // Y
		dimJ = m_data->seismic()->depth();
	} else {
		return;
	}
	if ((point.x()<0 || point.x()>=dimJ) && (point.y()<0 && point.y()>=dimI)) {
		return;
	}
	RgtSeed seed;
	double worldX, worldY;
	switch (m_dir) {
	case SliceDirection::Inline:
		seed.x = point.y();
		seed.y = point.x();
		seed.z = m_currentSlice;
		m_data->seismic()->ijToInlineXlineTransfoForInline()->imageToWorld(point.x(), point.y(), worldX, worldY);
		break;
	case SliceDirection::XLine:
		seed.x = point.y();
		seed.y = m_currentSlice;
		seed.z = point.x();
		m_data->seismic()->ijToInlineXlineTransfoForXline()->imageToWorld(point.x(), point.y(), worldX, worldY);
		break;
	}

	// get rgt
	// buffer is transposed and point too
	seed.rgtValue = getValueFromSliceRepForSeed(m_rgt, point.x(), point.y(), m_data->channelRgt());
	seed.seismicValue = getValueFromSliceRepForSeed(m_seismic, point.x(), point.y(), m_data->channelSeismic());

	if (m_data->getDTauPolygonMainCache()!=0) {
		m_data->applyDtauToSeeds(); //applyDTauToData();
		m_polygonMainCache.clear();
	}

	m_data->moveSelectedSeed(seed);
}

void MultiSeedRgtSliceRep::moveSelectedPoint(int x, int y) {
	moveSelectedPoint(QPoint(x, y));
}

void MultiSeedRgtSliceRep::updateSeedsRepresentation() {
	if (m_layer) {
		m_layer->refresh();
	}
}

// graphic
void MultiSeedRgtSliceRep::updateGraphicRepresentation() { // ui update
	if (m_layer) {
		m_layer->refresh();
	}
}

void MultiSeedRgtSliceRep::setReferences(std::vector<float*> ref)
{
	m_reference = ref;
}



template<typename RgtType>
struct UpdateMainHorizonKernel {
	template<typename SeismicType>
	struct UpdateMainHorizonKernelLevel2 {
		static void run(MultiSeedSliceRep* ext, const void* _rgtBuf, int channelRgt, const void* _seismicBuf, int channelSeismic,
						long dimx, long dimy, std::map<std::size_t, RgtSeed> seeds, FixedLayerFromDataset* constrainLayer,
						std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers, QPolygon& poly, bool useSnap,
						bool useMedian, int lwx, int slice, SliceDirection dir, int distancePower,
						int snapWindow, int polarity, float tdeb, float pasech, long dtauReference,
						std::vector<double>& weights, std::vector<double>& means, std::vector<int>& staticPointCurve,
						std::vector<int>& selectedIndexCurve) {

			const RgtType* rgtBuf = static_cast<const RgtType*>(_rgtBuf);
			const SeismicType* seismicBuf = static_cast<const SeismicType*>(_seismicBuf);

			// cannot be used because buffer is locked -> deadlock issue
			//long dimx = rgtData->height();
			//long dimy = rgtData->width();

			std::vector<RgtSeed> seedsVec;
			seedsVec.resize(seeds.size());
			int i=0;
			std::size_t idFirstSeed;
			if (seeds.size()>0) {
				idFirstSeed = seeds.begin()->first;
			}
			for (const std::pair<std::size_t, RgtSeed> e : seeds) {
				seedsVec[i] = e.second;
				if (seeds.size()>0) {
					if (idFirstSeed>e.first) {
						idFirstSeed = e.first;
					}
				}
				i++;
			}

			weights.resize(dimy);
			means.resize(dimy);
			staticPointCurve.resize(dimy, 0);
			selectedIndexCurve.resize(dimy, 0);

			int type = polarity;

			bool isReferenceLayerSet = false;
			std::vector<ReferenceDuo> referenceLayersVec;
			std::vector<int> referenceValues;
			//std::vector<float> referenceLayerVector;
			//std::vector<float> referenceLayerRgtPropVector;

			QString rgtName("rgt");
			if (false  referenceLayers.size()!=0 ) {
				ReferenceDuo init;
				referenceLayersVec.resize(referenceLayers.size(), init);
				QString isoName("isochrone");
				for (std::size_t index=0; index<referenceLayersVec.size(); index++) {
					ReferenceDuo& pair = referenceLayersVec[index];
					pair.iso.resize(referenceLayers[0]->getNbTraces()*referenceLayers[0]->getNbProfiles());
					isReferenceLayerSet = referenceLayers[index]->readProperty(pair.iso.data(), isoName);
					if (isReferenceLayerSet) {
						pair.rgt.resize(referenceLayers[0]->getNbTraces()*referenceLayers[0]->getNbProfiles());
						isReferenceLayerSet = referenceLayers[index]->readProperty(pair.rgt.data(), rgtName);
					}
					if (!isReferenceLayerSet) {
						referenceLayersVec.clear();
						break;
					}
				}
			}

			if (isReferenceLayerSet) {
				referenceValues.resize(referenceLayers.size());
				for (std::size_t index=0; index<referenceLayersVec.size(); index++) {
					ReferenceDuo& pair = referenceLayersVec[index];
					if (dir==SliceDirection::Inline) {
						std::vector<float> tmpTab, tmpTab2;
						tmpTab.resize(referenceLayers[0]->getNbTraces(), 0);
						tmpTab2.resize(referenceLayers[0]->getNbTraces(), 0);
						for (long iy=0; iy<referenceLayers[0]->getNbTraces(); iy++) {
							long ix = (pair.iso[slice*referenceLayers[0]->getNbTraces()+iy] - tdeb) / pasech;
							//pair.rgt[slice*referenceLayers[0]->getNbTraces()+iy] = rgtBuf[ix*dimy+iy];
							tmpTab[iy] = rgtBuf[ix*dimy+iy + channelRgt * dimy * dimx];
						}
						UtFiltreMeanX(tmpTab.data(), tmpTab2.data(), tmpTab.size(), 1, 21);
						for (long iy=0; iy<referenceLayers[0]->getNbTraces(); iy++) {
							pair.rgt[slice*referenceLayers[0]->getNbTraces()+iy] = tmpTab2[iy];
						}
						referenceValues[index] = pair.rgt[slice*referenceLayers[0]->getNbTraces()+referenceLayers[0]->getNbTraces()/2];
					} else if (dir==SliceDirection::XLine) {
						std::vector<float> tmpTab, tmpTab2;
						tmpTab.resize(referenceLayers[0]->getNbProfiles(), 0);
						tmpTab2.resize(referenceLayers[0]->getNbProfiles(), 0);
						for (long iz=0; iz<referenceLayers[0]->getNbProfiles(); iz++) {
							long ix = (pair.iso[iz*referenceLayers[0]->getNbTraces()+slice] - tdeb) / pasech;
							//pair.rgt[iz*referenceLayers[0]->getNbTraces()+slice] = rgtBuf[ix*dimy+iz];
							tmpTab[iz] = rgtBuf[ix*dimy+iz + channelRgt * dimy * dimx];
						}
						UtFiltreMeanX(tmpTab.data(), tmpTab2.data(), tmpTab.size(), 1, 21);
						for (long iz=0; iz<referenceLayers[0]->getNbProfiles(); iz++) {
							pair.rgt[iz*referenceLayers[0]->getNbTraces()+slice] = tmpTab2[iz];
						}
						referenceValues[index] = pair.rgt[(referenceLayers[0]->getNbProfiles()/2)*referenceLayers[0]->getNbTraces()+slice];
					} else {
						qDebug() << "MultiSeedHorizonExtension : invalid orientation";
					}
					referenceLayers[index]->writeProperty(pair.rgt.data(), rgtName);
				}

				for (std::size_t index=0; index<seedsVec.size(); index++) {
					RgtSeed& seed = seedsVec[index];
					qDebug() << "Before " << seed.rgtValue;

					seed.rgtValue = getNewRgtValueFromReference(seed.y, seed.z, seed.x, seed.rgtValue, tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues);

					qDebug() << "After " << seed.rgtValue;
				}
			}

			std::sort(seedsVec.begin(), seedsVec.end(), [](RgtSeed a, RgtSeed b){
				return a.rgtValue<b.rgtValue;
			});

			bool isConstrainLayerSet = false;
			std::vector<float> constrainLayerVector;
			if (constrainLayer!=nullptr) {
				constrainLayerVector.resize(constrainLayer->getNbTraces()*constrainLayer->getNbProfiles());
				QString isoName("isochrone");
				isConstrainLayerSet = constrainLayer->readProperty(constrainLayerVector.data(), isoName);
			}


			std::vector<int> points;
			points.resize(dimy);

	//#pragma omp parallel for schedule (dynamic)
			for (int y=0; y<dimy; y++) {
				std::vector<double> dist;
				dist.resize(seedsVec.size());

				double som=0.0 ;
				long iy, iz;
				if (dir==SliceDirection::XLine) {
					iz = y;
					iy = slice;
				} else {
					iz = slice;
					iy = y;
				}

				long ix, foundIx;
				bool seedFound = false;
				if (isConstrainLayerSet && constrainLayerVector[iy+iz*constrainLayer->getNbTraces()]!=-9999) {
					long ixOri = (constrainLayerVector[iy+iz*constrainLayer->getNbTraces()]-tdeb)/pasech;
					ix = ixOri;
					if (dtauReference>0) {
						ix = std::max(ix, 0l);
						while (ix<dimx && rgtBuf[ix*dimy+y + channelRgt * dimy * dimx]<rgtBuf[ixOri*dimy+y + channelRgt * dimy * dimx]+dtauReference) {
							ix++;
						}
						ix = std::min(ix, dimx-1);
					} else if(dtauReference<0) {
						ix = std::min(ix, dimx);
						long oldX = ix;
						while (ix>=0 && rgtBuf[ix*dimy+y + channelRgt * dimy * dimx]>rgtBuf[ixOri*dimy+y + channelRgt * dimy * dimx]+dtauReference) {
							ix--;
						}
						if (ix<dimx-1 && rgtBuf[ix*dimy+y + channelRgt * dimy * dimx]<rgtBuf[ixOri*dimy+y + channelRgt * dimy * dimx]+dtauReference) {
							ix++;
						}

					}
					seedFound = true;
					foundIx = ix;
					staticPointCurve[y] += 1;
				} else {
					for(int i=0; (i < seedsVec.size()); i++) {
						long val = ((iy - seedsVec[i].y)*(iy-seedsVec[i].y) + (iz -seedsVec[i].z)*(iz-seedsVec[i].z));
						if (val!=0) {
							dist[i] = 1.0 / std::pow(val,distancePower/2.0) ;
							som += dist[i] ;
						} else {
							dist[i] = 0;
							ix = seedsVec[i].x;
							seedFound = true;
							foundIx = ix;
							staticPointCurve[y] += 1;
						}
					}
				}



				double ixDouble = 0 ;
				ix = 1;
				double weightedIso = 0;
				for(int i=0; i < seedsVec.size() ; i++) {
					if (isReferenceLayerSet) {
						while ( getNewRgtValueFromReference(iy, iz, ix, rgtBuf[y + ix*dimy + channelRgt * dimy * dimx], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues)  < seedsVec[i].rgtValue  && ix < dimx ) {
							ix ++ ;
						}
						if (ix==dimx) {
							ix = dimx-1;
						}
						double ix_rgt = getNewRgtValueFromReference(iy, iz, ix, rgtBuf[y + ix*dimy + channelRgt * dimy * dimx], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues);
						if (ix_rgt==seedsVec[i].rgtValue || ix==0) {
							ixDouble = ix;
						} else {
							double ix_floor_rgt = getNewRgtValueFromReference(iy, iz, ix-1, rgtBuf[y + (ix-1)*dimy + channelRgt * dimy * dimx], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues);
							ixDouble = ix-1 + (seedsVec[i].rgtValue - ix_floor_rgt) / (ix_rgt - ix_floor_rgt);
							if (ixDouble>ix) {
								ixDouble = ix;
							} else if (ixDouble<ix-1) {
								ixDouble = ix-1;
							}
						}
					} else {
						while ( ix < dimx && rgtBuf[ix*dimy + y + channelRgt * dimy * dimx]  < seedsVec[i].rgtValue ) {
							ix ++ ;
						}
						if (ix==dimx) {
							ix = dimx-1;
						}
						if (rgtBuf[ix*dimy + y + channelRgt * dimy * dimx]==seedsVec[i].rgtValue || ix==0) {
							ixDouble = ix;
						} else {
							double ix_floor_rgt = rgtBuf[(ix-1)*dimy + y + channelRgt * dimy * dimx];
							double ix_rgt = rgtBuf[ix*dimy + y + channelRgt * dimy * dimx];
							ixDouble = ix-1 + (seedsVec[i].rgtValue - ix_floor_rgt) / (ix_rgt - ix_floor_rgt);

							if (ixDouble>ix) {
									ixDouble = ix;
							} else if (ixDouble<ix-1) {
									ixDouble = ix-1;
							}

						}
					}

					weightedIso += ixDouble*dist[i] ;
				}
				if(!seedFound) {
					if (som!=0) {
						points[y] = weightedIso/som ;
					} else {
						points[y] = 0;
					}
					staticPointCurve[y] = 0;
				} else{
					points[y] = foundIx;
				}
				weights[y] = som;
				means[y] = weightedIso;
				selectedIndexCurve[y] = points[y];
			}

			if (useSnap) {
				// snap
				for (int y=0; y<dimy; y++) {
					int x = points[y];
					std::vector<SeismicType> traceBuf;
					traceBuf.resize(dimx);
					for (std::size_t index=0; index<dimx; index++) {
						traceBuf[index] = seismicBuf[y+index*dimy + channelSeismic * dimy * dimx];
					}
					int newx = bl_indpol(x, traceBuf.data(), dimx, type, snapWindow);
					points[y] = (newx==SLOPE::RAIDE)? x : newx;
				}
			}

			if (useMedian) {
				// apply median
				UtFiltreMedianeX(points.data(), points.size(), 1, lwx);
			}

			for (int y=0; y<dimy; y++) {
				poly << QPoint(y, points[y]); // points are transposed
			}
		}
	};

	static void run(ImageFormats::QSampleType seismicType, MultiSeedRgtSliceRep* ext, const void* _rgtBuf, int channelRgt, const void* _seismicBuf, int channelSeismic,
			long dimx, long dimy, std::map<std::size_t, RgtSeed> seeds, FixedLayerFromDataset* constrainLayer,
			std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers, QPolygon& poly, bool useSnap,
			bool useMedian, int lwx, int slice, SliceDirection dir, int distancePower,
			int snapWindow, int polarity, float tdeb, float pasech, long dtauReference,
			std::vector<double>& weights, std::vector<double>& means, std::vector<int>& staticPointCurve,
			std::vector<int>& selectedIndexCurve) {
		SampleTypeBinder binder(seismicType);
		binder.bind<UpdateMainHorizonKernelLevel2>(ext, _rgtBuf, channelRgt, _seismicBuf, channelSeismic,
				dimx, dimy, seeds, constrainLayer,
				referenceLayers, poly, useSnap,
				useMedian, lwx, slice, dir, distancePower,
				snapWindow, polarity, tdeb, pasech, dtauReference,
				weights, means, staticPointCurve,
				selectedIndexCurve);
	}
};

template<typename RgtType>
struct UpdateMainHorizonWithCacheKernel {
	template<typename SeismicType>
	struct UpdateMainHorizonWithCacheKernelLevel2 {
		static void run(MultiSeedRgtSliceRep* ext, const void* rgtData, int channelRgt, const void* seismicData, int channelSeismic,
						long dimx, long dimy, std::vector<RgtSeed> newSeeds, std::vector<RgtSeed> removedSeeds,
						std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers, QPolygon& poly, bool useSnap,
						bool useMedian, int lwx, int slice, SliceDirection dir, int distancePower,
						int snapWindow, int polarity, float tdeb, float pasech, long dtauReference,
						std::vector<double>& weights, std::vector<double>& means, std::vector<int>& staticPointCurve,
						std::vector<int>& selectedIndexCurve) {
			const RgtType* rgtBuf = static_cast<const RgtType*>(rgtData);
			const SeismicType* seismicBuf = static_cast<const SeismicType*>(seismicData);

			// CUDAImagePaletteHolder is transposed
			//long dimx = rgtData->height();
			//long dimy = rgtData->width();

			int i=0;

			int type = polarity;

			std::vector<ReferenceDuo> referenceLayersVec;
			std::vector<int> referenceValues;
			bool isReferenceLayerSet = false;
			QString rgtName("rgt");
			if (false referenceLayers.size()!=0 ) {
				ReferenceDuo init;
				referenceLayersVec.resize(referenceLayers.size(), init);
				QString isoName("isochrone");
				for (std::size_t index=0; index<referenceLayersVec.size(); index++) {
					ReferenceDuo& pair = referenceLayersVec[index];
					pair.iso.resize(referenceLayers[0]->getNbTraces()*referenceLayers[0]->getNbProfiles());
					isReferenceLayerSet = referenceLayers[index]->readProperty(pair.iso.data(), isoName);
					if (isReferenceLayerSet) {
						pair.rgt.resize(referenceLayers[0]->getNbTraces()*referenceLayers[0]->getNbProfiles());
						isReferenceLayerSet = referenceLayers[index]->readProperty(pair.rgt.data(), rgtName);
					}
					if (!isReferenceLayerSet) {
						referenceLayersVec.clear();
						break;
					}
				}

			}

			if (isReferenceLayerSet) {
				referenceValues.resize(referenceLayers.size());
				for (std::size_t index=0; index<referenceLayersVec.size(); index++) {
					ReferenceDuo& pair = referenceLayersVec[index];
					if (dir==SliceDirection::Inline) {
						std::vector<float> tmpTab, tmpTab2;
						tmpTab.resize(referenceLayers[0]->getNbTraces(), 0);
						tmpTab2.resize(referenceLayers[0]->getNbTraces(), 0);
						for (long iy=0; iy<referenceLayers[0]->getNbTraces(); iy++) {
							long ix = (pair.iso[slice*referenceLayers[0]->getNbTraces()+iy] - tdeb) / pasech;
							//pair.rgt[slice*referenceLayers[0]->getNbTraces()+iy] = rgtBuf[ix*dimy+iy];
							tmpTab[iy] = rgtBuf[ix*dimy+iy + channelRgt*dimy*dimx];
						}
						UtFiltreMeanX(tmpTab.data(), tmpTab2.data(), tmpTab.size(), 1, 21);
						for (long iy=0; iy<referenceLayers[0]->getNbTraces(); iy++) {
							pair.rgt[slice*referenceLayers[0]->getNbTraces()+iy] = tmpTab2[iy];
						}
						referenceValues[index] = pair.rgt[slice*referenceLayers[0]->getNbTraces()+referenceLayers[0]->getNbTraces()/2];
					} else if (dir==SliceDirection::XLine) {
						std::vector<float> tmpTab, tmpTab2;
						tmpTab.resize(referenceLayers[0]->getNbProfiles(), 0);
						tmpTab2.resize(referenceLayers[0]->getNbProfiles(), 0);
						for (long iz=0; iz<referenceLayers[0]->getNbProfiles(); iz++) {
							long ix = (pair.iso[iz*referenceLayers[0]->getNbTraces()+slice] - tdeb) / pasech;
							//pair.rgt[iz*referenceLayers[0]->getNbTraces()+slice] = rgtBuf[ix*dimy+iz];
							tmpTab[iz] = rgtBuf[ix*dimy+iz + channelRgt*dimy*dimx];
						}
						UtFiltreMeanX(tmpTab.data(), tmpTab2.data(), tmpTab.size(), 1, 21);
						for (long iz=0; iz<referenceLayers[0]->getNbProfiles(); iz++) {
							pair.rgt[iz*referenceLayers[0]->getNbTraces()+slice] = tmpTab2[iz];
						}
						referenceValues[index] = pair.rgt[(referenceLayers[0]->getNbProfiles()/2)*referenceLayers[0]->getNbTraces()+slice];
					} else {
						qDebug() << "MultiSeedHorizonExtension : invalid orientation";
					}
					referenceLayers[index]->writeProperty(pair.rgt.data(), rgtName);
				}

				for (std::size_t index=0; index<newSeeds.size(); index++) {
					RgtSeed& seed = newSeeds[index];

					seed.rgtValue = getNewRgtValueFromReference(seed.y, seed.z, seed.x, seed.rgtValue, tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues);
				}

				for (std::size_t index=0; index<removedSeeds.size(); index++) {
					RgtSeed& seed = removedSeeds[index];

					seed.rgtValue = getNewRgtValueFromReference(seed.y, seed.z, seed.x, seed.rgtValue, tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues);
				}
			}

			std::sort(newSeeds.begin(), newSeeds.end(), [](RgtSeed a, RgtSeed b){
				return a.rgtValue<b.rgtValue;
			});

			std::sort(removedSeeds.begin(), removedSeeds.end(), [](RgtSeed a, RgtSeed b){
				return a.rgtValue<b.rgtValue;
			});
			std::vector<int> points;
			points.resize(dimy);

	#pragma omp parallel for schedule (dynamic)
			for (int y=0; y<dimy; y++) {

				std::vector<double> newDist, removedDist;
				newDist.resize(newSeeds.size());
				removedDist.resize(removedSeeds.size());
				double som=0.0 ;
				long iy, iz;
				if (dir==SliceDirection::XLine) {
					iz = y;
					iy = slice;
				} else {
					iz = slice;
					iy = y;
				}

				for(int i=0; (i < removedSeeds.size()); i++) {
					long val = ((iy - removedSeeds[i].y)*(iy-removedSeeds[i].y) + (iz -removedSeeds[i].z)*(iz-removedSeeds[i].z));
					if (val!=0) {
						removedDist[i] = 1.0 / std::pow(val,distancePower/2.0) ;
						som -= removedDist[i] ;
					} else {
						removedDist[i] = 0;
						staticPointCurve[y] -= 1;
					}
				}

				long ix, foundIx;
				bool oriSeedFound = staticPointCurve[y]>0;
				bool seedFound = oriSeedFound;
				if (seedFound) {
					ix = selectedIndexCurve[y];
				} else if (staticPointCurve[y]<0) {
					staticPointCurve[y] = 0;
					qDebug() << "MutliSeedHorizonExtension Warning : unexpected state";
				}
				for(int i=0; (i < newSeeds.size()); i++) {
					long val = ((iy - newSeeds[i].y)*(iy-newSeeds[i].y) + (iz -newSeeds[i].z)*(iz-newSeeds[i].z));
					if (val!=0) {
						newDist[i] = 1.0 / std::pow(val,distancePower/2.0) ;
						som += newDist[i] ;
					} else {
						newDist[i] = 0;
						ix = newSeeds[i].x;
						foundIx = newSeeds[i].x;
						seedFound = true;
						staticPointCurve[y] += 1;
					}
				}



				ix = 1 ;
				double ixDouble = 0;
				double weightedIso = 0;
				for(int i=0; i < newSeeds.size() ; i++) {
					if (isReferenceLayerSet) {
						while ( getNewRgtValueFromReference(iy, iz, ix, rgtBuf[y + ix*dimy + channelRgt*dimy*dimx], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues)  < newSeeds[i].rgtValue  && ix < dimx ) {
							ix ++ ;
						}
						if (ix==dimx) {
							ix = dimx-1;
						}
						double ix_rgt = getNewRgtValueFromReference(iy, iz, ix, rgtBuf[y + ix*dimy + channelRgt*dimy*dimx], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues);
						if (ix==0 || ix_rgt== newSeeds[i].rgtValue) {
							ixDouble = ix;
						} else {
							double ix_floor_rgt = getNewRgtValueFromReference(iy, iz, ix-1, rgtBuf[y + (ix-1)*dimy + channelRgt*dimy*dimx], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues);
							ixDouble = ix-1 + (newSeeds[i].rgtValue - ix_floor_rgt) / (ix_rgt - ix_floor_rgt);

							if (ixDouble>ix) {
								ixDouble = ix;
							} else if (ixDouble<ix-1) {
								ixDouble = ix-1;
							}
						}
					} else {
						while ( rgtBuf[ix*dimy + y + channelRgt*dimy*dimx]  < newSeeds[i].rgtValue  && ix < dimx ) {
							ix ++ ;
						}
						if (ix==dimx) {
							ix = dimx-1;
						}
						if (ix==0 || rgtBuf[ix*dimy + y + channelRgt*dimy*dimx]== newSeeds[i].rgtValue) {
							ixDouble = ix;
						} else {
							double ix_floor_rgt = rgtBuf[y+(ix-1)*dimy + channelRgt*dimy*dimx];
							double ix_rgt = rgtBuf[ix*dimy+y + channelRgt*dimy*dimx];
							ixDouble = ix-1 + (newSeeds[i].rgtValue - ix_floor_rgt) / (ix_rgt - ix_floor_rgt);

							if (ixDouble>ix) {
								ixDouble = ix;
							} else if (ixDouble<ix-1) {
								ixDouble = ix-1;
							}
						}
					}
					weightedIso += ixDouble*newDist[i] ;
				}
				ix = 1;
				ixDouble = 0;
				for(int i=0; i < removedSeeds.size() ; i++) {
					if (isReferenceLayerSet) {
						while (  getNewRgtValueFromReference(iy, iz, ix, rgtBuf[y + ix*dimy + channelRgt*dimy*dimx], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues) < removedSeeds[i].rgtValue  && ix < dimx ) {
							ix ++ ;
						}
						if (ix==dimx) {
							ix = dimx-1;
						}
						double ix_rgt = getNewRgtValueFromReference(iy, iz, ix, rgtBuf[y + ix*dimy + channelRgt*dimy*dimx], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues);
						if (ix==0 || ix_rgt== removedSeeds[i].rgtValue) {
							ixDouble = ix;
						} else {
							double ix_floor_rgt = getNewRgtValueFromReference(iy, iz, ix-1, rgtBuf[y + (ix-1)*dimy + channelRgt*dimy*dimx], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues);
							ixDouble = ix-1 + (removedSeeds[i].rgtValue - ix_floor_rgt) / (ix_rgt - ix_floor_rgt);

							if (ixDouble>ix) {
								ixDouble = ix;
							} else if (ixDouble<ix-1) {
								ixDouble = ix-1;
							}
						}
					} else {
						while ( rgtBuf[ix*dimy + y + channelRgt*dimy*dimx]  < removedSeeds[i].rgtValue  && ix < dimx ) {
							ix ++ ;
						}
						if (ix==dimx) {
							ix=dimx-1;
						}
						if (ix==0 || rgtBuf[ix*dimy + y + channelRgt*dimy*dimx]== removedSeeds[i].rgtValue) {
							ixDouble = ix;
						} else {
							double ix_floor_rgt = rgtBuf[y+dimy*(ix-1) + channelRgt*dimy*dimx];
							double ix_rgt = rgtBuf[ix*dimy+y + channelRgt*dimy*dimx];
							ixDouble = ix-1 + (removedSeeds[i].rgtValue - ix_floor_rgt) / (ix_rgt - ix_floor_rgt);

							if (ixDouble>ix) {
								ixDouble = ix;
							} else if (ixDouble<ix-1) {
								ixDouble = ix - 1;
							}
						}
					}
					weightedIso -= ixDouble*removedDist[i] ;
				}

				if ( som + weights[y] >0.5) {
					int as = 1;
				}

				som += weights[y];
				weightedIso += means[y];
				if(!seedFound) {
					if (som!=0) {
						points[y] = weightedIso/som ;
					} else {
						points[y] = 0;
					}
					staticPointCurve[y] = 0;
				} else if(oriSeedFound) {
					points[y] = selectedIndexCurve[y];
				} else {
					points[y] = foundIx;
				}
				weights[y] = som;
				means[y] = weightedIso;
				selectedIndexCurve[y] = points[y];
			}

			if (useSnap) {
				// snap
				for (int y=0; y<dimy; y++) {
					int x = points[y];
					std::vector<SeismicType> traceBuf;
					traceBuf.resize(dimx);
					for (std::size_t index=0; index<dimx; index++) {
						traceBuf[index] = seismicBuf[y+index*dimy + channelSeismic*dimx*dimy];
					}
					int newx = bl_indpol(x, traceBuf.data(), dimx, type, snapWindow);
					points[y] = (newx==SLOPE::RAIDE)? x : newx;
				}
			}

			if (useMedian) {
				// apply median
				UtFiltreMedianeX(points.data(), points.size(), 1, lwx);
			}

			for (int y=0; y<dimy; y++) {
				poly << QPoint(y, points[y]); // points are transposed
			}
		}
	};

	static void run(ImageFormats::QSampleType seismicType, MultiSeedRgtSliceRep* ext, const void* rgtData, int channelRgt, const void* seismicData, int channelSeismic,
			long dimx, long dimy, std::vector<RgtSeed>& newSeeds, std::vector<RgtSeed>& removedSeeds,
			std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers, QPolygon& poly, bool useSnap,
			bool useMedian, int lwx, int slice, SliceDirection dir, int distancePower,
			int snapWindow, int polarity, float tdeb, float pasech, long dtauReference,
			std::vector<double>& weights, std::vector<double>& means, std::vector<int>& staticPointCurve,
			std::vector<int>& selectedIndexCurve) {
		SampleTypeBinder binder(seismicType);
		binder.bind<UpdateMainHorizonWithCacheKernelLevel2>(ext, rgtData, channelRgt, seismicData, channelSeismic,
				dimx, dimy, newSeeds, removedSeeds,
				referenceLayers, poly, useSnap,
				useMedian, lwx, slice, dir, distancePower,
				snapWindow, polarity, tdeb, pasech, dtauReference,
				weights, means, staticPointCurve,
				selectedIndexCurve);
	}
};

void MultiSeedRgtSliceRep::updateMainHorizon() { // polygons update
	initSliceReps();

	long width = m_data->seismic()->width();
	long depth = m_data->seismic()->depth();
	long dimx = m_data->seismic()->height();
	long dimy;
	if (m_dir==SliceDirection::Inline) { // Z
		dimy = width;
	} else if(m_dir==SliceDirection::XLine) { // Y
		dimy = depth;
	} else {
		return;
	}

	CUDAImagePaletteHolder* rgtData = m_rgt->image();
	CUDAImagePaletteHolder* seismicData = m_seismic->image();

	if (rgtData->width()!=seismicData->width() || rgtData->height()!=seismicData->height() ||
		dimx!=rgtData->height() || dimy!=rgtData->width()) {
		// error
		qDebug() << "ERROR : ViewHorizon : fillPolygonFromRGTValue datas do not match";
		return;
	}



	// only support short
	// buffer are transposed : y is the quickest, pos = x * dimy + y


	if (m_data->getDTauPolygonMainCache()!=0) {
//		applyDTauToData();
		m_data->applyDtauToSeeds(this);
		m_polygonMainCache.clear();
		clearCurveSpeedupCache();
	} else if (m_polygonMainCache.size()!=0) {
		m_polygonMainCache.clear();
		clearCurveSpeedupCache();
	}

	m_polygonMain.clear();

	correctSeedsFromImage();

	const void* rgtBuf = static_cast<const void*>(m_rgt->lockCache().data());
	const void* seismicBuf = static_cast<const void*>(m_seismic->lockCache().data());

	float tdeb = m_data->seismic()->sampleTransformation()->b();
	float pasech = m_data->seismic()->sampleTransformation()->a();

	SampleTypeBinder binder(m_rgt->image()->sampleType());
	std::vector<std::shared_ptr<FixedLayerFromDataset>> references = m_data->getReferences();
	if (m_meansCurve.size()==dimy && m_weightsCurve.size()==dimy && m_staticPointCurve.size()==dimy &&
			m_selectedIndexCurve.size()==dimy) {
		binder.bind< UpdateMainHorizonWithCacheKernel> (m_seismic->image()->sampleType(), this, rgtBuf, m_data->channelRgt(), seismicBuf, m_data->channelSeismic(), dimx, dimy, m_newPoints, m_removedPoints, references,
			m_polygonMain, m_data->useSnap(), m_data->useMedian(), m_data->getLWXMedianFilter(),  m_currentSlice,  m_dir, m_data->getDistancePower(), m_data->getSnapWindow(),
			m_data->getPolarity(), tdeb, pasech,
			m_data->getDTauReference(), m_weightsCurve, m_meansCurve, m_staticPointCurve, m_selectedIndexCurve);
		m_newPoints.clear();
		m_removedPoints.clear();
	} else {
		clearCurveSpeedupCache();
		binder.bind<UpdateMainHorizonKernel> (m_seismic->image()->sampleType(), this, rgtBuf, m_data->channelRgt(), seismicBuf, m_data->channelSeismic(), dimx, dimy, m_data->getMap(), m_data->constrainLayer(), references,
			m_polygonMain, m_data->useSnap(), m_data->useMedian(), m_data->getLWXMedianFilter(),  m_currentSlice,  m_dir, m_data->getDistancePower(), m_data->getSnapWindow(),
			m_data->getPolarity(), tdeb, pasech,
			m_data->getDTauReference(), m_weightsCurve, m_meansCurve, m_staticPointCurve, m_selectedIndexCurve);
	}

	for (int n=0; n<m_polygonReference.size(); n++)
	{
		m_polygonReference[n].clear();
	}

//	std::vector<std::shared_ptr<FixedLayerFromDataset>> references = m_data->getReferences();
	int N = references.size();
	m_polygonReference.resize(N);
	for (int n=0; n<N; n++)
	{
		//float *data = references[n];
		CUDAImagePaletteHolder* image = references[n]->image(FixedLayerFromDataset::ISOCHRONE);
		image->lockPointer();
		float* data = static_cast<float*>(image->backingPointer()); // image from FixedLayerFromDataset is always float else use readProperty with a buffer
		if ( m_dir==SliceDirection::Inline )
		{
			QPolygon currentPolygon;
			for (int y=0; y<width; y++)
			{
//				m_polygonReference[n] << QPoint(y, (int)((data[width*m_currentSlice+y]-tdeb)/pasech));
				if (data[width*m_currentSlice+y]!=-9999.0) {
					currentPolygon << QPoint(y, (int)((data[width*m_currentSlice+y]-tdeb)/pasech));
				} else if (currentPolygon.size()!=0) {
					m_polygonReference[n].push_back(currentPolygon);
					currentPolygon.clear();
				}
			}
			if (currentPolygon.size()!=0) {
				m_polygonReference[n].push_back(currentPolygon);
			}
		}
		else
		{
			QPolygon currentPolygon;
			for (int z=0; z<depth; z++)
			{
//				m_polygonReference[n] << QPoint(z, (int)((data[width*z+m_currentSlice]-tdeb)/pasech));
				if (data[width*z+m_currentSlice]!=-9999.0) {
					currentPolygon << QPoint(z, (int)((data[width*z+m_currentSlice]-tdeb)/pasech));
				} else if (currentPolygon.size()!=0) {
					m_polygonReference[n].push_back(currentPolygon);
					currentPolygon.clear();
				}
			}
			if (currentPolygon.size()!=0) {
				m_polygonReference[n].push_back(currentPolygon);
			}
		}
		image->unlockPointer();
	}


	m_rgt->unlockCache();
	m_seismic->unlockCache();

	if (m_data->getHorizonMode()==DELTA_T) {
		updateDeltaHorizon();
	} else {
		updateGraphicRepresentation();
	}
}

//void MultiSeedSliceRep::applyDTauToData() {
//	if (m_data->getDTauPolygonMainCache()!=0) {
//		clearCurveSpeedupCache();
//
//		std::vector<std::tuple<RgtSeed, RgtSeed, std::size_t>> dataForSignals;
//
//		long width = m_data->seismic()->width();
//		long depth = m_data->seismic()->depth();
//		long dimx = m_data->seismic()->height();
//		long dimy;
//		if (m_dir==SliceDirection::Inline) { // Z
//			dimy = width;
//		} else if(m_dir==SliceDirection::XLine) { // Y
//			dimy = depth;
//		} else {
//			return;
//		}
//
//		CUDAImagePaletteHolder* rgtData = m_rgt->image();
//		CUDAImagePaletteHolder* seismicData = m_seismic->image();
//		rgtData->lockPointer();
//		seismicData->lockPointer();
//
//		short* rgtBuf = static_cast<short*>(rgtData->backingPointer());
//		short* seismicBuf = static_cast<short*>(seismicData->backingPointer());
//
//		// apply delta tau for all seeds
//		long polarityVal = 0;
//		std::map<std::size_t, RgtSeed>::const_iterator it = m_data->getMap().cbegin();
//		while (it!=m_data->getMap().cend()) {
//			RgtSeed oldSeed = it->second;
//
//			long y, x;
//			if (m_dir==SliceDirection::Inline) {
//				y = it->second.y;
//			} else {
//				y = it->second.z;
//			}
//
//			// move point
//			x = it->second.x;
//			if (m_data->getDTauPolygonMainCache()>0) {
//				x = std::max(x, 0l);
//				while (x<dimx && rgtBuf[x*dimy+y]<it->second.rgtValue+m_data->getDTauPolygonMainCache()) {
//					x++;
//				}
//				x = std::min(x, dimx-1);
//			} else {
//				x = std::min(x, dimx);
//				long oldX = x;
//				while (x>=0 && rgtBuf[x*dimy+y]>it->second.rgtValue+m_data->getDTauPolygonMainCache()) {
//					x--;
//				}
//				if (x<dimx-1 && rgtBuf[x*dimy+y]<it->second.rgtValue+m_data->getDTauPolygonMainCache()) {
//					x++;
//				}
//
//			}
//
//			RgtSeed newSeed;
//
//			newSeed.x = x;
//			newSeed.seismicValue = seismicBuf[x*dimy+y];
//			newSeed.rgtValue = it->second.rgtValue + m_data->getDTauPolygonMainCache();
//			newSeed.y = it->second.y;
//			newSeed.z = it->second.z;
//
//			polarityVal += it->second.seismicValue;
//
//			// send signal
//			std::tuple<RgtSeed, RgtSeed, std::size_t> myTuple(oldSeed, newSeed, it->first);
//			dataForSignals.push_back(myTuple);
//			//emit pointMoved(oldSeed, it->second, it->first);
//			it++;
//		}
//
//		rgtData->unlockPointer();
//		seismicData->unlockPointer();
//
//		disconnect(m_data, &MultiSeedHorizon::polarityChanged, this, &MultiSeedSliceRep::updateMainHorizon);
//		m_data->setPolarity((polarityVal>=0) ? 1 : -1);
//		connect(m_data, &MultiSeedHorizon::polarityChanged, this, &MultiSeedSliceRep::updateMainHorizon);
//		// emit signals
////		disconnect(m_data, &MultiSeedHorizon::pointMoved, this, &MultiSeedSliceRep::pointMovedSynchro);
//
//		emit m_data->startApplyDTauTransaction(this);
//		for (std::tuple<RgtSeed, RgtSeed, std::size_t>& myTuple : dataForSignals) {
//			//emit pointMoved(std::get<0>(myTuple), std::get<1>(myTuple), std::get<2>(myTuple));
//			m_data->moveSeed(std::get<2>(myTuple), std::get<1>(myTuple));
//		}
//		m_data->setDTauPolygonMainCache(0);
//		emit m_data->endApplyDTauTransaction(this);
////		connect(m_data, &MultiSeedHorizon::pointMoved, this, &MultiSeedSliceRep::pointMovedSynchro);
//	}
//	m_polygonMainCache.clear();
//}

void MultiSeedRgtSliceRep::updateDeltaHorizon() {
	if (m_data->getHorizonMode()==DELTA_T) {
		m_polygonBottom.clear();
		m_polygonTop.clear();
		for (QPoint pt : m_polygonMain) {
			m_polygonTop << QPoint(pt.x(), pt.y() + m_data->getDeltaTop());
			m_polygonBottom << QPoint(pt.x(), pt.y() + m_data->getDeltaBottom());
		}
	}
	updateGraphicRepresentation();
}

QPen MultiSeedRgtSliceRep::getPen() const {
	return m_penMain;
}

void MultiSeedRgtSliceRep::setPen(const QPen& pen) {
	m_penMain = pen;
	updateGraphicRepresentation();
}

QPen MultiSeedRgtSliceRep::getPenDelta() const {
	return m_penDelta;
}

void MultiSeedRgtSliceRep::setPenDelta(const QPen& pen) {
	m_penDelta = pen;
	updateGraphicRepresentation();
}

QPen MultiSeedRgtSliceRep::getPenPoints() const {
	return m_pointsPen;
}

void MultiSeedRgtSliceRep::setPenPoints(const QPen& pen) {
	m_pointsPen = pen;
	updateSeedsRepresentation();
}

void MultiSeedRgtSliceRep::clearCurveSpeedupCache() {
	m_weightsCurve.clear();
	m_meansCurve.clear();
	m_staticPointCurve.clear();
	m_selectedIndexCurve.clear();
	m_newPoints.clear();
	m_removedPoints.clear();
}

void MultiSeedRgtSliceRep::newPointCreatedSynchro(RgtSeed seed, int id) {
	if (m_data->getMap().size()>1) {
		m_newPoints.push_back(seed);
	} else {
		clearCurveSpeedupCache();
	}
	updateMainHorizon();
	updateSeedsRepresentation();
}

void MultiSeedRgtSliceRep::pointRemovedSynchro(RgtSeed seed, int id) {
    //m_removedPoints.push_back(seed);
    clearCurveSpeedupCache();
    updateMainHorizon();
    updateSeedsRepresentation();
}

void MultiSeedRgtSliceRep::pointMovedSynchro(RgtSeed oldSeed, RgtSeed newSeed, int id) {
    //m_removedPoints.push_back(oldSeed);
    //m_newPoints.push_back(seed);
    clearCurveSpeedupCache();

    updateMainHorizon();
    updateSeedsRepresentation();
}

void MultiSeedRgtSliceRep::seedsResetSynchro() {
	clearCurveSpeedupCache();
	updateMainHorizon();
	updateSeedsRepresentation();
}

QString MultiSeedRgtSliceRep::name() const {
	return m_data->name();
}

void MultiSeedRgtSliceRep::setSliceIJPosition(int val) {
	QMutexLocker lock(&m_polygonMutex);
	m_currentSlice = val;
	clearCurveSpeedupCache();
	updateMainHorizon();
}

QGraphicsItem * MultiSeedRgtSliceRep::getOverlayItem(DataControler * controler,QGraphicsItem *parent) {
	return nullptr;
}

void MultiSeedRgtSliceRep::notifyDataControlerMouseMoved(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys) {
	if (m_data->getBehaviorMode()!=MOUSETRACKING) {
		return;
	}
	double imageX, imageY;
	switch (m_dir) {
	case SliceDirection::Inline:
		m_data->seismic()->ijToInlineXlineTransfoForInline()->worldToImage(worldX, worldY, imageX, imageY);
		break;
	case SliceDirection::XLine:
		m_data->seismic()->ijToInlineXlineTransfoForXline()->worldToImage(worldX, worldY, imageX, imageY);
		break;
	}
	long dimI = m_data->seismic()->height();
	long dimJ;
	if (m_dir==SliceDirection::Inline) { // Z
		dimJ = m_data->seismic()->width();
	} else if(m_dir==SliceDirection::XLine) { // Y
		dimJ = m_data->seismic()->depth();
	} else {
		return;
	}
	if ((imageX<0 || imageX>=dimJ) && (imageY<0 && imageY>=dimI)) {
		return;
	}

	QPoint pt(imageX, imageY);
	moveTrackingReference(pt);

}

void MultiSeedRgtSliceRep::notifyDataControlerMousePressed(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys) {
	if (m_data->getBehaviorMode()==POINTPICKING && (button & Qt::LeftButton)) {
		initSliceReps();
		double imageX, imageY;
		if (direction()==SliceDirection::Inline) {
			dynamic_cast<Seismic3DDataset*>(m_seismic->data())->ijToInlineXlineTransfoForInline()->worldToImage(worldX, worldY, imageX, imageY);
		} else {
			dynamic_cast<Seismic3DDataset*>(m_seismic->data())->ijToInlineXlineTransfoForXline()->worldToImage(worldX, worldY, imageX, imageY);
		}
		std::size_t index=0;

		if (imageX>=m_seismic->image()->width() || imageX<0 ||
			imageY>=m_seismic->image()->height() || imageY<0) {
			return;
		}

		if (m_data->isMultiSeed()) {
			addPointAndSelect(imageX, imageY);
		} else if (!m_data->isMultiSeed()) {
			QPoint imagePoint(imageX, imageY);
			if (m_data->getMap().size()==0) {
				addPointAndSelect(imagePoint);
			} else {
				moveSelectedPoint(imagePoint);
			}
		}
	}
}

void MultiSeedRgtSliceRep::notifyDataControlerMouseRelease(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys) {

}

void MultiSeedRgtSliceRep::notifyDataControlerMouseDoubleClick(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys) {

}

QGraphicsItem * MultiSeedRgtSliceRep::releaseOverlayItem(DataControler * controler) {
	return nullptr;
}

template<typename InputType>
struct MoveTrackingReference_GetRGTVal_Kernel {
	static void run(std::vector<float>& tmpTab, long iy, long iz, long dimJ, long dimx, long dimy, float tdeb, float pasech,
			SliceRep* rgtData, int channelRgt, std::vector<float>& isochrone, SliceDirection dir) {
		const std::vector<char>& cache = rgtData->lockCache();
		const InputType* tab = static_cast<const InputType*>(static_cast<const void*>(cache.data()));
		for (std::size_t _j=0; _j<dimJ; _j++) {
			if (dir==SliceDirection::Inline) {
				int index_trace = (isochrone[iz*dimy+_j] - tdeb) / pasech;
				//referenceVec[index_ref].rgt[iz*dimy+_j] = static_cast<short*>(m_rgt->image()->backingPointer())[_j + index_trace*dimJ];
				tmpTab[_j] = tab[_j + index_trace*dimJ + channelRgt *dimJ*dimx];
			} else {
				int index_trace = (isochrone[_j*dimy+iy] - tdeb) / pasech;
				//referenceVec[index_ref].rgt[_j*dimy+iy] = static_cast<short*>(m_rgt->image()->backingPointer())[_j + index_trace*dimJ];
				tmpTab[_j] = tab[_j + index_trace*dimJ + channelRgt *dimJ*dimx];
			}
		}
		rgtData->unlockCache();
	}
};

template<typename InputType>
struct MoveTrackingReference_GetFinalValues {
	static double run(const void* rgtData, int channel, long iy, long iz, long dimJ, long dimy, long dimx, float tdeb, float pasech,
			const std::vector<int>& traceReferenceLimits, const std::vector<ReferenceDuo>& referenceVec,
			const std::vector<int>& referenceValue, std::size_t index, long index_ref, QPoint pt, long x) {
		const InputType* tabRGT = static_cast<const InputType*>(rgtData);
		double rgtTopValue = getNewRgtValueFromReference(iy, iz, traceReferenceLimits[index_ref],
						tabRGT[traceReferenceLimits[index_ref]*dimJ + index + channel*dimJ*dimx], tdeb, pasech, dimy,
						referenceVec, referenceValue);
		double rgtBottomValue = getNewRgtValueFromReference(iy, iz, traceReferenceLimits[index_ref+1],
						tabRGT[traceReferenceLimits[index_ref+1]*dimJ + index + channel*dimJ*dimx], tdeb, pasech, dimy,
						referenceVec, referenceValue);
		double rgtInitValue = getNewRgtValueFromReference(iy, iz, x,
						tabRGT[x*dimJ + index + channel*dimJ*dimx], tdeb, pasech, dimy, referenceVec, referenceValue);

		double rgtPointValue;
		if (pt.y()>=traceReferenceLimits[index_ref] && pt.y()<=traceReferenceLimits[index_ref+1]) {
			rgtPointValue = getNewRgtValueFromReference(iy, iz, pt.y(),
						tabRGT[pt.y()*dimJ + pt.x() + channel*dimJ*dimx], tdeb, pasech, dimy, referenceVec, referenceValue);
		} else if(pt.y()<traceReferenceLimits[index_ref]) {
			// top saturation
			rgtPointValue = rgtTopValue;
		} else if(pt.y()>traceReferenceLimits[index_ref+1]) {
			// bottom saturation
			rgtPointValue = rgtBottomValue;
		}

		double dtauRelative = (rgtPointValue - rgtInitValue) / (rgtBottomValue - rgtTopValue);
		return dtauRelative;
	}
};

// return without modification if imagePoint not int polygon or dTau==0
void MultiSeedRgtSliceRep::moveTrackingReference(QPoint pt) {
	std::vector<std::shared_ptr<FixedLayerFromDataset>> referenceLayers = m_data->getReferences();
	if (true referenceLayers.size()==0 ) {
		long dimJ;
		if (m_dir==SliceDirection::Inline) { // Z
			dimJ = m_data->seismic()->width();
		} else if(m_dir==SliceDirection::XLine) { // Y
			dimJ = m_data->seismic()->depth();
		} else {
			return;
		}
		// find delta tau
		if (m_polygonMainCache.size()==0) {
			m_polygonMainCache = m_polygonMain;
		}
		long x;
		std::size_t index = 0;
		while (index<m_polygonMain.size() && m_polygonMain[index].x()!=pt.x()) {
			index++;
		}
		if (index<m_polygonMain.size()) {
			x = m_polygonMainCache[index].y();
		} else {
			return;
		}

		//clearCurveSpeedupCache();

		QPoint pointRef(pt.x(), x);
		//m_rgt->image()->lockPointer();
		long imageVal = getValueFromSliceRepForSeed(m_rgt, pt.x(), pt.y(), m_data->channelRgt());
		long refVal = getValueFromSliceRepForSeed(m_rgt, pointRef.x(), pointRef.y(), m_data->channelRgt());
		//long imageVal = static_cast<short*>(m_rgt->image()->backingPointer())[pt.x() + pt.y() * dimJ];
		//long refVal = static_cast<short*>(m_rgt->image()->backingPointer())[pointRef.x() + pointRef.y() * dimJ];
		//m_rgt->image()->unlockPointer();
		long dTau = imageVal - refVal;

		applyDeltaTau(dTau);
	} else {
		long x;
		std::size_t index = pt.x();
		//index = (index<0) ? 0 : index;
		index = (index>=m_polygonMain.size()) ? m_polygonMain.size()-1 : index;

		if (m_staticPointCurve[index]==0) {
			double _x = m_meansCurve[index] / m_weightsCurve[index];
			long x_floor = std::floor(_x);

			x = x_floor;
		} else {
			x = m_selectedIndexCurve[index];
			QPoint pointRef(pt.x(), x);
			//process::getNewRgtValueFromReference()
		}

		// get data characteristics
		Seismic3DDataset* dataset = m_data->seismic();
		float tdeb = dataset->sampleTransformation()->b();
		float pasech = dataset->sampleTransformation()->a();
		long dimx = dataset->height();
		long dimy = dataset->width();
		long dimz = dataset->depth();
		long dimJ;

		// get point position in map
		long iy, iz;
		if (m_dir==SliceDirection::Inline) {
			iy = index;
			iz = m_currentSlice;
			dimJ = dimy;
		} else {
			iy = m_currentSlice;
			iz = index;
			dimJ = dimz;
		}


		long index_ref = -1;
		QString isoName("isochrone");
		std::vector<int> traceReferenceLimits; // stock indexes from referenceLayers on selected trace
		traceReferenceLimits.resize(referenceLayers.size()+2, dimx-1);
		traceReferenceLimits[0] = 0;

		std::vector<ReferenceDuo> referenceVec;
		ReferenceDuo init;
		referenceVec.resize(referenceLayers.size(), init);

		long indexBottom;
		do {
			index_ref ++;

			// build referenceVec at the same time
			std::vector<float>& isochrone = referenceVec[index_ref].iso;
			isochrone.resize(dimy*dimz);
			referenceVec[index_ref].rgt.resize(dimy*dimz);

			referenceLayers[index_ref]->readProperty(isochrone.data(), isoName);
			indexBottom = (isochrone[iz*dimy+iy] - tdeb ) / pasech;
			traceReferenceLimits[index_ref+1] = indexBottom;

			std::vector<float> tmpTab, tmpTab2;
			tmpTab.resize(dimJ, 0);
			tmpTab2.resize(dimJ, 0);

			SampleTypeBinder binder(m_rgt->image()->sampleType());
			binder.bind<MoveTrackingReference_GetRGTVal_Kernel>(tmpTab, iy, iz, dimJ, dimx, dimy, tdeb, pasech,
					m_rgt, m_data->channelRgt(), isochrone, m_dir);

			UtFiltreMeanX(tmpTab.data(), tmpTab2.data(), dimJ, 1, 21);
			for (std::size_t _j=0; _j<dimJ; _j++) {
				if (m_dir==SliceDirection::Inline) {
					int index_trace = (isochrone[iz*dimy+_j] - tdeb) / pasech;
					referenceVec[index_ref].rgt[iz*dimy+_j] = tmpTab2[_j];
				} else {
					int index_trace = (isochrone[_j*dimy+iy] - tdeb) / pasech;
					referenceVec[index_ref].rgt[_j*dimy+iy] = tmpTab2[_j];
				}
			}
		} while(index_ref+1<referenceLayers.size() && x>indexBottom);

		for (int i=index_ref+1; i<referenceVec.size(); i++) {
			std::vector<float>& isochrone = referenceVec[i].iso;
			isochrone.resize(dimy*dimz);
			referenceVec[i].rgt.resize(dimy*dimz);
			referenceLayers[i]->readProperty(isochrone.data(), isoName);
			traceReferenceLimits[i+1] = indexBottom;
			std::vector<float> tmpTab, tmpTab2;
			tmpTab.resize(dimJ, 0);
			tmpTab2.resize(dimJ, 0);

			SampleTypeBinder binder(m_rgt->image()->sampleType());
			binder.bind<MoveTrackingReference_GetRGTVal_Kernel>(tmpTab, iy, iz, dimJ, dimx, dimy, tdeb, pasech,
					m_rgt, m_data->channelRgt(), isochrone, m_dir);

			UtFiltreMeanX(tmpTab.data(), tmpTab2.data(), dimJ, 1, 21);
			for (std::size_t _j=0; _j<dimJ; _j++) {
				if (m_dir==SliceDirection::Inline) {
					int index_trace = (isochrone[iz*dimy+_j] - tdeb) / pasech;
					referenceVec[index_ref].rgt[iz*dimy+_j] = tmpTab2[_j];
				} else {
					int index_trace = (isochrone[_j*dimy+iy] - tdeb) / pasech;
					referenceVec[index_ref].rgt[_j*dimy+iy] = tmpTab2[_j];
				}
			}
		}

		if (x>indexBottom) {
			index_ref ++;
		}

		// extract layer "central" value
		std::vector<int> referenceValue;
		referenceValue.resize(referenceLayers.size());
		for (std::size_t _index = 0; _index<((index_ref+1< referenceLayers.size()) ? index_ref+1  : referenceLayers.size()); _index++) {
			if (m_dir==SliceDirection::Inline) {
				referenceValue[_index] = referenceVec[_index].rgt[iz*dimy + (dimy/2)];
			}  else {
				referenceValue[_index] = referenceVec[_index].rgt[(dimz/2)*dimy + iy];
			}
		}

		// check if new point in layer
		const std::vector<char>& cache = m_rgt->lockCache();

		SampleTypeBinder binder(m_rgt->image()->sampleType());
		double dtauRelative = binder.bind<MoveTrackingReference_GetFinalValues>(static_cast<const void*>(cache.data()), m_data->channelRgt(),
				iy, iz, dimJ, dimy, dimx, tdeb, pasech, traceReferenceLimits, referenceVec, referenceValue, index, index_ref, pt, x);

		m_rgt->unlockCache();

		applyDeltaTauRelative(dtauRelative, index_ref, referenceVec, referenceValue);//, m_rgtVisual->getDimensions());
	}

}

template<typename InputType>
struct UpdateMainHorizonWithShiftKernel {
	static void run(SliceRep* rgtData, int channelRgt, QPolygon& polygonMain, QPolygon polygonMainCache, int dTau,
					int dtauPolygonMainCache, const std::vector<double>& means, const std::vector<double>& weights,
					const std::vector<int>& selectedPoints, const std::vector<int>& staticIndice) {
		long dimx = rgtData->image()->height();

		long dimy = rgtData->image()->width();
		const std::vector<char>& cacheBuffer = rgtData->lockCache();
		const InputType* rgtBuf = static_cast<const InputType*>(static_cast<const void*>(cacheBuffer.data()));
		for (int index=0; index<polygonMain.size(); index++) {
			//QPoint pt = polygonMain[index];
			long x; //pt.x();
			long y = polygonMain[index].x();
			int tauRef;

			if (staticIndice[y]==0) {
				double _x = means[y] / weights[y];
				long x_floor = std::floor(_x);
				long x_ceil = x_floor + 1;
				int x_floor_rgt = rgtBuf[x_floor*dimy+y + channelRgt*dimx*dimy];
				int x_ceil_rgt = rgtBuf[x_ceil*dimy+y + channelRgt*dimx*dimy];

				double rgt_value = x_floor_rgt + (_x-x_floor) / (x_ceil-x_floor) * (x_ceil_rgt - x_floor_rgt);
				tauRef = rgt_value + dtauPolygonMainCache;
				x = x_floor;
			} else {
				x = selectedPoints[y];
				tauRef = rgtBuf[x*dimy+y + channelRgt*dimx*dimy] + dtauPolygonMainCache;
			}

			if (dTau>0) {
				x = std::max(x, 0l);
				while (x<dimx && rgtBuf[x*dimy+y + channelRgt*dimx*dimy]<tauRef) {
					x++;
				}
				x = std::min(x, dimx-1);
			} else {
				x = std::min(x, dimx);
				long oldX = x;
				while (x>=0 && rgtBuf[x*dimy+y + channelRgt*dimx*dimy]>tauRef) {
					x--;
				}
				if (x<dimx-1 && rgtBuf[x*dimy+y + channelRgt*dimx*dimy]<tauRef) {
					x++;
				}

			}

			QPoint newPoint(y, x);
			polygonMain[index] = newPoint;
		}
		rgtData->unlockCache();
	}
};

template<typename InputType>
struct UpdateMainHorizonWithRelativeKernel {
	static void run(SliceRep* rgtData, int channelRgt, QPolygon& polygonMain, double dTauRelative, std::size_t indexLayerBottom,
			const std::vector<double>& means, const std::vector<double>& weights,
			const std::vector<int>& selectedPoints, const std::vector<int>& staticIndice,
			const std::vector<ReferenceDuo>& referenceVec,
			const std::vector<int>& referenceValues,
			SliceDirection direction, int currentSlice) {
		Seismic3DDataset* rgtDataset = dynamic_cast<Seismic3DDataset*>(rgtData->data());

		long dimx = rgtDataset->height();
		float tdeb = rgtDataset->sampleTransformation()->b();
		float pasech = rgtDataset->sampleTransformation()->a();
		long dimy = rgtDataset->width();
		long dimz = rgtDataset->depth();
		long dimJ;

		const void* cacheTab = static_cast<const void*>(rgtData->lockCache().data());
		const InputType* rgtBuf = static_cast<const InputType*>(cacheTab);
		for (int index=0; index<polygonMain.size(); index++) {
			//QPoint pt = polygonMain[index];
			long x; //pt.x();
			long y = polygonMain[index].x();
			double tauRef;

			// get point position in map
			long iy, iz;
			if (direction==SliceDirection::Inline) {
				iy = y;
				iz = currentSlice;
				dimJ = dimy;
			} else {
				iy = currentSlice;
				iz = y;
				dimJ = dimz;
			}

			long index_ref = -1;
			std::vector<int> traceReferenceLimits; // stock indexes from referenceLayers on selected trace
			long top_index;
			double top_rgt;
			long bottom_index;
			double bottom_rgt;
			if (indexLayerBottom==0) {
				top_index = 0;
			} else {
				top_index = (referenceVec[indexLayerBottom-1].iso[iz*dimy+iy] - tdeb ) / pasech;
			}
			if (indexLayerBottom==referenceVec.size()) {
				bottom_index = dimx-1;
			} else {
				bottom_index = (referenceVec[indexLayerBottom].iso[iz*dimy+iy] - tdeb) / pasech;
			}
			top_rgt = getNewRgtValueFromReference(iy, iz, top_index, rgtBuf[top_index*dimJ + y + channelRgt*dimx*dimJ], tdeb, pasech, dimy, referenceVec, referenceValues);
			bottom_rgt = getNewRgtValueFromReference(iy, iz, bottom_index, rgtBuf[bottom_index*dimJ + y + channelRgt*dimx*dimJ], tdeb, pasech, dimy, referenceVec, referenceValues);

			double dTau = dTauRelative * (bottom_rgt - top_rgt);

			if (staticIndice[y]==0) {
				double _x = means[y] / weights[y];
				long x_floor = std::floor(_x);
				long x_ceil = x_floor + 1;
				int x_floor_rgt = getNewRgtValueFromReference(iy, iz, x_floor, rgtBuf[x_floor*dimJ+y + channelRgt*dimx*dimJ], tdeb, pasech, dimy, referenceVec, referenceValues);
				int x_ceil_rgt = getNewRgtValueFromReference(iy, iz, x_ceil, rgtBuf[x_ceil*dimJ+y + channelRgt*dimx*dimJ], tdeb, pasech, dimy, referenceVec, referenceValues);

				double rgt_value = x_floor_rgt + (_x-x_floor) / (x_ceil-x_floor) * (x_ceil_rgt - x_floor_rgt);
				tauRef = rgt_value + dTau;
				x = x_floor;
			} else {
				x = selectedPoints[y];
				tauRef = rgtBuf[x*dimJ+y + channelRgt*dimx*dimJ] + dTau;
			}
			if (tauRef>bottom_rgt) {
				tauRef = bottom_rgt;
			}
			if (tauRef<top_rgt) {
				tauRef = top_rgt;
			}

			if (dTau>0) {
				x = std::max(x, 0l);
				while (x<dimx && getNewRgtValueFromReference(iy, iz, x, rgtBuf[x*dimJ+y + channelRgt*dimx*dimJ], tdeb, pasech, dimy, referenceVec, referenceValues)<tauRef) {
					x++;
				}
				x = std::min(x, dimx-1);
			} else {
				x = std::min(x, dimx);
				long oldX = x;
				while (x>=0 && getNewRgtValueFromReference(iy, iz, x, rgtBuf[x*dimJ+y + channelRgt*dimx*dimJ], tdeb, pasech, dimy, referenceVec, referenceValues)>tauRef) {
					x--;
				}
				if (x<dimx-1 && getNewRgtValueFromReference(iy, iz, x, rgtBuf[x*dimJ+y + channelRgt*dimx*dimJ], tdeb, pasech, dimy, referenceVec, referenceValues)<tauRef) {
					x++;
				}

			}

			QPoint newPoint(y, x);
			polygonMain[index] = newPoint;
		}
		rgtData->unlockCache();
	}
};

void MultiSeedRgtSliceRep::applyDeltaTauRelative(double dTau, std::size_t indexLayerBottom, const std::vector<ReferenceDuo>& referenceVec,
		const std::vector<int>& referenceValues) {
	if (dTau==0) {
		return;
	}

	long dimx = m_data->seismic()->height();

	//m_dtauReference += dTau;
	m_data->setDTauRelativePolygonMainCache(m_data->getDTauRelativePolygonMainCache() + dTau);

	// refresh gui
	if (m_data->useSnap()) {
		updateMainHorizon();
	} else {

//		AccesserCachedImage& rgtData = m_rgtVisual->getDisplayedImageAccesser();
		SampleTypeBinder binder(m_rgt->image()->sampleType());
		binder.bind<UpdateMainHorizonWithRelativeKernel> (m_rgt, m_data->channelRgt(), m_polygonMain, dTau, indexLayerBottom,
				m_meansCurve, m_weightsCurve, m_selectedIndexCurve, m_staticPointCurve, referenceVec, referenceValues,
				m_dir, m_currentSlice);



		if (m_data->getHorizonMode()==DELTA_T) {
			updateDeltaHorizon();
		} else {
			updateGraphicRepresentation();
		}
	}
}


void MultiSeedRgtSliceRep::applyDeltaTau(int dTau) {
	initSliceReps();

	if (dTau==0) {
		return;
	}

	long dimx = m_data->seismic()->width();

//	m_data->setDTauReference(dTau);
	m_data->setDTauPolygonMainCache(dTau);

	// refresh gui
	if (m_data->useSnap()) {
		updateMainHorizon();
	} else {


		//AccesserCachedImage& rgtData = m_rgtVisual->getDisplayedImageAccesser();
		SampleTypeBinder binder(m_rgt->image()->sampleType());
		binder.bind<UpdateMainHorizonWithShiftKernel> (m_rgt, m_data->channelRgt(), m_polygonMain,
						m_polygonMainCache, dTau, m_data->getDTauPolygonMainCache(),
						m_meansCurve, m_weightsCurve, m_selectedIndexCurve, m_staticPointCurve);



		if (m_data->getHorizonMode()==DELTA_T) {
			updateDeltaHorizon();
		} else {
			updateGraphicRepresentation();
		}
	}
}

void MultiSeedRgtSliceRep::initSliceReps() {
	AbstractSectionView* view = dynamic_cast<AbstractSectionView*>(m_parent);
	if ((m_rgt==nullptr || m_seismic==nullptr) && view!=nullptr) {
		std::pair<SliceRep*, SliceRep*> out = findSliceRepsFromSectionInnerViewAndData(m_data, view);
		m_seismic = out.first;
		m_rgt = out.second;
	}
}

template<typename InputType>
void MultiSeedRgtSliceRep::CorrectSeedsFromImageKernel<InputType>::run(const void* rgtData,
		int channelRgt, MultiSeedRgtSliceRep* obj, std::size_t dimI, std::size_t dimJ,
		QList<std::tuple<RgtSeed, RgtSeed, std::size_t>>& seedsChangeList) {
	const InputType* rgtBuf = static_cast<const InputType*>(rgtData);

	const std::map<std::size_t, RgtSeed>& seeds = obj->m_data->getMap();

	// Detect seeds in image and change their x if needed
	#pragma omp parallel for
	for(long i=0; i<seeds.size(); i++) {
		std::map<std::size_t, RgtSeed>::const_iterator seedPairIt = std::begin(seeds);
		std::advance(seedPairIt, i);
		const RgtSeed& seed = seedPairIt->second;
		if ((seed.y==obj->m_currentSlice && obj->m_dir==SliceDirection::XLine && seed.z>=0 && seed.z<dimJ) ||
				(seed.z==obj->m_currentSlice && obj->m_dir==SliceDirection::Inline && seed.y>=0 && seed.y<dimJ)) {
			// seed in image
			long ix = seed.x;
			long iy = (obj->m_dir==SliceDirection::Inline) ? seed.y : seed.z;
			double dtauReference = seed.rgtValue - rgtBuf[ix*dimJ + iy + channelRgt*dimI*dimJ];
			if (dtauReference>0) {
				ix = std::max(ix, 0l);
				while (ix<dimI && rgtBuf[ix*dimJ+iy + channelRgt*dimI*dimJ]<rgtBuf[seed.x*dimJ+iy + channelRgt*dimI*dimJ]+dtauReference) {
					ix++;
				}
				ix = std::min(ix, static_cast<long>(dimI-1));
			} else if(dtauReference<0) {
				ix = std::min(ix, static_cast<long>(dimI));
				long oldX = ix;
				while (ix>=0 && rgtBuf[ix*dimJ+iy + channelRgt*dimI*dimJ]>rgtBuf[seed.x*dimJ+iy + channelRgt*dimI*dimJ]+dtauReference) {
					ix--;
				}
				if (ix<dimI-1 && rgtBuf[ix*dimJ+iy + channelRgt*dimI*dimJ]<rgtBuf[seed.x*dimJ+iy + channelRgt*dimI*dimJ]+dtauReference) {
					ix++;
				}
			}
			if (ix<0) {
				ix = 0;
			} else if (ix>=dimI) {
				ix = dimI - 1;
			}
			if (ix!=seed.x) {
				RgtSeed newSeed = seed;
				newSeed.x = ix;
				#pragma omp critical
				{
					seedsChangeList.append(std::tuple<RgtSeed, RgtSeed, std::size_t>(seed, newSeed, seedPairIt->first));
				}
			}
		}
	}
}

void MultiSeedRgtSliceRep::correctSeedsFromImage() {
	initSliceReps(); // needed before accessing buffers

	CUDAImagePaletteHolder* rgtData = m_rgt->image();

	std::size_t dimJ = rgtData->width();
	std::size_t dimI = rgtData->height();

	// lock buffers
	const std::vector<char>& rgtVect = m_rgt->lockCache();

	// pasech and tdeb are not needed as references and constrains do not matter here
	QList<std::tuple<RgtSeed, RgtSeed, std::size_t>> seedsChangeList;

	SampleTypeBinder binder(m_rgt->image()->sampleType());
	binder.bind<CorrectSeedsFromImageKernel>(static_cast<const void*>(rgtVect.data()),
			m_data->channelRgt(), this, dimI, dimJ, seedsChangeList);

	//unlock buffers
	m_rgt->unlockCache();

	// apply seeds changes
	if (seedsChangeList.size()>0) {
		emit m_data->startApplyDTauTransaction(this);
		for (std::tuple<RgtSeed, RgtSeed, std::size_t>& myTuple : seedsChangeList) {
			//emit pointMoved(std::get<0>(myTuple), std::get<1>(myTuple), std::get<2>(myTuple));
			m_data->moveSeed(std::get<2>(myTuple), std::get<1>(myTuple));
		}
		emit m_data->endApplyDTauTransaction(this);
	}
}

bool MultiSeedRgtSliceRep::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> MultiSeedRgtSliceRep::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_data->seismic()->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString MultiSeedRgtSliceRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

template<typename InputType>
struct GetMeanTauOnCurveKernel {
	static long run(const void* buffer, int channel, long width, long height, const QPolygon& poly) {
		long N = 0;
		long tau;

		const InputType* tab = static_cast<const InputType*>(buffer);
		for (long iy=0; iy<width; iy++) {
			long ix = poly.at(iy).y();
			if (ix>=0 && ix<height) {
				InputType val = tab[iy + ix * width + channel*width*height];
				tau += val;
				N++;
			}
		}
		tau /= N;
		return tau;
	}
};

long MultiSeedRgtSliceRep::getMeanTauOnCurve(bool* ok) {
	long tau;

	initSliceReps();
	*ok = m_rgt!=nullptr && m_polygonMain.size()==m_rgt->image()->width();
	if (*ok) {
		CUDAImagePaletteHolder* rgtData = m_rgt->image();
		long width = rgtData->width();
		long height = rgtData->height();

		SampleTypeBinder binder(m_rgt->image()->sampleType());
		tau = binder.bind<GetMeanTauOnCurveKernel>(static_cast<const void*>(m_rgt->lockCache().data()), m_data->channelRgt(), width, height, m_polygonMain);
		m_rgt->unlockCache();
	}
	return tau;
}

template<typename InputType>
struct GetValueFromSliceRepForSeedKernel {
	static int run(const void* _tab, std::size_t idx) {
		InputType* tab = (InputType*) _tab;
		return tab[idx];
	}
};

int MultiSeedRgtSliceRep::getValueFromSliceRepForSeed(SliceRep* rep, std::size_t x, std::size_t y, std::size_t channel) {
	std::size_t dimJ = rep->image()->height();
	std::size_t dimI = rep->image()->width();
	std::size_t idx = x + y * dimI + channel * dimI * dimJ;
	const std::vector<char>& sliceRepCache = rep->lockCache();
	SampleTypeBinder binder(rep->image()->sampleType());
	int outVal = binder.bind<GetValueFromSliceRepForSeedKernel>((const void*) sliceRepCache.data(), idx);
	rep->unlockCache();
	return outVal;
}

*/

AbstractGraphicRep::TypeRep MultiSeedRgtSliceRep::getTypeGraphicRep() {
    return AbstractGraphicRep::NotDefined;
}
