#include "multiseedrandomrep.h"
#include "multiseedrandomlayer.h"
#include "seismic3ddataset.h"
#include "multiseedhorizon.h"
#include "randomrep.h"
#include "imouseimagedataprovider.h"
#include "affine2dtransformation.h"
#include "cudaimagepaletteholder.h"
#include "fixedlayerfromdataset.h"
#include "randomlineview.h"
#include "sampletypebinder.h"

#include <cmath>
#include <tuple>
#include <QMutexLocker>


std::pair<RandomRep*, RandomRep*> MultiSeedRandomRep::findRandomRepsFromRandomInnerViewAndData(
		MultiSeedHorizon *data, RandomLineView *parent) {
	RandomRep* seismic = nullptr;
	RandomRep* rgt = nullptr;

	const QList<AbstractGraphicRep*>& reps = parent->getVisibleReps();
	std::size_t index = 0;
	while (index<reps.size() && (seismic==nullptr || rgt==nullptr)) {
		RandomRep* slice = dynamic_cast<RandomRep*>(reps[index]);
		if (slice!=nullptr && slice->data()==data->seismic()) {
			seismic = slice;
		} else if (slice!=nullptr && slice->data()==data->rgt()) {
			rgt = slice;
		}
		index++;
	}

	std::pair<RandomRep*, RandomRep*> out;
	out.first = seismic;
	out.second = rgt;


	return out;
}

MultiSeedRandomRep::MultiSeedRandomRep(MultiSeedHorizon *data, AbstractInnerView *parent) :
		AbstractGraphicRep(parent), m_eventFilterClass({Qt::Key_S}, this)  {
	m_data = data;
	m_layer = nullptr;

	m_penMain = QPen(QColor(255, 0, 0));
	m_penDelta = QPen(QColor(0, 255, 0));
	m_pointsPen = QPen(QColor(0, 0, 255));

	m_currentRgtImage = nullptr;
	m_currentSeismicImage = nullptr;

//	connect(m_data->seismic()->image(), &CUDAImagePaletteHolder::dataChanged, this, [this]() {
//		setSliceIJPosition(m_currentSlice);
//	}, Qt::AutoConnection);
//	connect(m_data->rgt()->image(), &CUDAImagePaletteHolder::dataChanged, this, [this]() {
//		setSliceIJPosition(m_currentSlice);
//	}, Qt::AutoConnection);

	// connect for data
	connect(m_data, &MultiSeedHorizon::deltaChanged, this, &MultiSeedRandomRep::updateDeltaHorizon);
	connect(m_data, &MultiSeedHorizon::polarityChanged, this, &MultiSeedRandomRep::updateMainHorizon);
	connect(m_data, &MultiSeedHorizon::lwxMedianFilterChanged, this, &MultiSeedRandomRep::updateMainHorizon);
	connect(m_data, &MultiSeedHorizon::distancePowerChanged, this, &MultiSeedRandomRep::updateMainHorizon);
	connect(m_data, &MultiSeedHorizon::useMedianChanged, this, &MultiSeedRandomRep::updateMainHorizon);
	connect(m_data, &MultiSeedHorizon::snapWindowChanged, this, &MultiSeedRandomRep::updateMainHorizon);
	connect(m_data, &MultiSeedHorizon::useSnapChanged, this, &MultiSeedRandomRep::updateMainHorizon);
	connect(m_data, &MultiSeedHorizon::horizonModeChanged, this, &MultiSeedRandomRep::updateDeltaHorizon);
	connect(m_data, &MultiSeedHorizon::parametersUpdated, this, &MultiSeedRandomRep::updateMainHorizon);
	connect(m_data, &MultiSeedHorizon::newPointCreated, this, &MultiSeedRandomRep::newPointCreatedSynchro);
	connect(m_data, &MultiSeedHorizon::pointRemoved, this, &MultiSeedRandomRep::pointRemovedSynchro);
	connect(m_data, &MultiSeedHorizon::pointMoved, this, &MultiSeedRandomRep::pointMovedSynchro);
	connect(m_data, &MultiSeedHorizon::seedsReset, this, &MultiSeedRandomRep::seedsResetSynchro);
	connect(m_data, &MultiSeedHorizon::constrainChanged, this, &MultiSeedRandomRep::updateMainHorizonNewConstrain);
	connect(m_data, &MultiSeedHorizon::constrainIsoChanged, this, &MultiSeedRandomRep::updateMainHorizonNewConstrain);
	connect(m_data, &MultiSeedHorizon::referencesChanged, this, &MultiSeedRandomRep::updateMainHorizonNoCache);
	connect(m_data, &MultiSeedHorizon::dtauPolygonMainCacheChanged, this, &MultiSeedRandomRep::applyDeltaTau);

	connect(m_data, &MultiSeedHorizon::startApplyDTauTransaction, this, &MultiSeedRandomRep::disconnectMoveSlot);
	connect(m_data, &MultiSeedHorizon::endApplyDTauTransaction, this, &MultiSeedRandomRep::connectMoveSlotAndUpdate);

	connect(m_data, &MultiSeedHorizon::seismicChanged, this, &MultiSeedRandomRep::updateSeismicAfterSwap);
	connect(m_data, &MultiSeedHorizon::rgtChanged, this, &MultiSeedRandomRep::updateRgtAfterSwap);

	m_parent->installEventFilter(&m_eventFilterClass);
	connect(&m_eventFilterClass, &ContinuousPressEventFilter::keyPressSignal, [this](int key) {
		if (key==Qt::Key_S) {
//			m_seedEditButton->setChecked(false);
			m_oldBehaviorMode = m_data->getBehaviorMode();
			m_data->setBehaviorMode(MOUSETRACKING);
		}
	});
    connect(&m_eventFilterClass, &ContinuousPressEventFilter::keyReleaseSignal, [this](int key) {
		if (key==Qt::Key_S) {
			m_data->setBehaviorMode(m_oldBehaviorMode);
			//disconnect(m_horizonExtenstion, &view2d::ViewHorizonExtension::currentTauChanged, this, QOverload<int,bool>::of(&LayerSpectrumDialog::setGeologicalTime));
		}
	});
}

MultiSeedRandomRep::~MultiSeedRandomRep() {
	m_parent->removeEventFilter(&m_eventFilterClass);
	if (m_layer!=nullptr) {
		delete m_layer;
	}
}

void MultiSeedRandomRep::disconnectMoveSlot(QObject* requestingObj) {
	disconnect(m_data, &MultiSeedHorizon::dtauPolygonMainCacheChanged, this, &MultiSeedRandomRep::applyDeltaTau);
	disconnect(m_data, &MultiSeedHorizon::pointMoved, this, &MultiSeedRandomRep::pointMovedSynchro);
}

void MultiSeedRandomRep::connectMoveSlotAndUpdate(QObject* requestingObj) {
	connect(m_data, &MultiSeedHorizon::dtauPolygonMainCacheChanged, this, &MultiSeedRandomRep::applyDeltaTau);
	connect(m_data, &MultiSeedHorizon::pointMoved, this, &MultiSeedRandomRep::pointMovedSynchro);
	if (requestingObj!=this) { // emiting rep is supposed to take care of it itself
		m_polygonMainCache.clear();
		clearCurveSpeedupCache();
		correctSeedsFromImage();
		updateMainHorizon();
	}
}

void MultiSeedRandomRep::updateMainHorizonNewConstrain() {
	clearCurveSpeedupCache();
	m_data->setDTauReference(0);
	updateMainHorizon();
}

void MultiSeedRandomRep::updateMainHorizonNoCache() {
	clearCurveSpeedupCache();
	updateMainHorizon();
}

const QPolygon& MultiSeedRandomRep::getMainPolygon() const {
	//QMutexLocker lock(&m_polygonMutex);
	return m_polygonMain;
}

const QPolygon& MultiSeedRandomRep::getTopPolygon() const {
	//QMutexLocker lock(&m_polygonMutex);
	return m_polygonTop;
}

const QPolygon& MultiSeedRandomRep::getBottomPolygon() const {
	//QMutexLocker lock(&m_polygonMutex);
	return m_polygonBottom;
}

// jd
const std::vector<std::vector<QPolygon>>& MultiSeedRandomRep::getReferencePolygon() const {
	//QMutexLocker lock(&m_polygonMutex);
	return m_polygonReference;
}


//AbstractGraphicRep
QWidget* MultiSeedRandomRep::propertyPanel() {
	return nullptr;
}

GraphicLayer* MultiSeedRandomRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {

	if (m_layer==nullptr) {
	    updateMainHorizonNoCache();
		m_layer = new MultiSeedRandomLayer(this, scene, defaultZDepth, parent);
	}
	return m_layer;
}

std::size_t MultiSeedRandomRep::addPoint(QPoint point) {
	if (m_data->getBehaviorMode()==MOUSETRACKING) {
		return MultiSeedHorizon::INVALID_ID;
	}

	initRandomReps();

	if (m_data->getDTauPolygonMainCache()!=0) {
		m_data->applyDtauToSeeds(); //applyDTauToData();
		m_polygonMainCache.clear();
	}

	QPolygon polygon = dynamic_cast<RandomLineView*>(view())->discreatePolyLine();

	// care point is transposed
	long dimI = m_data->seismic()->height();
	long dimJ = polygon.size();
//	if (m_dir==SliceDirection::Inline) { // Z
//		dimJ = m_data->seismic()->width();
//	} else if(m_dir==SliceDirection::XLine) { // Y
//		dimJ = m_data->seismic()->depth();
//	} else {
//		return MultiSeedHorizon::INVALID_ID;
//	}
	if ((point.x()<0 || point.x()>=dimJ) && (point.y()<0 && point.y()>=dimI)) {
		return MultiSeedHorizon::INVALID_ID;
	}
	RgtSeed seed;
//	double worldX, worldY;
//	switch (m_dir) {
//	case SliceDirection::Inline:
//		seed.x = point.y();
//		seed.y = point.x();
//		seed.z = m_currentSlice;
////		m_data->seismic()->ijToInlineXlineTransfoForInline()->imageToWorld(point.x(), point.y(), worldX, worldY);
//		break;
//	case SliceDirection::XLine:
//		seed.x = point.y();
//		seed.y = m_currentSlice;
//		seed.z = point.x();
////		m_data->seismic()->ijToInlineXlineTransfoForXline()->imageToWorld(point.x(), point.y(), worldX, worldY);
//		break;
//	}
	seed.x = point.y();
	seed.y = polygon[point.x()].x();
	seed.z = polygon[point.x()].y();

	// get rgt
	// buffer is transposed and point too
	seed.rgtValue = getValueForSeed(m_rgt, point.x(), point.y(), m_data->channelRgt(), m_currentRgtImage);
	seed.seismicValue = getValueForSeed(m_seismic, point.x(), point.y(), m_data->channelSeismic(), m_currentSeismicImage);

	return m_data->addPoint(seed);
}

std::size_t MultiSeedRandomRep::addPoint(int x, int y) { // add seed on current section
	return addPoint(QPoint(x, y));
}

std::size_t MultiSeedRandomRep::addPointAndSelect(QPoint point) {
	std::size_t id = addPoint(point);
	if (id!=MultiSeedHorizon::INVALID_ID) {
		m_data->selectPoint(id);
	}
	return id;
}

std::size_t MultiSeedRandomRep::addPointAndSelect(int x, int y) {
	return addPointAndSelect(QPoint(x, y));
}

void MultiSeedRandomRep::moveSelectedPoint(QPointF point) {
	initRandomReps();

	IMouseImageDataProvider::MouseInfo info;
	if (m_rgt && m_rgt->mouseData(point.x(), point.y(), info) && info.values.size() < 1) {
		return;
	}

	double imageX, imageY;
//	switch (m_dir) {
//	case SliceDirection::Inline:
//		m_data->seismic()->ijToInlineXlineTransfoForInline()->worldToImage(point.x(), point.y(), imageX, imageY);
//		break;
//	case SliceDirection::XLine:
//		m_data->seismic()->ijToInlineXlineTransfoForXline()->worldToImage(point.x(), point.y(), imageX, imageY);
//		break;
//	}
	m_data->seismic()->sampleTransformation()->indirect(point.y(), imageY);
	imageX = dynamic_cast<RandomLineView*>(view())->getDiscreatePolyLineIndexFromScenePos(point);

	QPoint pt(imageX, imageY);// = m_rgtVisual->sceneToVisual(point);
	moveSelectedPoint(pt);
}

void MultiSeedRandomRep::moveSelectedPoint(QPoint point) { // move point and update
	// care point is transposed
	if (m_data->getSelectedId()==MultiSeedHorizon::INVALID_ID) {
		return;
	}
	initRandomReps();

	QPolygon polygon = dynamic_cast<RandomLineView*>(view())->discreatePolyLine();
	long dimI = m_data->seismic()->height();
	long dimJ = polygon.size();
//	if (m_dir==SliceDirection::Inline) { // Z
//		dimJ = m_data->seismic()->width();
//	} else if(m_dir==SliceDirection::XLine) { // Y
//		dimJ = m_data->seismic()->depth();
//	} else {
//		return;
//	}
	if ((point.x()<0 || point.x()>=dimJ) && (point.y()<0 && point.y()>=dimI)) {
		return;
	}
	RgtSeed seed;
//	double worldX, worldY;
//	switch (m_dir) {
//	case SliceDirection::Inline:
//		seed.x = point.y();
//		seed.y = point.x();
//		seed.z = m_currentSlice;
//		m_data->seismic()->ijToInlineXlineTransfoForInline()->imageToWorld(point.x(), point.y(), worldX, worldY);
//		break;
//	case SliceDirection::XLine:
//		seed.x = point.y();
//		seed.y = m_currentSlice;
//		seed.z = point.x();
//		m_data->seismic()->ijToInlineXlineTransfoForXline()->imageToWorld(point.x(), point.y(), worldX, worldY);
//		break;
//	}
	seed.x = point.y();
	seed.y = polygon[point.x()].x();
	seed.z = polygon[point.x()].y();

	// get rgt
	// buffer is transposed and point too
	seed.rgtValue = getValueForSeed(m_rgt, point.x(), point.y(), m_data->channelRgt(), m_currentRgtImage);
	seed.seismicValue = getValueForSeed(m_seismic, point.x(), point.y(), m_data->channelSeismic(), m_currentSeismicImage);

	if (m_data->getDTauPolygonMainCache()!=0) {
		m_data->applyDtauToSeeds(); //applyDTauToData();
		m_polygonMainCache.clear();
	}

	m_data->moveSelectedSeed(seed);
}

void MultiSeedRandomRep::moveSelectedPoint(int x, int y) {
	moveSelectedPoint(QPoint(x, y));
}

void MultiSeedRandomRep::updateSeedsRepresentation() {
	if (m_layer) {
		m_layer->refresh();
	}
}

// graphic
void MultiSeedRandomRep::updateGraphicRepresentation() { // ui update
	if (m_layer) {
		m_layer->refresh();
	}
}

void MultiSeedRandomRep::setReferences(std::vector<float*> ref)
{
	m_reference = ref;
}



template<typename RgtType>
struct UpdateMainHorizonKernel {
	template<typename SeismicType>
	struct UpdateMainHorizonKernelLevel2 {
		static void run(MultiSeedRandomRep* ext, const void* rgtData, const void* seismicData,
						long dimx, long dimy, std::map<std::size_t, RgtSeed> seeds, FixedLayerFromDataset* constrainLayer,
						std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers, QPolygon& poly, bool useSnap,
						bool useMedian, int lwx, const QPolygon& polygon, int distancePower,
						int snapWindow, int polarity, float tdeb, float pasech, long dtauReference,
						std::vector<double>& weights, std::vector<double>& means, std::vector<int>& staticPointCurve,
						std::vector<int>& selectedIndexCurve, std::vector<bool>& constrainCurve) {

			const RgtType* rgtBuf = static_cast<const RgtType*>(rgtData);
			const SeismicType* seismicBuf = static_cast<const SeismicType*>(seismicData);

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
			constrainCurve.resize(dimy, false);

			int type = polarity;

			bool isReferenceLayerSet = false;
			std::vector<ReferenceDuo> referenceLayersVec;
			std::vector<int> referenceValues;
			//std::vector<float> referenceLayerVector;
			//std::vector<float> referenceLayerRgtPropVector;

			QString rgtName("rgt");
			if (false/*referenceLayers.size()!=0*/) {
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
	//				if (dir==SliceDirection::Inline) {
	//					for (long iy=0; iy<referenceLayers[0]->getNbTraces(); iy++) {
	//						long ix = (pair.iso[slice*referenceLayers[0]->getNbTraces()+iy] - tdeb) / pasech;
	//						pair.rgt[slice*referenceLayers[0]->getNbTraces()+iy] = rgtBuf[ix*dimy+iy];
	//					}
	//					referenceValues[index] = pair.rgt[slice*referenceLayers[0]->getNbTraces()+referenceLayers[0]->getNbTraces()/2];
	//				} else if (dir==SliceDirection::XLine) {
	//					for (long iz=0; iz<referenceLayers[0]->getNbProfiles(); iz++) {
	//						long ix = (pair.iso[iz*referenceLayers[0]->getNbTraces()+slice] - tdeb) / pasech;
	//						pair.rgt[iz*referenceLayers[0]->getNbTraces()+slice] = rgtBuf[ix*dimy+iz];
	//					}
	//					referenceValues[index] = pair.rgt[(referenceLayers[0]->getNbProfiles()/2)*referenceLayers[0]->getNbTraces()+slice];
	//				} else {
	//					qDebug() << "MultiSeedHorizonExtension : invalid orientation";
	//				}
					for (long idx=0; idx<polygon.size(); idx++) {
						long iz = polygon[idx].y();
						long iy = polygon[idx].x();
						long ix = (pair.iso[iz*referenceLayers[0]->getNbTraces()+iy] - tdeb) / pasech;
						pair.rgt[iz*referenceLayers[0]->getNbTraces()+iy] = rgtBuf[ix*dimy+idx];
					}
					long izMid = polygon[polygon.size()/2].y();
					long iyMid = polygon[polygon.size()/2].x();
					referenceValues[index] = pair.rgt[izMid*referenceLayers[0]->getNbTraces()+iyMid];
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
	//			if (dir==SliceDirection::XLine) {
	//				iz = y;
	//				iy = slice;
	//			} else {
	//				iz = slice;
	//				iy = y;
	//			}
				iy = polygon[y].x();
				iz = polygon[y].y();

				long ix, foundIx;
				bool seedFound = false;
				if (isConstrainLayerSet && constrainLayerVector[iy+iz*constrainLayer->getNbTraces()]!=-9999) {
					long ixOri = (constrainLayerVector[iy+iz*constrainLayer->getNbTraces()]-tdeb)/pasech;
					ix = ixOri;
					if (dtauReference>0) {
						ix = std::max(ix, 0l);
						while (ix<dimx && rgtBuf[ix*dimy+y]<rgtBuf[ixOri*dimy+y]+dtauReference) {
							ix++;
						}
						ix = std::min(ix, dimx-1);
					} else if(dtauReference<0) {
						ix = std::min(ix, dimx);
						long oldX = ix;
						while (ix>=0 && rgtBuf[ix*dimy+y]>rgtBuf[ixOri*dimy+y]+dtauReference) {
							ix--;
						}
						if (ix<dimx-1 && rgtBuf[ix*dimy+y]<rgtBuf[ixOri*dimy+y]+dtauReference) {
							ix++;
						}

					}
					seedFound = true;
					foundIx = ix;
					staticPointCurve[y] += 1;
					constrainCurve[y] = true;
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


				/*InputType val=0;
				if (isReferenceLayerSet) {
					int index_val = (referenceLayerVector[iz*referenceLayer->getNbTraces() + iy] - tdeb) / pasech;
					val = rgtBuf[y*dimx + index_val];
				}*/

				double ixDouble = 0 ;
				ix = 1;
				double weightedIso = 0;
				for(int i=0; i < seedsVec.size() ; i++) {
					if (isReferenceLayerSet) {
						while ( getNewRgtValueFromReference(iy, iz, ix, rgtBuf[y + ix*dimy], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues)  < seedsVec[i].rgtValue  && ix < dimx ) {
							ix ++ ;
						}
						if (ix==dimx) {
							ix = dimx-1;
						}
						double ix_rgt = getNewRgtValueFromReference(iy, iz, ix, rgtBuf[y + ix*dimy], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues);
						if (ix_rgt==seedsVec[i].rgtValue || ix==0) {
							ixDouble = ix;
						} else {
							double ix_floor_rgt = getNewRgtValueFromReference(iy, iz, ix-1, rgtBuf[y + (ix-1)*dimy], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues);
							ixDouble = ix-1 + (seedsVec[i].rgtValue - ix_floor_rgt) / (ix_rgt - ix_floor_rgt);
							if (ixDouble>ix) {
								ixDouble = ix;
							} else if (ixDouble<ix-1) {
								ixDouble = ix-1;
							}
						}
					} else {
						while ( ix < dimx && rgtBuf[ix*dimy + y]  < seedsVec[i].rgtValue ) {
							ix ++ ;
						}
						if (ix==dimx) {
							ix = dimx-1;
						}
						if (rgtBuf[ix*dimy + y]==seedsVec[i].rgtValue || ix==0) {
							ixDouble = ix;
						} else {
							double ix_rgt = rgtBuf[ix*dimy + y];
							int previousIx = ix-1;
							if (ix_rgt>0) {
								// rewind zeros, for patch debug purposes
								bool continueSearch = rgtBuf[previousIx*dimy + y]==0;
								while (previousIx>=0 && continueSearch) {
									continueSearch = rgtBuf[previousIx*dimy + y]==0;
									if (continueSearch) {
										previousIx--;
									}
								}
								if (continueSearch || previousIx<0) {
									previousIx = ix-1;
								}
							}
							double ix_floor_rgt = rgtBuf[previousIx*dimy + y];
							ixDouble = previousIx + (seedsVec[i].rgtValue - ix_floor_rgt) / (ix_rgt - ix_floor_rgt) * (ix - previousIx);

							if (ixDouble>ix) {
									ixDouble = ix;
							} else if (ixDouble<previousIx) {
									ixDouble = previousIx;
							}

						}
					}

					weightedIso += ixDouble*dist[i] ;
				}
				if(!seedFound) {
					if (som!=0) {
						points[y] = std::round(weightedIso/som) ;
						if (points[y]>=dimx) {
							points[y] = dimx - 1;
						} else if (points[y]<0) {
							points[y] = 0;
						}
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
						traceBuf[index] = seismicBuf[y+index*dimy];
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

	static void run(ImageFormats::QSampleType seismicType, MultiSeedRandomRep* ext, const void* rgtData, const void* seismicData,
					long dimx, long dimy, const std::map<std::size_t, RgtSeed>& seeds, FixedLayerFromDataset* constrainLayer,
					std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers, QPolygon& poly, bool useSnap,
					bool useMedian, int lwx, const QPolygon& polygon, int distancePower,
					int snapWindow, int polarity, float tdeb, float pasech, long dtauReference,
					std::vector<double>& weights, std::vector<double>& means, std::vector<int>& staticPointCurve,
					std::vector<int>& selectedIndexCurve, std::vector<bool>& constrainCurve) {
		SampleTypeBinder binder(seismicType);
		binder.bind<UpdateMainHorizonKernelLevel2>(ext, rgtData, seismicData,
				dimx, dimy, seeds, constrainLayer,
				referenceLayers, poly, useSnap,
				useMedian, lwx, polygon, distancePower,
				snapWindow, polarity, tdeb, pasech, dtauReference,
				weights, means, staticPointCurve,
				selectedIndexCurve, constrainCurve);
	}
};

template<typename RgtType>
struct UpdateMainHorizonWithCacheKernel {
	template<typename SeismicType>
	struct UpdateMainHorizonWithCacheKernelLevel2 {
		static void run(MultiSeedRandomRep* ext, const void* rgtData, const void* seismicData,
						long dimx, long dimy, std::vector<RgtSeed> newSeeds, std::vector<RgtSeed> removedSeeds,
						std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers, QPolygon& poly, bool useSnap,
						bool useMedian, int lwx, const QPolygon& polygon, int distancePower,
						int snapWindow, int polarity, float tdeb, float pasech, long dtauReference,
						std::vector<double>& weights, std::vector<double>& means, std::vector<int>& staticPointCurve,
						std::vector<int>& selectedIndexCurve, const std::vector<bool>& constrainCurve) {
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
			if (false/*referenceLayers.size()!=0*/) {
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
	//				if (dir==SliceDirection::Inline) {
	//					for (long iy=0; iy<referenceLayers[0]->getNbTraces(); iy++) {
	//						long ix = (pair.iso[slice*referenceLayers[0]->getNbTraces()+iy] - tdeb) / pasech;
	//						pair.rgt[slice*referenceLayers[0]->getNbTraces()+iy] = rgtBuf[ix*dimy+iy];
	//					}
	//					referenceValues[index] = pair.rgt[slice*referenceLayers[0]->getNbTraces()+referenceLayers[0]->getNbTraces()/2];
	//				} else if (dir==SliceDirection::XLine) {
	//					for (long iz=0; iz<referenceLayers[0]->getNbProfiles(); iz++) {
	//						long ix = (pair.iso[iz*referenceLayers[0]->getNbTraces()+slice] - tdeb) / pasech;
	//						pair.rgt[iz*referenceLayers[0]->getNbTraces()+slice] = rgtBuf[ix*dimy+iz];
	//					}
	//					referenceValues[index] = pair.rgt[(referenceLayers[0]->getNbProfiles()/2)*referenceLayers[0]->getNbTraces()+slice];
	//				} else {
	//					qDebug() << "MultiSeedHorizonExtension : invalid orientation";
	//				}
					for (long idx=0; idx<polygon.size(); idx++) {
						long iy = polygon[idx].x();
						long iz = polygon[idx].y();
						long ix = (pair.iso[iz*referenceLayers[0]->getNbTraces()+iy] - tdeb) / pasech;
						pair.rgt[iz*referenceLayers[0]->getNbTraces()+iy] = rgtBuf[ix*dimy+idx];
					}

					long iyMid = polygon[polygon.size()/2].x();
					long izMid = polygon[polygon.size()/2].y();
					referenceValues[index] = pair.rgt[izMid*referenceLayers[0]->getNbTraces()+iyMid];
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
	//			if (dir==SliceDirection::XLine) {
	//				iz = y;
	//				iy = slice;
	//			} else {
	//				iz = slice;
	//				iy = y;
	//			}
				iy = polygon[y].x();
				iz = polygon[y].y();

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
				bool oriSeedFound = staticPointCurve[y]>0 || constrainCurve[y];
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


				/*InputType val=0;
				if (isReferenceLayerSet) {
					int index_val = (referenceLayerVector[iz*referenceLayers[0]->getNbTraces() + iy] - tdeb) / pasech;
					val = rgtBuf[y*dimx + index_val];
				}*/

				ix = 1 ;
				double ixDouble = 0;
				double weightedIso = 0;
				for(int i=0; i < newSeeds.size() ; i++) {
					if (isReferenceLayerSet) {
						while ( getNewRgtValueFromReference(iy, iz, ix, rgtBuf[y + ix*dimy], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues)  < newSeeds[i].rgtValue  && ix < dimx ) {
							ix ++ ;
						}
						if (ix==dimx) {
							ix = dimx-1;
						}
						double ix_rgt = getNewRgtValueFromReference(iy, iz, ix, rgtBuf[y + ix*dimy], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues);
						if (ix==0 || ix_rgt== newSeeds[i].rgtValue) {
							ixDouble = ix;
						} else {
							double ix_floor_rgt = getNewRgtValueFromReference(iy, iz, ix-1, rgtBuf[y + (ix-1)*dimy], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues);
							ixDouble = ix-1 + (newSeeds[i].rgtValue - ix_floor_rgt) / (ix_rgt - ix_floor_rgt);

							if (ixDouble>ix) {
								ixDouble = ix;
							} else if (ixDouble<ix-1) {
								ixDouble = ix-1;
							}
						}
					} else {
						while ( rgtBuf[ix*dimy + y]  < newSeeds[i].rgtValue  && ix < dimx ) {
							ix ++ ;
						}
						if (ix==dimx) {
							ix = dimx-1;
						}
						if (ix==0 || rgtBuf[ix*dimy + y]== newSeeds[i].rgtValue) {
							ixDouble = ix;
						} else {
							double ix_floor_rgt = rgtBuf[y+(ix-1)*dimy];
							double ix_rgt = rgtBuf[ix*dimy+y];
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
						while (  getNewRgtValueFromReference(iy, iz, ix, rgtBuf[y + ix*dimy], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues) < removedSeeds[i].rgtValue  && ix < dimx ) {
							ix ++ ;
						}
						if (ix==dimx) {
							ix = dimx-1;
						}
						double ix_rgt = getNewRgtValueFromReference(iy, iz, ix, rgtBuf[y + ix*dimy], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues);
						if (ix==0 || ix_rgt== removedSeeds[i].rgtValue) {
							ixDouble = ix;
						} else {
							double ix_floor_rgt = getNewRgtValueFromReference(iy, iz, ix-1, rgtBuf[y + (ix-1)*dimy], tdeb, pasech, referenceLayers[0]->getNbTraces(), referenceLayersVec, referenceValues);
							ixDouble = ix-1 + (removedSeeds[i].rgtValue - ix_floor_rgt) / (ix_rgt - ix_floor_rgt);

							if (ixDouble>ix) {
								ixDouble = ix;
							} else if (ixDouble<ix-1) {
								ixDouble = ix-1;
							}
						}
					} else {
						while ( rgtBuf[ix*dimy + y]  < removedSeeds[i].rgtValue  && ix < dimx ) {
							ix ++ ;
						}
						if (ix==dimx) {
							ix=dimx-1;
						}
						if (ix==0 || rgtBuf[ix*dimy + y]== removedSeeds[i].rgtValue) {
							ixDouble = ix;
						} else {
							double ix_rgt = rgtBuf[ix*dimy+y];
							int previousIx = ix-1;
							if (ix_rgt>0) {
								// rewind zeros, for patch debug purposes
								bool continueSearch = rgtBuf[previousIx*dimy + y]==0;
								while (previousIx>=0 && continueSearch) {
									continueSearch = rgtBuf[previousIx*dimy + y]==0;
									if (continueSearch) {
										previousIx--;
									}
								}
								if (continueSearch || previousIx<0) {
									previousIx = ix-1;
								}
							}
							double ix_floor_rgt = rgtBuf[y+dimy*previousIx];
							ixDouble = previousIx + (removedSeeds[i].rgtValue - ix_floor_rgt) / (ix_rgt - ix_floor_rgt) * (ix - previousIx);

							if (ixDouble>ix) {
								ixDouble = ix;
							} else if (ixDouble<previousIx) {
								ixDouble = previousIx;
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
						points[y] = std::round(weightedIso/som) ;
						if (points[y]>=dimx) {
							points[y] = dimx - 1;
						} else if (points[y]<0) {
							points[y] = 0;
						}
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
						traceBuf[index] = seismicBuf[y+index*dimy];
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

	static void run(ImageFormats::QSampleType seismicType, MultiSeedRandomRep* ext, const void* rgtData, const void* seismicData,
					long dimx, long dimy, const std::vector<RgtSeed>& newSeeds, std::vector<RgtSeed>& removedSeeds,
					std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers, QPolygon& poly, bool useSnap,
					bool useMedian, int lwx, const QPolygon& polygon, int distancePower,
					int snapWindow, int polarity, float tdeb, float pasech, long dtauReference,
					std::vector<double>& weights, std::vector<double>& means, std::vector<int>& staticPointCurve,
					std::vector<int>& selectedIndexCurve, const std::vector<bool>& constrainCurve) {
		SampleTypeBinder binder(seismicType);
		binder.bind<UpdateMainHorizonWithCacheKernelLevel2>(ext, rgtData, seismicData,
				dimx, dimy, newSeeds, removedSeeds,
				referenceLayers, poly, useSnap,
				useMedian, lwx, polygon, distancePower,
				snapWindow, polarity, tdeb, pasech, dtauReference,
				weights, means, staticPointCurve,
				selectedIndexCurve, constrainCurve);
	}
};

void MultiSeedRandomRep::updateMainHorizon() { // polygons update
	initRandomReps();

	long width = m_data->seismic()->width();
	long depth = m_data->seismic()->depth();
	long dimx = m_data->seismic()->height();
	QPolygon polygon = dynamic_cast<RandomLineView*>(view())->discreatePolyLine();
	long dimy = polygon.size();
//	if (m_dir==SliceDirection::Inline) { // Z
//		dimy = width;
//	} else if(m_dir==SliceDirection::XLine) { // Y
//		dimy = depth;
//	} else {
//		return;
//	}

	CUDAImagePaletteHolder* rgtData = m_currentRgtImage;
	CUDAImagePaletteHolder* seismicData = m_currentSeismicImage;

	if (rgtData->width()!=seismicData->width() || rgtData->height()!=seismicData->height() ||
		dimx!=rgtData->height()) {
		// error
		qDebug() << "ERROR : MultiSeedRandomRep : random reps and data do not match";
		return;
	}



	// only support short
	// buffer are transposed : y is the quickest, pos = x * dimy + y


	if (m_data->getDTauPolygonMainCache()!=0) {
//		applyDTauToData();
		m_data->applyDtauToSeeds(this);
		correctSeedsFromImage();
		m_polygonMainCache.clear();
		clearCurveSpeedupCache();
	} else if (m_polygonMainCache.size()!=0) {
		m_polygonMainCache.clear();
		clearCurveSpeedupCache();
	}

	m_polygonMain.clear();
	ImageFormats::QSampleType seismicSampleType = m_currentSeismicImage->sampleType();

	SampleTypeBinder binder(rgtData->sampleType());

	const void* rgtBuf = lockRgtCache();
	const void* seismicBuf = lockSeismicCache();

	float tdeb = m_data->seismic()->sampleTransformation()->b();
	float pasech = m_data->seismic()->sampleTransformation()->a();

	std::vector<std::shared_ptr<FixedLayerFromDataset>> references = m_data->getReferences();
	if (m_meansCurve.size()==dimy && m_weightsCurve.size()==dimy && m_staticPointCurve.size()==dimy &&
			m_selectedIndexCurve.size()==dimy && m_constrainCurve.size()==dimy) {
		binder.bind <UpdateMainHorizonWithCacheKernel>(seismicSampleType, this, rgtBuf, seismicBuf, dimx, dimy, m_newPoints, m_removedPoints, references,
			m_polygonMain, m_data->useSnap(), m_data->useMedian(), m_data->getLWXMedianFilter(),  polygon, m_data->getDistancePower(), m_data->getSnapWindow(),
			m_data->getPolarity(), tdeb, pasech,
			m_data->getDTauReference(), m_weightsCurve, m_meansCurve, m_staticPointCurve, m_selectedIndexCurve, m_constrainCurve);
		m_newPoints.clear();
		m_removedPoints.clear();
	} else {
		clearCurveSpeedupCache();
		binder.bind <UpdateMainHorizonKernel>(seismicSampleType, this, rgtBuf, seismicBuf, dimx, dimy, m_data->getMap(), m_data->constrainLayer(), references,
			m_polygonMain, m_data->useSnap(), m_data->useMedian(), m_data->getLWXMedianFilter(),  polygon, m_data->getDistancePower(), m_data->getSnapWindow(),
			m_data->getPolarity(), tdeb, pasech,
			m_data->getDTauReference(), m_weightsCurve, m_meansCurve, m_staticPointCurve, m_selectedIndexCurve, m_constrainCurve);
	}

	for (int n=0; n<m_polygonReference.size(); n++)
	{
		m_polygonReference[n].clear();
	}

//	std::vector<std::shared_ptr<FixedLayerFromDataset>> references = m_data->getReferences();
	//fprintf(stderr, "ici\n");
	int N = references.size();
	m_polygonReference.resize(N);
	for (int n=0; n<N; n++)
	{
		//float *data = references[n];
		CUDAImagePaletteHolder* image = references[n]->image(FixedLayerFromDataset::ISOCHRONE);
		image->lockPointer();
		float* data = static_cast<float*>(image->backingPointer()); // image from FixedLayerFromDataset is always float else use readProperty with a buffer
//		if ( m_dir==SliceDirection::Inline )
//		{
//			for (int y=0; y<width; y++)
//			{
//				m_polygonReference[n] << QPoint(y, (int)((data[width*m_currentSlice+y]-tdeb)/pasech));
//			}
//		}
//		else
//		{
//			for (int z=0; z<depth; z++)
//			{
//				m_polygonReference[n] << QPoint(z, (int)((data[width*z+m_currentSlice]-tdeb)/pasech));
//			}
//		}
		QPolygon currentPolygon;
		for (int idx=0; idx<polygon.size(); idx++)
		{
//			m_polygonReference[n] << QPoint(idx, (int)((data[width*polygon[idx].y()+polygon[idx].x()]-tdeb)/pasech));
			if (data[width*polygon[idx].y()+polygon[idx].x()]!=-9999.0) {
				currentPolygon << QPoint(idx, (int)((data[width*polygon[idx].y()+polygon[idx].x()]-tdeb)/pasech));
			} else if (currentPolygon.size()!=0) {
				m_polygonReference[n].push_back(currentPolygon);
				currentPolygon.clear();
			}
		}
		if (currentPolygon.size()!=0) {
			m_polygonReference[n].push_back(currentPolygon);
		}
		image->unlockPointer();
	}

/*
	m_polygonReference.resize(2);
	m_polygonReference[0].clear();
	m_polygonReference[1].clear();
	for (int i=0; i<1500; i++)
	{
		m_polygonReference[0] << QPoint(i, 150-m_currentSlice);
		m_polygonReference[1] << QPoint(i, 250-m_currentSlice);
	}
*/



	unlockRgtCache();
	unlockSeismicCache();

	if (m_data->getHorizonMode()==DELTA_T) {
		updateDeltaHorizon();
	} else {
		updateGraphicRepresentation();
	}
}

//void MultiSeedRandomRep::applyDTauToData() {
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
//		disconnect(m_data, &MultiSeedHorizon::polarityChanged, this, &MultiSeedRandomRep::updateMainHorizon);
//		m_data->setPolarity((polarityVal>=0) ? 1 : -1);
//		connect(m_data, &MultiSeedHorizon::polarityChanged, this, &MultiSeedRandomRep::updateMainHorizon);
//		// emit signals
////		disconnect(m_data, &MultiSeedHorizon::pointMoved, this, &MultiSeedRandomRep::pointMovedSynchro);
//
//		emit m_data->startApplyDTauTransaction(this);
//		for (std::tuple<RgtSeed, RgtSeed, std::size_t>& myTuple : dataForSignals) {
//			//emit pointMoved(std::get<0>(myTuple), std::get<1>(myTuple), std::get<2>(myTuple));
//			m_data->moveSeed(std::get<2>(myTuple), std::get<1>(myTuple));
//		}
//		m_data->setDTauPolygonMainCache(0);
//		emit m_data->endApplyDTauTransaction(this);
////		connect(m_data, &MultiSeedHorizon::pointMoved, this, &MultiSeedRandomRep::pointMovedSynchro);
//	}
//	m_polygonMainCache.clear();
//}

void MultiSeedRandomRep::updateDeltaHorizon() {
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

QPen MultiSeedRandomRep::getPen() const {
	return m_penMain;
}

void MultiSeedRandomRep::setPen(const QPen& pen) {
	m_penMain = pen;
	updateGraphicRepresentation();
}

QPen MultiSeedRandomRep::getPenDelta() const {
	return m_penDelta;
}

void MultiSeedRandomRep::setPenDelta(const QPen& pen) {
	m_penDelta = pen;
	updateGraphicRepresentation();
}

QPen MultiSeedRandomRep::getPenPoints() const {
	return m_pointsPen;
}

void MultiSeedRandomRep::setPenPoints(const QPen& pen) {
	m_pointsPen = pen;
	updateSeedsRepresentation();
}

void MultiSeedRandomRep::clearCurveSpeedupCache() {
	m_weightsCurve.clear();
	m_meansCurve.clear();
	m_staticPointCurve.clear();
	m_selectedIndexCurve.clear();
	m_constrainCurve.clear();
	m_newPoints.clear();
	m_removedPoints.clear();
}

void MultiSeedRandomRep::newPointCreatedSynchro(RgtSeed seed, int id) {
	if (m_data->getMap().size()>1) {
		m_newPoints.push_back(seed);
	} else {
		clearCurveSpeedupCache();
	}
	updateMainHorizon();
	updateSeedsRepresentation();
}

void MultiSeedRandomRep::pointRemovedSynchro(RgtSeed seed, int id) {
    //m_removedPoints.push_back(seed);
    clearCurveSpeedupCache();
    updateMainHorizon();
    updateSeedsRepresentation();
}

void MultiSeedRandomRep::pointMovedSynchro(RgtSeed oldSeed, RgtSeed newSeed, int id) {
    //m_removedPoints.push_back(oldSeed);
    //m_newPoints.push_back(seed);
    clearCurveSpeedupCache();

    updateMainHorizon();
    updateSeedsRepresentation();
}

void MultiSeedRandomRep::seedsResetSynchro() {
	clearCurveSpeedupCache();
	updateMainHorizon();
	updateSeedsRepresentation();
}

QString MultiSeedRandomRep::name() const {
	return m_data->name();
}

//void MultiSeedRandomRep::setSliceIJPosition(int val) {
//	QMutexLocker lock(&m_polygonMutex);
//	m_currentSlice = val;
//	clearCurveSpeedupCache();
//	updateMainHorizon();
//}

QGraphicsItem * MultiSeedRandomRep::getOverlayItem(DataControler * controler,QGraphicsItem *parent) {
	return nullptr;
}

void MultiSeedRandomRep::notifyDataControlerMouseMoved(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys) {
	if (m_data->getBehaviorMode()!=MOUSETRACKING) {
		return;
	}
	double imageX, imageY;
//	switch (m_dir) {
//	case SliceDirection::Inline:
//		m_data->seismic()->ijToInlineXlineTransfoForInline()->worldToImage(worldX, worldY, imageX, imageY);
//		break;
//	case SliceDirection::XLine:
//		m_data->seismic()->ijToInlineXlineTransfoForXline()->worldToImage(worldX, worldY, imageX, imageY);
//		break;
//	}
	m_data->seismic()->sampleTransformation()->indirect(worldY, imageY);
	imageX = dynamic_cast<RandomLineView*>(view())->getDiscreatePolyLineIndexFromScenePos(QPointF(worldX, worldY));

	QPolygon polygon = dynamic_cast<RandomLineView*>(view())->discreatePolyLine();

	long dimI = m_data->seismic()->height();
	long dimJ = polygon.size();
//	if (m_dir==SliceDirection::Inline) { // Z
//		dimJ = m_data->seismic()->width();
//	} else if(m_dir==SliceDirection::XLine) { // Y
//		dimJ = m_data->seismic()->depth();
//	} else {
//		return;
//	}
	if ((imageX<0 || imageX>=dimJ) && (imageY<0 && imageY>=dimI)) {
		return;
	}

	QPoint pt(imageX, imageY);
	moveTrackingReference(pt);

}

void MultiSeedRandomRep::notifyDataControlerMousePressed(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys) {
	if (m_data->getBehaviorMode()==POINTPICKING && (button & Qt::LeftButton)) {
		initRandomReps();
		double imageX, imageY;
//		if (direction()==SliceDirection::Inline) {
//			dynamic_cast<Seismic3DDataset*>(m_seismic->data())->ijToInlineXlineTransfoForInline()->worldToImage(worldX, worldY, imageX, imageY);
//		} else {
//			dynamic_cast<Seismic3DDataset*>(m_seismic->data())->ijToInlineXlineTransfoForXline()->worldToImage(worldX, worldY, imageX, imageY);
//		}
		m_data->seismic()->sampleTransformation()->indirect(worldY, imageY);
		imageX = dynamic_cast<RandomLineView*>(view())->getDiscreatePolyLineIndexFromScenePos(QPointF(worldX, worldY));
		std::size_t index=0;

		QPolygon polygon = dynamic_cast<RandomLineView*>(view())->discreatePolyLine();

		if (imageX>=polygon.size() || imageX<0 ||
			imageY>=m_currentSeismicImage->height() || imageY<0) {
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

void MultiSeedRandomRep::notifyDataControlerMouseRelease(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys) {

}

void MultiSeedRandomRep::notifyDataControlerMouseDoubleClick(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys) {

}

QGraphicsItem * MultiSeedRandomRep::releaseOverlayItem(DataControler * controler) {
	return nullptr;
}

template<typename InputType>
struct Random_MoveTrackingReference_GetRGTVal_Kernel {
	static void run(long dimJ, long dimy, long dimx, float tdeb, float pasech, long index_ref, const QPolygon& polygon,
			const void* rgtData, std::vector<float>& isochrone, std::vector<ReferenceDuo>& referenceVec) {
		const InputType* rgtBuf = static_cast<const InputType*>(rgtData);
		for (std::size_t _j=0; _j<dimJ; _j++) {
			int index_trace = (isochrone[polygon[_j].y()*dimy+polygon[_j].x()] - tdeb) / pasech;
			referenceVec[index_ref].rgt[polygon[_j].y()*dimy+polygon[_j].x()] = rgtBuf[_j + index_trace*dimJ];
		}
	}
};

template<typename InputType>
struct Random_MoveTrackingReference_GetTauVal_Kernel {
	static double run(const void* rgtBuf, long iy, long iz, long x, long dimJ, long dimx, long dimy,
			std::size_t index, long index_ref, float tdeb, float pasech, const std::vector<int>& traceReferenceLimits,
			const std::vector<ReferenceDuo>& referenceVec, const std::vector<int>& referenceValue, QPoint pt) {
		const InputType* tabRGT = static_cast<const InputType*>(rgtBuf);
		double rgtTopValue = getNewRgtValueFromReference(iy, iz, traceReferenceLimits[index_ref],
						tabRGT[traceReferenceLimits[index_ref]*dimJ + index], tdeb, pasech, dimy,
						referenceVec, referenceValue);
		double rgtBottomValue = getNewRgtValueFromReference(iy, iz, traceReferenceLimits[index_ref+1],
						tabRGT[traceReferenceLimits[index_ref+1]*dimJ + index], tdeb, pasech, dimy,
						referenceVec, referenceValue);
		double rgtInitValue = getNewRgtValueFromReference(iy, iz, x,
						tabRGT[x*dimJ + index], tdeb, pasech, dimy, referenceVec, referenceValue);

		double rgtPointValue;
		if (pt.y()>=traceReferenceLimits[index_ref] && pt.y()<=traceReferenceLimits[index_ref+1]) {
			rgtPointValue = getNewRgtValueFromReference(iy, iz, pt.y(),
						tabRGT[pt.y()*dimJ + pt.x()], tdeb, pasech, dimy, referenceVec, referenceValue);
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
void MultiSeedRandomRep::moveTrackingReference(QPoint pt) {
	std::vector<std::shared_ptr<FixedLayerFromDataset>> referenceLayers = m_data->getReferences();
	if (true/*referenceLayers.size()==0*/) {
		QPolygon polygon = dynamic_cast<RandomLineView*>(view())->discreatePolyLine();
		long dimJ = polygon.size();
//		if (m_dir==SliceDirection::Inline) { // Z
//			dimJ = m_data->seismic()->width();
//		} else if(m_dir==SliceDirection::XLine) { // Y
//			dimJ = m_data->seismic()->depth();
//		} else {
//			return;
//		}
		// find delta tau
		if (m_polygonMainCache.size()==0) {
			m_polygonMainCache = m_polygonMain;
		}
		long x;
		std::size_t index = 0;
		while (index<m_polygonMain.size() && m_polygonMain[index].x()!=pt.x()) {
			index++;
		}
		if (index<m_polygonMain.size() && m_constrainCurve.size()>m_polygonMain[index].x() && !m_constrainCurve[m_polygonMain[index].x()]) {
			x = m_polygonMainCache[index].y();
		} else {
			return;
		}

		//clearCurveSpeedupCache();

		QPoint pointRef(pt.x(), x);
		long imageVal = getValueForSeed(m_rgt, pt.x(), pt.y(), m_data->channelRgt(), m_currentRgtImage);
		long refVal = getValueForSeed(m_rgt, pointRef.x(), pointRef.y(), m_data->channelRgt(), m_currentRgtImage);
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
			/*long x_ceil = x_floor + 1;
			int x_floor_rgt = rgtBuf[x_floor+dimx*y];
			int x_ceil_rgt = rgtBuf[x_ceil+dimx*y];

			double rgt_value = x_floor_rgt + (_x-x_floor) / (x_ceil-x_floor) * (x_ceil_rgt - x_floor_rgt);
			tauRef = rgt_value + dtauPolygonMainCache;*/
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
		QPolygon polygon = dynamic_cast<RandomLineView*>(view())->discreatePolyLine();
		iy = polygon[index].x();
		iz = polygon[index].y();
//		if (m_dir==SliceDirection::Inline) {
//			iy = index;
//			iz = m_currentSlice;
//			dimJ = dimy;
//		} else {
//			iy = m_currentSlice;
//			iz = index;
//			dimJ = dimz;
//		}


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

			SampleTypeBinder binder(m_currentRgtImage->sampleType());
			binder.bind<Random_MoveTrackingReference_GetRGTVal_Kernel>(dimJ, dimy, dimx, tdeb, pasech, index_ref, polygon,
					lockRgtCache(), isochrone, referenceVec);
			unlockRgtCache();
		} while(index_ref+1<referenceLayers.size() && x>indexBottom);

		for (int i=index_ref+1; i<referenceVec.size(); i++) {
			std::vector<float>& isochrone = referenceVec[i].iso;
			isochrone.resize(dimy*dimz);
			referenceVec[i].rgt.resize(dimy*dimz);
			referenceLayers[i]->readProperty(isochrone.data(), isoName);
			traceReferenceLimits[i+1] = indexBottom;
			/*for (std::size_t _j=0; _j<dimJ; _j++) {
//				if (m_dir==SliceDirection::Inline) {
//					int index_trace = (isochrone[iz*dimy+_j] - tdeb) / pasech;
//					referenceVec[i].rgt[iz*dimy+_j] = static_cast<short*>(m_rgt->image()->backingPointer())[_j + index_trace*dimJ];
//				} else {
//					int index_trace = (isochrone[_j*dimy+iy] - tdeb) / pasech;
//					referenceVec[i].rgt[_j*dimy+iy] = static_cast<short*>(m_rgt->image()->backingPointer())[_j + index_trace*dimJ];
//				}
				int index_trace = (isochrone[polygon[_j].y()*dimy+polygon[_j].x()] - tdeb) / pasech;
				referenceVec[i].rgt[polygon[_j].y()*dimy+polygon[_j].x()] = static_cast<short*>(m_rgt->image()->backingPointer())[_j + index_trace*dimJ];
			}*/
			SampleTypeBinder binder(m_currentRgtImage->sampleType());
			binder.bind<Random_MoveTrackingReference_GetRGTVal_Kernel>(dimJ, dimy, dimx, tdeb, pasech, index_ref, polygon,
					lockRgtCache(), isochrone, referenceVec);
			unlockRgtCache();
		}

		if (x>indexBottom) {
			index_ref ++;
		}

		// extract layer "central" value
		std::vector<int> referenceValue;
		referenceValue.resize(referenceLayers.size());
		for (std::size_t _index = 0; _index<((index_ref+1< referenceLayers.size()) ? index_ref+1  : referenceLayers.size()); _index++) {
//			if (m_dir==SliceDirection::Inline) {
//				referenceValue[_index] = referenceVec[_index].rgt[iz*dimy + (dimy/2)];
//			}  else {
//				referenceValue[_index] = referenceVec[_index].rgt[(dimz/2)*dimy + iy];
//			}
			long iyMid = polygon[polygon.size()/2].x();
			long izMid = polygon[polygon.size()/2].y();
			referenceValue[_index] = referenceVec[_index].rgt[(dimz/2)*dimy + iy];
		}

		// check if new point in layer
		SampleTypeBinder binder(m_currentRgtImage->sampleType());
		double dtauRelative = binder.bind<Random_MoveTrackingReference_GetTauVal_Kernel>(lockRgtCache(), iy, iz, x, dimJ,
				dimx, dimy, index, index_ref, tdeb, pasech, traceReferenceLimits, referenceVec, referenceValue, pt);

		unlockRgtCache();
		applyDeltaTauRelative(dtauRelative, index_ref, referenceVec, referenceValue);//, m_rgtVisual->getDimensions());
	}

}

template<typename InputType>
struct UpdateMainHorizonWithShiftKernel {
	static void run(const void* rgtData, long dimx, long dimy, QPolygon& polygonMain, QPolygon polygonMainCache, int dTau,
					int dtauPolygonMainCache, const std::vector<double>& means, const std::vector<double>& weights,
					const std::vector<int>& selectedPoints, const std::vector<int>& staticIndice, const std::vector<bool>& constrainCurve) {
		const InputType* rgtBuf = static_cast<const InputType*>(rgtData);
		for (int index=0; index<polygonMain.size(); index++) {
			//QPoint pt = polygonMain[index];
			long x; //pt.x();
			long y = polygonMain[index].x();
			int tauRef;

			if (staticIndice[y]==0) {
				double _x = means[y] / weights[y];
				long x_floor = std::floor(_x);
				if (x_floor<0) {
					x = 0;
					tauRef = rgtBuf[x*dimy+y] + dtauPolygonMainCache;
				} else if (x_floor>dimx-2) {
					x = dimx-1;
					tauRef = rgtBuf[x*dimy+y] + dtauPolygonMainCache;
				} else {
					long x_ceil = x_floor + 1;
					int x_floor_rgt = rgtBuf[x_floor*dimy+y];
					int x_ceil_rgt = rgtBuf[x_ceil*dimy+y];

					double rgt_value = x_floor_rgt + (_x-x_floor) / (x_ceil-x_floor) * (x_ceil_rgt - x_floor_rgt);
					tauRef = rgt_value + dtauPolygonMainCache;
					x = x_floor;
				}
			} else if (!constrainCurve[y]) {
				x = selectedPoints[y];
				tauRef = rgtBuf[x*dimy+y] + dtauPolygonMainCache;
			} else {
				x = selectedPoints[y];
				tauRef = rgtBuf[x*dimy+y];
			}

			if (dTau>0) {
				x = std::max(x, 0l);
				while (x<dimx && rgtBuf[x*dimy+y]<tauRef) {
					x++;
				}
				x = std::min(x, dimx-1);
			} else {
				x = std::min(x, dimx);
				long oldX = x;
				while (x>=0 && rgtBuf[x*dimy+y]>tauRef) {
					x--;
				}
				if (x<dimx-1 && rgtBuf[x*dimy+y]<tauRef) {
					x++;
				}

			}

			QPoint newPoint(y, x);
			polygonMain[index] = newPoint;
		}
	}
};

template<typename InputType>
struct UpdateMainHorizonWithRelativeKernel {
	static void run(const void* rgtData, Seismic3DAbstractDataset* rgtDataset, QPolygon& polygonMain, double dTauRelative, std::size_t indexLayerBottom,
			const std::vector<double>& means, const std::vector<double>& weights,
			const std::vector<int>& selectedPoints, const std::vector<int>& staticIndice, const std::vector<bool>& constrainCurve,
			const std::vector<ReferenceDuo>& referenceVec,
			const std::vector<int>& referenceValues,
			const QPolygon& polygon) {
		long dimx = rgtDataset->height();
		float tdeb = rgtDataset->sampleTransformation()->b();
		float pasech = rgtDataset->sampleTransformation()->a();
		long dimy = rgtDataset->width();
		long dimz = rgtDataset->depth();
		long dimJ;

		const InputType* rgtBuf = static_cast<const InputType*>(rgtData);
		for (int index=0; index<polygonMain.size(); index++) {
			//QPoint pt = polygonMain[index];
			long x; //pt.x();
			long y = polygonMain[index].x();
			double tauRef;

			// get point position in map
			long iy, iz;
//			if (direction==SliceDirection::Inline) {
//				iy = y;
//				iz = currentSlice;
//				dimJ = dimy;
//			} else {
//				iy = currentSlice;
//				iz = y;
//				dimJ = dimz;
//			}
			dimJ = polygon.size();
			iy = polygon[y].x();
			iz = polygon[y].y();

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
			top_rgt = getNewRgtValueFromReference(iy, iz, top_index, rgtBuf[top_index*dimJ + y], tdeb, pasech, dimy, referenceVec, referenceValues);
			bottom_rgt = getNewRgtValueFromReference(iy, iz, bottom_index, rgtBuf[bottom_index*dimJ + y], tdeb, pasech, dimy, referenceVec, referenceValues);

			double dTau = dTauRelative * (bottom_rgt - top_rgt);

			if (staticIndice[y]==0) {
				double _x = means[y] / weights[y];
				long x_floor = std::floor(_x);
				long x_ceil = x_floor + 1;
				int x_floor_rgt = getNewRgtValueFromReference(iy, iz, x_floor, rgtBuf[x_floor*dimJ+y], tdeb, pasech, dimy, referenceVec, referenceValues);
				int x_ceil_rgt = getNewRgtValueFromReference(iy, iz, x_ceil, rgtBuf[x_ceil*dimJ+y], tdeb, pasech, dimy, referenceVec, referenceValues);

				double rgt_value = x_floor_rgt + (_x-x_floor) / (x_ceil-x_floor) * (x_ceil_rgt - x_floor_rgt);
				tauRef = rgt_value + dTau;
				x = x_floor;
			} else if (!constrainCurve[y]) {
				x = selectedPoints[y];
				tauRef = rgtBuf[x*dimJ+y] + dTau;
			} else {
				x = selectedPoints[y];
				tauRef = rgtBuf[x*dimJ+y];
			}
			if (tauRef>bottom_rgt) {
				tauRef = bottom_rgt;
			}
			if (tauRef<top_rgt) {
				tauRef = top_rgt;
			}

			if (dTau>0) {
				x = std::max(x, 0l);
				while (x<dimx && getNewRgtValueFromReference(iy, iz, x, rgtBuf[x*dimJ+y], tdeb, pasech, dimy, referenceVec, referenceValues)<tauRef) {
					x++;
				}
				x = std::min(x, dimx-1);
			} else {
				x = std::min(x, dimx);
				long oldX = x;
				while (x>=0 && getNewRgtValueFromReference(iy, iz, x, rgtBuf[x*dimJ+y], tdeb, pasech, dimy, referenceVec, referenceValues)>tauRef) {
					x--;
				}
				if (x<dimx-1 && getNewRgtValueFromReference(iy, iz, x, rgtBuf[x*dimJ+y], tdeb, pasech, dimy, referenceVec, referenceValues)<tauRef) {
					x++;
				}

			}

			QPoint newPoint(y, x);
			polygonMain[index] = newPoint;
		}
	}
};

void MultiSeedRandomRep::applyDeltaTauRelative(double dTau, std::size_t indexLayerBottom, const std::vector<ReferenceDuo>& referenceVec,
		const std::vector<int>& referenceValues) {
	// commented because it block the display if the main horizon != cache horizon and dtau == 0
//	if (dTau==0) {
//		return;
//	}

	long dimx = m_data->seismic()->height();

	//m_dtauReference += dTau;
	m_data->setDTauRelativePolygonMainCache(m_data->getDTauRelativePolygonMainCache() + dTau);

	// refresh gui
	if (m_data->useSnap()) {
		updateMainHorizon();
	} else {
		QPolygon polygon = dynamic_cast<RandomLineView*>(view())->discreatePolyLine();
		/*if (m_dtauPolygonMainCache==0) {
			m_polygonMainCache = m_polygonMain;
		}*/

//		AccesserCachedImage& rgtData = m_rgtVisual->getDisplayedImageAccesser();
		SampleTypeBinder binder(m_currentRgtImage->sampleType());
		binder.bind<UpdateMainHorizonWithRelativeKernel> (lockRgtCache(), m_data->rgt(), m_polygonMain, dTau, indexLayerBottom,
				m_meansCurve, m_weightsCurve, m_selectedIndexCurve, m_staticPointCurve, m_constrainCurve, referenceVec, referenceValues,
				polygon);
		unlockRgtCache();


		if (m_data->getHorizonMode()==DELTA_T) {
			updateDeltaHorizon();
		} else {
			updateGraphicRepresentation();
		}
	}
}


void MultiSeedRandomRep::applyDeltaTau(int dTau) {
	initRandomReps();

	// commented because it block the display if the main horizon != cache horizon and dtau == 0
//	if (dTau==0) {
//		return;
//	}

	long dimx = m_data->seismic()->width();

	//m_data->setDTauReference(dTau);
	m_data->setDTauPolygonMainCache(dTau);

	// refresh gui
	if (m_data->useSnap()) {
		updateMainHorizon();
	} else {
		/*if (m_dtauPolygonMainCache==0) {
				m_polygonMainCache = m_polygonMain;
		}*/
		//m_data->setDTauPolygonMainCache(dTau);

		//AccesserCachedImage& rgtData = m_rgtVisual->getDisplayedImageAccesser();
		long imageDimX = m_currentRgtImage->height();
		long imageDimY = m_currentRgtImage->width();
		SampleTypeBinder binder(m_currentRgtImage->sampleType());
		binder.bind <UpdateMainHorizonWithShiftKernel> (lockRgtCache(), imageDimX, imageDimY,
						m_polygonMain, m_polygonMainCache, dTau, m_data->getDTauPolygonMainCache(),
						m_meansCurve, m_weightsCurve, m_selectedIndexCurve, m_staticPointCurve, m_constrainCurve);


		unlockRgtCache();

		if (m_data->getHorizonMode()==DELTA_T) {
			updateDeltaHorizon();
		} else {
			updateGraphicRepresentation();
		}
	}
}

void MultiSeedRandomRep::initRandomReps() {
	RandomLineView* view = dynamic_cast<RandomLineView*>(m_parent);
	if ((m_rgt==nullptr || m_seismic==nullptr) && view!=nullptr) {
		std::pair<RandomRep*, RandomRep*> out = findRandomRepsFromRandomInnerViewAndData(m_data, view);
		m_seismic = out.first;
		m_rgt = out.second;


		if (m_seismic!=nullptr && m_seismic->image()==nullptr) {
			m_seismic = nullptr;
		}
		if (m_rgt!=nullptr && m_rgt->image()==nullptr) {
			m_rgt = nullptr;
		}

		if (m_seismic!=nullptr){
			connect(m_seismic, &RandomRep::layerHidden, this, &MultiSeedRandomRep::unsetSeismicRep);
			connect(m_seismic, &RandomRep::destroyed, this, &MultiSeedRandomRep::unsetSeismicRep);
		}

		if(m_rgt != nullptr){
			connect(m_rgt, &RandomRep::layerHidden, this, &MultiSeedRandomRep::unsetRgtRep);
			connect(m_rgt, &RandomRep::destroyed, this, &MultiSeedRandomRep::unsetRgtRep);
		}
	}
	updateInternalImages();
}

template<typename InputType>
void MultiSeedRandomRep::CorrectSeedsFromImageKernel<InputType>::run(const void* rgtData,
		MultiSeedRandomRep* obj, std::size_t dimI, std::size_t dimJ,
		QList<std::tuple<RgtSeed, RgtSeed, std::size_t>>& seedsChangeList, const QPolygon& polygon) {
	const InputType* rgtBuf = static_cast<const InputType*>(rgtData);

	const std::map<std::size_t, RgtSeed>& seeds = obj->m_data->getMap();
	// Detect seeds in image and change their x if needed
	#pragma omp parallel for
	for(long i=0; i<seeds.size(); i++) {
		std::map<std::size_t, RgtSeed>::const_iterator seedPairIt = std::begin(seeds);
		std::advance(seedPairIt, i);
		const RgtSeed& seed = seedPairIt->second;
		long index = 0;
		while (index<polygon.size() && polygon[index]!=QPoint(seed.y, seed.z)) {
			index++;
		}
		if (index<polygon.size()) {
			// seed in image
			long ix = seed.x;
			long iy = index;
			double dtauReference = seed.rgtValue - rgtBuf[ix*dimJ + iy];
			if (dtauReference>0) {
				ix = std::max(ix, 0l);
				while (ix<dimI && rgtBuf[ix*dimJ+iy]<rgtBuf[seed.x*dimJ+iy]+dtauReference) {
					ix++;
				}
				ix = std::min(ix, static_cast<long>(dimI-1));
			} else if(dtauReference<0) {
				ix = std::min(ix, static_cast<long>(dimI));
				long oldX = ix;
				while (ix>=0 && rgtBuf[ix*dimJ+iy]>rgtBuf[seed.x*dimJ+iy]+dtauReference) {
					ix--;
				}
				if (ix<dimI-1 && rgtBuf[ix*dimJ+iy]<rgtBuf[seed.x*dimJ+iy]+dtauReference) {
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

void MultiSeedRandomRep::correctSeedsFromImage() {
	initRandomReps(); // needed before accessing buffers

	CUDAImagePaletteHolder* rgtData = m_currentRgtImage;
	CUDAImagePaletteHolder* seismicData = m_currentSeismicImage;

	std::size_t dimJ = rgtData->width();
	std::size_t dimI = rgtData->height();

	SampleTypeBinder binder(m_currentRgtImage->sampleType());

	// lock buffers
	const void* rgtBuf = lockRgtCache();

	// pasech and tdeb are not needed as references and constrains do not matter here

	const QPolygon& polygon = dynamic_cast<RandomLineView*>(view())->discreatePolyLine();
	QList<std::tuple<RgtSeed, RgtSeed, std::size_t>> seedsChangeList;

	binder.bind<CorrectSeedsFromImageKernel>(rgtBuf,
			this, dimI, dimJ, seedsChangeList, polygon);

	//unlock buffers
	unlockRgtCache();

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

bool MultiSeedRandomRep::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> MultiSeedRandomRep::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_data->seismic()->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString MultiSeedRandomRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}



template<typename InputType>
struct GetValueFromSliceRepForSeedKernel {
	static int run(const void* _tab, std::size_t idx) {
		InputType* tab = (InputType*) _tab;
		return tab[idx];
	}
};

int MultiSeedRandomRep::getValueFromSliceRepForSeed(RandomRep* rep, std::size_t x, std::size_t y, std::size_t channel) {
	std::size_t dimJ = rep->image()->height();
	std::size_t dimI = rep->image()->width();
	std::size_t idx = x + y * dimI + channel * dimI * dimJ;
	const std::vector<char>& randomRepCache = rep->lockCache();
	SampleTypeBinder binder(rep->image()->sampleType());
	int outVal = binder.bind<GetValueFromSliceRepForSeedKernel>((const void*) randomRepCache.data(), idx);
	rep->unlockCache();
	return outVal;
}

int MultiSeedRandomRep::getValueForSeed(RandomRep* rep, std::size_t x, std::size_t y, std::size_t channel,
		CUDAImagePaletteHolder* cacheImage) {
	int out;
	if (rep) {
		out = getValueFromSliceRepForSeed(rep, x, y, channel);
	} else {
		double val;
		cacheImage->valueAt(x, y, val);
		out = val;
	}
	return out;
}

CUDAImagePaletteHolder* MultiSeedRandomRep::seismicImage() const {
	return m_currentSeismicImage;
}

CUDAImagePaletteHolder* MultiSeedRandomRep::rgtImage() const {
	return m_currentRgtImage;
}

void MultiSeedRandomRep::updateInternalImages() {
	bool oldOwnSeismic = m_ownSeismicImage;
	bool oldOwnRgt = m_ownRgtImage;
	updateInternalImage(m_seismic, m_data->seismic(), m_currentSeismicImage, m_ownSeismicImage);
	updateInternalImage(m_rgt, m_data->rgt(), m_currentRgtImage, m_ownRgtImage);

	RandomLineView* randomView = dynamic_cast<RandomLineView*>(view());
	QPolygon discreatePolygon = randomView->discreatePolyLine();

	bool bIsPolygonChanged = false;
    if(m_cacheDiscreatePolygon != discreatePolygon){
        bIsPolygonChanged = true;
	}

	if ((!oldOwnSeismic && m_ownSeismicImage)
	        || ((bIsPolygonChanged && m_ownSeismicImage) ) ) {
		reloadSeismicBuffer();
	}

	if ((!oldOwnRgt && m_ownRgtImage)
	       || ((bIsPolygonChanged && m_ownRgtImage))) {
		reloadRgtBuffer();
	}

	m_cacheDiscreatePolygon = discreatePolygon;
}

void MultiSeedRandomRep::updateInternalImage(RandomRep* randomRep, Seismic3DAbstractDataset* dataset,
		CUDAImagePaletteHolder*& image, bool& ownCurrentImage) {
	if (randomRep && ownCurrentImage) {
		image->deleteLater();
		ownCurrentImage = false;
		image = randomRep->image();
	} else if (randomRep && !ownCurrentImage) {
		image = randomRep->image();
	} else if (!randomRep) {
		RandomLineView* randomView = dynamic_cast<RandomLineView*>(view());
		QPolygon discreatePolygon = randomView->discreatePolyLine();

		if((!ownCurrentImage) || ((image != nullptr) && (image->width() != discreatePolygon.size()))){
		    const AffineTransformation* sampleTransform = dataset->sampleTransformation();
		    std::array<double, 6> transform;

		    transform[0]=0;
		    transform[1]=1;
		    transform[2]=0;

		    transform[3]=sampleTransform->b();
		    transform[4]=0;
		    transform[5]=sampleTransform->a();


		    Affine2DTransformation* transformation = new Affine2DTransformation(discreatePolygon.size(), dataset->height(), transform, this);
		    image = new CUDAImagePaletteHolder(
		            discreatePolygon.size(), dataset->height(),
		            dataset->sampleType(),
		            transformation, randomView);
		    ownCurrentImage = true;
		}
	}
}

void MultiSeedRandomRep::reloadSeismicBuffer() {
	RandomLineView* randomView = dynamic_cast<RandomLineView*>(view());
	if (m_ownSeismicImage && randomView) {
		m_data->seismic()->loadRandomLine(m_currentSeismicImage, randomView->discreatePolyLine(), m_data->channelSeismic());
	}
}

void MultiSeedRandomRep::reloadRgtBuffer() {
	RandomLineView* randomView = dynamic_cast<RandomLineView*>(view());
	if (m_ownRgtImage && randomView) {
		m_data->rgt()->loadRandomLine(m_currentRgtImage, randomView->discreatePolyLine(), m_data->channelRgt());
	}
}

const void* MultiSeedRandomRep::lockSeismicCache() const {
	const void* seismicBuf = nullptr;
	m_seismicMutex.lock();
	if (m_seismic) {
		m_seismicSaveRep = m_seismic;

		std::size_t width = m_currentSeismicImage->width();
		std::size_t height = m_currentSeismicImage->height();

		seismicBuf = static_cast<const void*>(m_seismic->lockCache().data() + m_data->channelSeismic() *
				width * height * m_data->seismic()->sampleType().byte_size());
		m_seismicLockState = LOCKSTATE::RANDOMREP;
	} else {
		m_seismicSaveImage = m_currentSeismicImage;
		// care about unlocking the pointer
		m_currentSeismicImage->lockPointer();
		seismicBuf = m_currentSeismicImage->backingPointer();
		m_seismicLockState = LOCKSTATE::IMAGE;
	}
	return seismicBuf;
}

void MultiSeedRandomRep::unlockSeismicCache() const {
	if (m_seismicLockState==LOCKSTATE::RANDOMREP) {
		m_seismicSaveRep->unlockCache();
		m_seismicSaveRep = nullptr;
		m_seismicLockState = LOCKSTATE::NOLOCK;
	} else if (m_seismicLockState==LOCKSTATE::IMAGE) {
		m_seismicSaveImage->unlockPointer();
		m_seismicSaveImage = nullptr;
		m_seismicLockState = LOCKSTATE::NOLOCK;
	}
	m_seismicMutex.unlock();
}

const void* MultiSeedRandomRep::lockRgtCache() const {
	const void* rgtBuf = nullptr;
	m_rgtMutex.lock();
	if (m_rgt) {
		m_rgtSaveRep = m_rgt;

		std::size_t width = m_currentRgtImage->width();
		std::size_t height = m_currentRgtImage->height();

		rgtBuf = static_cast<const void*>(m_rgt->lockCache().data() + m_data->channelRgt() *
				width * height * m_data->rgt()->sampleType().byte_size());
		m_rgtLockState = LOCKSTATE::RANDOMREP;
	} else {
		m_rgtSaveImage = m_currentRgtImage;
		// care about unlocking the pointer
		m_currentRgtImage->lockPointer();
		rgtBuf = m_currentRgtImage->backingPointer();
		m_rgtLockState = LOCKSTATE::IMAGE;
	}
	return rgtBuf;
}

void MultiSeedRandomRep::unlockRgtCache() const {
	if (m_rgtLockState==LOCKSTATE::RANDOMREP) {
		m_rgtSaveRep->unlockCache();
		m_rgtSaveRep = nullptr;
		m_rgtLockState = LOCKSTATE::NOLOCK;
	} else if (m_rgtLockState==LOCKSTATE::IMAGE) {
		m_rgtSaveImage->unlockPointer();
		m_rgtSaveImage = nullptr;
		m_rgtLockState = LOCKSTATE::NOLOCK;
	}
	m_rgtMutex.unlock();
}

void MultiSeedRandomRep::updateSeismicAfterSwap() {
	{
		QMutexLocker b(&m_seismicMutex);

		if (m_seismic) {
			disconnect(m_seismic, &RandomRep::layerHidden, this, &MultiSeedRandomRep::unsetSeismicRep);
			disconnect(m_seismic, &RandomRep::destroyed, this, &MultiSeedRandomRep::unsetSeismicRep);
		}

		m_seismic = nullptr;
		RandomLineView* view = dynamic_cast<RandomLineView*>(m_parent);
		if (view!=nullptr) {
			std::pair<RandomRep*, RandomRep*> out = findRandomRepsFromRandomInnerViewAndData(m_data, view);
			// only use seismic response, keep the same rgt
			m_seismic = out.first;


			if (m_seismic) {
				connect(m_seismic, &RandomRep::layerHidden, this, &MultiSeedRandomRep::unsetSeismicRep);
				connect(m_seismic, &RandomRep::destroyed, this, &MultiSeedRandomRep::unsetSeismicRep);
			}
		}
		updateInternalImages();
		clearCurveSpeedupCache();
	}
	updateMainHorizon();
}

void MultiSeedRandomRep::updateRgtAfterSwap() {
	{
		QMutexLocker b(&m_rgtMutex);

		if (m_rgt) {
			disconnect(m_rgt, &RandomRep::layerHidden, this, &MultiSeedRandomRep::unsetRgtRep);
			disconnect(m_rgt, &RandomRep::destroyed, this, &MultiSeedRandomRep::unsetRgtRep);
		}

		m_rgt = nullptr;
		RandomLineView* view = dynamic_cast<RandomLineView*>(m_parent);
		if (view!=nullptr) {
			std::pair<RandomRep*, RandomRep*> out = findRandomRepsFromRandomInnerViewAndData(m_data, view);
			// only use rgt response, keep the same seismic
			m_rgt = out.second;

			if (m_rgt) {
				connect(m_rgt, &RandomRep::layerHidden, this, &MultiSeedRandomRep::unsetRgtRep);
				connect(m_rgt, &RandomRep::destroyed, this, &MultiSeedRandomRep::unsetRgtRep);
			}
		}
		updateInternalImages();
		clearCurveSpeedupCache();
	}
	updateMainHorizon();
}

AbstractGraphicRep::TypeRep MultiSeedRandomRep::getTypeGraphicRep() {
    return AbstractGraphicRep::Courbe;
}

void MultiSeedRandomRep::deleteLayer() {
    if (m_layer != nullptr){
        delete m_layer;
        m_layer = nullptr;
    }
}

void MultiSeedRandomRep::unsetSeismicRep() {
   if (m_seismic!=nullptr) {
       QMutexLocker b(&m_seismicMutex);
       disconnect(m_seismic, &RandomRep::layerHidden, this, &MultiSeedRandomRep::unsetSeismicRep);
       disconnect(m_seismic, &RandomRep::destroyed, this, &MultiSeedRandomRep::unsetSeismicRep);
       m_seismic = nullptr;
   }
}

void MultiSeedRandomRep::unsetRgtRep() {
   if (m_rgt!=nullptr) {
       QMutexLocker b(&m_rgtMutex);
       disconnect(m_rgt, &RandomRep::layerHidden, this, &MultiSeedRandomRep::unsetRgtRep);
       disconnect(m_rgt, &RandomRep::destroyed, this, &MultiSeedRandomRep::unsetRgtRep);
       m_rgt = nullptr;
   }
}
