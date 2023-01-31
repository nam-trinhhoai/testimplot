#include "multiseedhorizon.h"
#include "multiseedhorizongraphicrepfactory.h"

#include "slicerep.h"
#include "fixedlayerfromdataset.h"
#include "seismic3ddataset.h"
#include "textcolortreewidgetitemdecorator.h"

#include <cmath>

std::size_t MultiSeedHorizon::INVALID_ID = 0;

MultiSeedHorizon::MultiSeedHorizon(QString name, WorkingSetManager *workingSet,
		Seismic3DDataset *seismic, int channelS, Seismic3DDataset *rgt, int channelT,
		QObject *parent) : EditableData(workingSet, parent) {
	m_rgt = rgt;
	m_channelRgt = channelT;
	m_seismic = seismic;
	m_channelSeismic = channelS;
	m_name = name;
	m_uuid = QUuid::createUuid();
	m_repFactory = new MultiSeedHorizonGraphicRepFactory(this);

	m_decorator = nullptr;
}

MultiSeedHorizon::~MultiSeedHorizon() {}

QString MultiSeedHorizon::name() const {
	return m_name;
}

QUuid MultiSeedHorizon::dataID() const {
	return m_uuid;
}

IGraphicRepFactory* MultiSeedHorizon::graphicRepFactory() {
	return m_repFactory;
}

HORIZONMODE MultiSeedHorizon::getHorizonMode() const {
	return m_horizonMode;
}

void MultiSeedHorizon::setHorizonMode(HORIZONMODE mode) {
	if (m_horizonMode!=mode) {
		m_horizonMode = mode;
		horizonModeChanged(m_horizonMode);
	}
}

BEHAVIORMODE MultiSeedHorizon::getBehaviorMode() const {
	return m_behaviorMode;
}

void MultiSeedHorizon::setBehaviorMode(BEHAVIORMODE mode) {
	if (m_behaviorMode!=mode) {
		m_behaviorMode = mode;
		behaviorModeChanged(m_behaviorMode);
	}
}

void MultiSeedHorizon::setRgtPickingMode(RGTPICKINGMODE mode)
{
	m_rgtPickingMode = mode;
}

RGTPICKINGMODE MultiSeedHorizon::getRgtPickingMode() const
{
	return m_rgtPickingMode;
}

/*
void MultiSeedHorizon::setRgtVolumicDialog(RgtVolumicDialog *rgtVolumicDialog)
{
	m_rgtVolumicDialog = rgtVolumicDialog;
}

RgtVolumicDialog *MultiSeedHorizon::getRgtVolumicDialog()
{
	return m_rgtVolumicDialog;
}
*/

// seed management
std::size_t MultiSeedHorizon::addPoint(RgtSeed seed) { // add point, can be used for 3D point
	std::size_t id = m_nextId;
	m_nextId ++;
	m_seeds[id] = seed;
	if (m_seeds.size()==1) {
		m_polarity = (seed.seismicValue>=0) ? 1 : -1;
	}
	//m_newPoints.push_back(seed);
	//updateMainHorizon();
	//updateSeedsRepresentation();
	emit newPointCreated(seed, id);
	return id;
}


std::size_t MultiSeedHorizon::addPointAndSelect(RgtSeed seed) {
	std::size_t id = addPoint(seed);
	m_currentSeed = id;
	return id;
}

int MultiSeedHorizon::pointCount() const {
	return m_seeds.size();
}

std::size_t MultiSeedHorizon::getSelectedId() const {
	return m_currentSeed;
}

RgtSeed MultiSeedHorizon::getSelectedPoint() const {
	RgtSeed seed;
	if (m_currentSeed!=INVALID_ID) {
		seed = m_seeds.at(m_currentSeed);
	}
	return seed;
}

RgtSeed& MultiSeedHorizon::operator[](const std::size_t& id) {
	return m_seeds.at(id);
}

RgtSeed& MultiSeedHorizon::operator[](std::size_t&& id) {
	return m_seeds.at(id);
}

const RgtSeed& MultiSeedHorizon::operator[](const std::size_t& id) const {
	return m_seeds.at(id);
}

const RgtSeed& MultiSeedHorizon::operator[](std::size_t&& id) const {
	return m_seeds.at(id);
}

bool MultiSeedHorizon::selectPoint(std::size_t id) { // if index invalid return false and m_currentSeed = 0;
	bool out = false;
	if (id!=INVALID_ID) {
		std::map<std::size_t, RgtSeed>::const_iterator it = m_seeds.begin();
		while(it!=m_seeds.end() && (*it).first!=id) {
			it++;
		}
		out = it!=m_seeds.end();
		if (out) {
			m_currentSeed = id;
		} else {
			m_currentSeed = INVALID_ID;
		}
	} else {
		m_currentSeed = INVALID_ID;
	}
	return out;
}

bool MultiSeedHorizon::removePoint(std::size_t id) {
	bool out = false;
	if (id!=INVALID_ID) {
		std::map<std::size_t, RgtSeed>::const_iterator it = m_seeds.begin();
		while(it!=m_seeds.end() && (*it).first!=id) {
			it++;
		}
		out = it!=m_seeds.end();
		if (out) {
			if (id==m_currentSeed) {
				m_currentSeed = INVALID_ID;
			}
			RgtSeed seed = (*it).second;
			m_seeds.erase(it);
			//clearCurveSpeedupCache();
			//updateMainHorizon();
			//updateSeedsRepresentation();
			emit pointRemoved(seed, id);
		}
	}
	return out;
}

void MultiSeedHorizon::moveSelectedSeed(RgtSeed newSeed) { // move point and update
	if (m_currentSeed==INVALID_ID) {
		return;
	}

	RgtSeed oldSeed = m_seeds.at(m_currentSeed);
	m_seeds[m_currentSeed] = newSeed;

	if (m_seeds.size()==1) {
		m_polarity = (newSeed.seismicValue>=0) ? 1 : -1;
	}
	//clearCurveSpeedupCache();

	//updateMainHorizon();
	//updateSeedsRepresentation();
	emit pointMoved(oldSeed, newSeed, m_currentSeed);
}

void MultiSeedHorizon::moveSeed(std::size_t id, RgtSeed newSeed) { // move point and update
	RgtSeed oldSeed = m_seeds.at(id);
	m_seeds[id] = newSeed;

	if (m_seeds.size()==1) {
		m_polarity = (newSeed.seismicValue>=0) ? 1 : -1;
	}
	//clearCurveSpeedupCache();

	//updateMainHorizon();
	//updateSeedsRepresentation();
	emit pointMoved(oldSeed, newSeed, id);
}

std::vector<std::size_t> MultiSeedHorizon::getAllIds() const {
	std::vector<std::size_t> ids;
	ids.resize(m_seeds.size());
	std::map<std::size_t, RgtSeed>::const_iterator it = m_seeds.begin();
	long index = 0;
	while (it!=m_seeds.end()) {
		ids[index] = (*it).first;
		index++;
		it++;
	}
	return ids;
}

const std::map<std::size_t, RgtSeed>& MultiSeedHorizon::getMap() const {
	return m_seeds;
}

//const std::vector<std::vector<RgtSeed>>& MultiSeedHorizon::getMap2() const {
//	return m_seeds2;
//}

void MultiSeedHorizon::setSeeds(const std::vector<RgtSeed>& seeds) {
	m_seeds.clear();
	for (const RgtSeed& seed : seeds) { // add new points with signals
		//addPoint(seed);
		std::size_t id = m_nextId;
		m_nextId ++;
		m_seeds[id] = seed;
		if (m_seeds.size()==1) {
			m_polarity = (seed.seismicValue>=0) ? 1 : -1;
		}
	}
	//clearCurveSpeedupCache();
	//updateMainHorizon();
	//updateSeedsRepresentation();

	emit seedsReset();
}

std::vector<std::shared_ptr<FixedLayerFromDataset>> MultiSeedHorizon::getReferences()
{
	return m_references;
}

// jd
//void MultiSeedHorizon::setSeeds2(const std::vector<std::vector<RgtSeed>>& seeds) {
//	m_seeds2.clear();
//	std::size_t id = 0;
//
//	m_seeds2.resize(seeds.size());
//	for (int i=0; i<seeds.size(); i++)
//	{
//		m_seeds2[i].resize(seeds[i].size());
//		for (int k=0; k<seeds[i].size(); k++)
//		{
//			m_seeds2[i][k] = seeds[i][k];
//		}
//
//	}
//
//	emit seedsReset();
//}


void MultiSeedHorizon::setReferences(const std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceData)
{
	m_references = referenceData;
	emit referencesChanged();
}

bool MultiSeedHorizon::useSnap() const {
	return m_useSnap;
}

void MultiSeedHorizon::toggleSnap(bool val) {
	if (m_useSnap!=val) {
		m_useSnap = val;
		emit useSnapChanged(m_useSnap);
		//updateMainHorizon();
	}
}

int MultiSeedHorizon::getSnapWindow() const {
	return m_snapWindow;
}

void MultiSeedHorizon::setSnapWindow(int val) {
	if (val!=m_snapWindow) {
		m_snapWindow = val;
		snapWindowChanged(m_snapWindow);
		//updateMainHorizon();
	}
}

bool MultiSeedHorizon::useMedian() const {
	return m_useMedian;
}

void MultiSeedHorizon::toggleMedian(bool val) {
	if (m_useMedian!=val) {
		m_useMedian = val;

		emit useMedianChanged(m_useMedian);
		//updateMainHorizon();
	}
}

int MultiSeedHorizon::getDistancePower() const {
	return m_distancePower;
}

void MultiSeedHorizon::setDistancePower(int val) {
	if (m_distancePower!=val) {
		m_distancePower = val;

		emit distancePowerChanged(m_distancePower);
		//updateMainHorizon();
	}
}

int MultiSeedHorizon::getLWXMedianFilter() const {
	return m_lwx;
}

void MultiSeedHorizon::setLWXMedianFilter(int val) {
	if (m_lwx!=val) {
		m_lwx = val;
		emit lwxMedianFilterChanged(m_lwx);
		/*if (m_useMedian) {
			updateMainHorizon();
		}*/
	}
}

int MultiSeedHorizon::getPolarity() const {
	return m_polarity;
}

void MultiSeedHorizon::setPolarity(int pol) {
	if (pol!=m_polarity) {
		m_polarity = pol;
		//updateMainHorizon();
		emit polarityChanged(m_polarity);
	}
}

void MultiSeedHorizon::setDelta(int d) { // give symetric deltas for top and bottom, ex: bottom <- abs(delta); top <- -abs(delta)
	int newDTop = -std::abs(d);
	int newDBottom = std::abs(d);
	if (newDTop!=m_deltaTop || newDBottom!=m_deltaBottom) {
		m_deltaTop = newDTop;
		m_deltaBottom = newDBottom;

		emit deltaChanged(m_deltaTop, m_deltaBottom);
		/*if (m_horizonMode!=DEFAULT) {
			updateDeltaHorizon();
		}*/
	}
}

int MultiSeedHorizon::getDeltaTop() const {
	return m_deltaTop;
}

void MultiSeedHorizon::setDeltaTop(int d) {
	if (d!=m_deltaTop) {
		m_deltaTop = d;

		emit deltaChanged(m_deltaTop, m_deltaBottom);
		/*if (m_horizonMode!=DEFAULT) {
			updateDeltaHorizon();
		}*/
	}
}
int MultiSeedHorizon::getDeltaBottom() const {
	return m_deltaBottom;
}

void MultiSeedHorizon::setDeltaBottom(int d) {
	if (d!=m_deltaBottom) {
		m_deltaBottom = d;

		emit deltaChanged(m_deltaTop, m_deltaBottom);
		/*if (m_horizonMode!=DEFAULT) {
			updateDeltaHorizon();
		}*/
	}
}

int MultiSeedHorizon::getPseudoTau() {
	double val = 0;
	for (std::pair<std::size_t, RgtSeed> e : m_seeds) {
			val += e.second.rgtValue;
	}
	val /= m_seeds.size();
	return static_cast<int>(std::roundl(val));
}

long MultiSeedHorizon::getDTauReference() {
	return m_dtauReference;
}

void MultiSeedHorizon::setDTauReference(long dtau) {
	if (dtau!=m_dtauReference) {
//		setConstrainLayer(nullptr);

		// old behavior no longer needed
//		qDebug() << "dtau modified" << dtau;
//		m_dtauReference = dtau;
//		emit dtauReferenceChanged(m_dtauReference);
	}
}

void MultiSeedHorizon::setConstrainLayer(FixedLayerFromDataset* layer) {
	if (layer!=m_constrainLayer) {
		if (m_constrainLayer) {
			disconnect(m_constrainLayer, &FixedLayerFromDataset::propertyModified, this, &MultiSeedHorizon::constrainBufferUpdated);
		}

		m_constrainLayer = layer;
		emit constrainChanged();

		if (m_constrainLayer) {
			connect(m_constrainLayer, &FixedLayerFromDataset::propertyModified, this, &MultiSeedHorizon::constrainBufferUpdated);
		}
	}
}

FixedLayerFromDataset* MultiSeedHorizon::constrainLayer() {
	return m_constrainLayer;
}

long MultiSeedHorizon::getDTauPolygonMainCache() const {
	return m_dtauPolygonMainCache;
}

void MultiSeedHorizon::setDTauPolygonMainCache(long dtau) {
	if (dtau!=m_dtauPolygonMainCache) {
		qDebug() << "dtau cache" << m_dtauReference << dtau << m_dtauPolygonMainCache;
		long oldTau = m_dtauPolygonMainCache;
		m_dtauPolygonMainCache = dtau;
		setDTauReference(m_dtauReference + dtau - oldTau);
		emit dtauPolygonMainCacheChanged(m_dtauPolygonMainCache);
	}
}

long MultiSeedHorizon::getDTauRelativePolygonMainCache() const {
	return m_dtauRelativePolygonMainCache;
}

void MultiSeedHorizon::setDTauRelativePolygonMainCache(long dtau) {
	if (dtau!=m_dtauRelativePolygonMainCache) {
		m_dtauRelativePolygonMainCache = dtau;
	}
}

void MultiSeedHorizon::setPseudoTau(int newTau) {
	int oldTau = getPseudoTau();
	if (m_constrainLayer!=nullptr) {
		setDTauReference(newTau - oldTau);
	}
	setDTauPolygonMainCache(newTau - oldTau);
}

bool MultiSeedHorizon::isMultiSeed() const {
	return m_isMultiSeed;
}

void MultiSeedHorizon::setMultiSeed(bool isMultiSeed) {
	m_isMultiSeed = isMultiSeed;
}

void MultiSeedHorizon::applyDtauToSeeds(QObject* requestingObj) {
	if (getDTauPolygonMainCache()!=0) {
//		clearCurveSpeedupCache();

		std::vector<std::tuple<RgtSeed, RgtSeed, std::size_t>> dataForSignals;

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

//		CUDAImagePaletteHolder* rgtData = m_rgt->image();
//		CUDAImagePaletteHolder* seismicData = m_seismic->image();
//		rgtData->lockPointer();
//		seismicData->lockPointer();
//
//		short* rgtBuf = static_cast<short*>(rgtData->backingPointer());
//		short* seismicBuf = static_cast<short*>(seismicData->backingPointer());

		// apply delta tau for all seeds
		long polarityVal = 0;
		std::map<std::size_t, RgtSeed>::const_iterator it = m_seeds.cbegin();
		while (it!=m_seeds.cend()) {
			RgtSeed oldSeed = it->second;

//			long y, x;
//			if (m_dir==SliceDirection::Inline) {
//				y = it->second.y;
//			} else {
//				y = it->second.z;
//			}

			// move point
//			x = it->second.x;
//			if (getDTauPolygonMainCache()>0) {
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

			RgtSeed newSeed;

			newSeed.x = it->second.x;
			newSeed.seismicValue = it->second.seismicValue; //seismicBuf[x*dimy+y];
			newSeed.rgtValue = it->second.rgtValue + getDTauPolygonMainCache();
			newSeed.y = it->second.y;
			newSeed.z = it->second.z;

			polarityVal += it->second.seismicValue;

			// send signal
			std::tuple<RgtSeed, RgtSeed, std::size_t> myTuple(oldSeed, newSeed, it->first);
			dataForSignals.push_back(myTuple);
			//emit pointMoved(oldSeed, it->second, it->first);
			it++;
		}

//		rgtData->unlockPointer();
//		seismicData->unlockPointer();

//		disconnect(m_data, &MultiSeedHorizon::polarityChanged, this, &MultiSeedSliceRep::updateMainHorizon);
		setPolarity((polarityVal>=0) ? 1 : -1);
//		connect(m_data, &MultiSeedHorizon::polarityChanged, this, &MultiSeedSliceRep::updateMainHorizon);
		// emit signals
//		disconnect(m_data, &MultiSeedHorizon::pointMoved, this, &MultiSeedSliceRep::pointMovedSynchro);

		emit startApplyDTauTransaction(requestingObj);
		for (std::tuple<RgtSeed, RgtSeed, std::size_t>& myTuple : dataForSignals) {
			//emit pointMoved(std::get<0>(myTuple), std::get<1>(myTuple), std::get<2>(myTuple));
			moveSeed(std::get<2>(myTuple), std::get<1>(myTuple));
		}
		m_dtauPolygonMainCache = 0;
		emit dtauPolygonMainCacheChanged(m_dtauPolygonMainCache);
		emit endApplyDTauTransaction(requestingObj);
//		connect(m_data, &MultiSeedHorizon::pointMoved, this, &MultiSeedSliceRep::pointMovedSynchro);
	}
//	m_polygonMainCache.clear();
}

template<typename InputType>
void MultiSeedHorizon::UpdateSeedsPositionWithReadKernel<InputType>::run(MultiSeedHorizon* obj,
		std::vector<std::tuple<RgtSeed, RgtSeed, std::size_t>>& dataForSignals) {
	long dimx = obj->m_rgt->height();
	long dimy = obj->m_rgt->width();
	long dimz = obj->m_rgt->depth();

	std::vector<InputType> vect;
	vect.resize(dimx * obj->m_rgt->dimV());

	// apply delta tau for all seeds
	std::map<std::size_t, RgtSeed>::const_iterator it = obj->m_seeds.cbegin();
	while (it!=obj->m_seeds.cend()) {
		RgtSeed oldSeed = it->second;

		obj->m_rgt->readSubTraceAndSwap(vect.data(), 0, dimx, oldSeed.y, oldSeed.z);
		double dtauReference = oldSeed.rgtValue - vect[oldSeed.x * obj->m_rgt->dimV() + obj->m_channelRgt];
		long ix=oldSeed.x;
		if (dtauReference>0) {
			ix = std::max(static_cast<long>(oldSeed.x), 0l);
			while (ix<dimx && vect[ix* obj->m_rgt->dimV() + obj->m_channelRgt]<vect[oldSeed.x* obj->m_rgt->dimV() + obj->m_channelRgt]+dtauReference) {
				ix++;
			}
			ix = std::min(ix, static_cast<long>(dimx-1));
		} else if(dtauReference<0) {
			ix = std::min(static_cast<long>(oldSeed.x), static_cast<long>(dimx));
			long oldX = ix;
			while (ix>=0 && vect[ix* obj->m_rgt->dimV() + obj->m_channelRgt]>vect[oldSeed.x* obj->m_rgt->dimV() + obj->m_channelRgt]+dtauReference) {
				ix--;
			}
			if (ix<dimx-1 && vect[ix* obj->m_rgt->dimV() + obj->m_channelRgt]<vect[oldSeed.x* obj->m_rgt->dimV() + obj->m_channelRgt]+dtauReference) {
				ix++;
			}
		}
		if (ix<0) {
			ix = 0;
		} else if (ix>=dimx) {
			ix = dimx-1;
		}

		RgtSeed newSeed;

		newSeed.x = ix;
		newSeed.seismicValue = it->second.seismicValue; //seismicBuf[x*dimy+y];
		newSeed.rgtValue = it->second.rgtValue;
		newSeed.y = it->second.y;
		newSeed.z = it->second.z;


		// send signal
		std::tuple<RgtSeed, RgtSeed, std::size_t> myTuple(oldSeed, newSeed, it->first);
		dataForSignals.push_back(myTuple);
		//emit pointMoved(oldSeed, it->second, it->first);
		it++;
	}
}

void MultiSeedHorizon::updateSeedsPositionWithRead(QObject* requestingObj) {

	std::vector<std::tuple<RgtSeed, RgtSeed, std::size_t>> dataForSignals;

	SampleTypeBinder binder(m_rgt->sampleType());
	binder.bind<UpdateSeedsPositionWithReadKernel>(this, dataForSignals);

	emit startApplyDTauTransaction(requestingObj);
	for (std::tuple<RgtSeed, RgtSeed, std::size_t>& myTuple : dataForSignals) {
		moveSeed(std::get<2>(myTuple), std::get<1>(myTuple));
	}
	m_dtauPolygonMainCache = 0;
	emit dtauPolygonMainCacheChanged(m_dtauPolygonMainCache);
	emit endApplyDTauTransaction(requestingObj);
}

template<typename InputType>
void MultiSeedHorizon::UpdateSeedsRgtValueWithDatasetKernel<InputType>::run(std::map<std::size_t, RgtSeed>& seeds,
		Seismic3DDataset* rgt, int channelRgt) {
	long dimx = rgt->height();
	long dimy = rgt->width();
	long dimz = rgt->depth();

	std::vector<InputType> vect;
	vect.resize(dimx * rgt->dimV());

	// apply delta tau for all seeds
	std::map<std::size_t, RgtSeed>::iterator it = seeds.begin();
	while (it!=seeds.end()) {
		RgtSeed& seed = it->second;

		rgt->readSubTraceAndSwap(vect.data(), 0, dimx, seed.y, seed.z);
		seed.rgtValue = vect[seed.x * rgt->dimV() + channelRgt];

		it++;
	}
}

std::map<std::size_t, RgtSeed> MultiSeedHorizon::getSeedsRgtValueWithDataset(const std::map<std::size_t, RgtSeed>& oldSeeds,
		Seismic3DDataset* rgt, int channelRgt) {
	std::map<std::size_t, RgtSeed> newSeeds = oldSeeds;

	SampleTypeBinder binder(rgt->sampleType());
	binder.bind<UpdateSeedsRgtValueWithDatasetKernel>(newSeeds, rgt, channelRgt);

	return newSeeds;
}

template<typename InputType>
void MultiSeedHorizon::UpdateSeedsSeismicValueWithDatasetKernel<InputType>::run(std::map<std::size_t, RgtSeed>& seeds,
		Seismic3DDataset* seismic, int channelSeismic) {
	long dimx = seismic->height();
	long dimy = seismic->width();
	long dimz = seismic->depth();

	std::vector<InputType> vect;
	vect.resize(dimx * seismic->dimV());

	// apply delta tau for all seeds
	std::map<std::size_t, RgtSeed>::iterator it = seeds.begin();
	while (it!=seeds.end()) {
		RgtSeed& seed = it->second;

		seismic->readSubTraceAndSwap(vect.data(), 0, dimx, seed.y, seed.z);
		seed.seismicValue = vect[seed.x * seismic->dimV() + channelSeismic];

		it++;
	}
}

std::map<std::size_t, RgtSeed> MultiSeedHorizon::getSeedsSeismicValueWithDataset(const std::map<std::size_t, RgtSeed>& oldSeeds,
		Seismic3DDataset* seismic, int channelSeismic) {
	std::map<std::size_t, RgtSeed> newSeeds = oldSeeds;

	SampleTypeBinder binder(seismic->sampleType());
	binder.bind<UpdateSeedsSeismicValueWithDatasetKernel>(newSeeds, seismic, channelSeismic);

	return newSeeds;
}

bool MultiSeedHorizon::changeRgtVolume(Seismic3DDataset* newRgt, int newChannelRgt) {
	bool validChange = (newRgt==m_rgt && newChannelRgt<m_rgt->dimV() && newChannelRgt>=0 && newChannelRgt!=m_channelRgt) ||
			(newRgt!=m_rgt && newRgt->cubeSeismicAddon().compare3DGrid((m_rgt->cubeSeismicAddon())) && newRgt->width()==m_rgt->width() &&
					newRgt->height()==m_rgt->height() && newRgt->depth()==m_rgt->depth());
	if (validChange) {
		applyDtauToSeeds();
		updateSeedsPositionWithRead(this);
		std::map<std::size_t, RgtSeed> seeds = getSeedsRgtValueWithDataset(m_seeds, newRgt, newChannelRgt);
		// there may be an issue if reps use new dataset before the seeds are updated
		m_rgt = newRgt;
		m_channelRgt = newChannelRgt;
		emit rgtChanged();

		emit startApplyDTauTransaction(this);
		for (const std::pair<std::size_t, RgtSeed>& seed : seeds) {
			//emit pointMoved(std::get<0>(myTuple), std::get<1>(myTuple), std::get<2>(myTuple));
			moveSeed(seed.first, seed.second);
		}
		emit endApplyDTauTransaction(this);
	}
	return validChange;
}


bool MultiSeedHorizon::changeSeismicVolume(Seismic3DDataset* newSeismic, int newChannelSeismic) {
	bool validChange = (newSeismic==m_seismic && newChannelSeismic<m_seismic->dimV() && newChannelSeismic>=0 && newChannelSeismic!=m_channelSeismic) ||
	(newSeismic->cubeSeismicAddon().compare3DGrid((m_seismic->cubeSeismicAddon())) && newSeismic->width()==m_seismic->width() &&
			newSeismic->height()==m_seismic->height() && newSeismic->depth()==m_seismic->depth());
	if (validChange) {
		applyDtauToSeeds();
		updateSeedsPositionWithRead(this);
		std::map<std::size_t, RgtSeed> seeds = getSeedsSeismicValueWithDataset(m_seeds, newSeismic, newChannelSeismic);
		// there may be an issue if reps use new dataset before the seeds are updated
		m_seismic = newSeismic;
		m_channelSeismic = newChannelSeismic;
		m_seeds = seeds;
		emit seismicChanged();

		emit startApplyDTauTransaction(this);
		for (const std::pair<std::size_t, RgtSeed>& seed : seeds) {
			//emit pointMoved(std::get<0>(myTuple), std::get<1>(myTuple), std::get<2>(myTuple));
			moveSeed(seed.first, seed.second);
		}
		emit endApplyDTauTransaction(this);

		// may be needed to change polarity
	}
	return validChange;
}

void MultiSeedHorizon::constrainBufferUpdated(QString name) {
	if (name.compare(FixedLayerFromDataset::ISOCHRONE)==0) {
		emit constrainIsoChanged();
	}
}

ITreeWidgetItemDecorator* MultiSeedHorizon::getTreeWidgetItemDecorator() {
	if (m_decorator==nullptr) {
		m_decorator = new TextColorTreeWidgetItemDecorator(QColor(Qt::cyan), this);
	}
	return m_decorator;
}
