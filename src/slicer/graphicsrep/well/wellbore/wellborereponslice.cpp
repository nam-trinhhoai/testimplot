#include "wellborereponslice.h"
#include "wellborelayeronslice.h"
#include "wellbore.h"
#include "wellhead.h"
#include "abstractsectionview.h"
#include "geometry2dtoolbox.h"
#include "affine2dtransformation.h"
#include "wellboreproppanelonslice.h"
#include "wellpickreponslice.h"
#include "wellpick.h"
#include "workingsetmanager.h"
#include "../../../widget/WPetrophysics/PlotWithMultipleKeys.h"

#include <cmath>

#include <QMenu>
#include <QAction>

#define WELLBOREREPONSLICE_EPSILON 1.0e-30

WellBoreRepOnSlice::WellBoreRepOnSlice(WellBore *wellBore, SliceDirection dir, AbstractInnerView *parent) :
	AbstractGraphicRep(parent), m_data(wellBore), m_dir(dir) ,m_currentWorldSlice(0){
	m_layer = nullptr;
	m_propPanel = nullptr;
	//setSectionType(SectionType::DEPTH);

	AbstractSectionView* sectionView = dynamic_cast<AbstractSectionView*>(parent);
	m_displayDistance = sectionView->displayDistance();

	connect(m_data, &WellBore::logChanged, this, &WellBoreRepOnSlice::logChanged);
	connect(sectionView, &AbstractSectionView::displayDistanceChanged, this, &WellBoreRepOnSlice::setDisplayDistance);
	connect(m_data, &WellBore::boreUpdated, this, &WellBoreRepOnSlice::reExtractDeviation);

	connect(m_data,&WellBore::deletedMenu,this,&WellBoreRepOnSlice::deleteWellBoreRepOnSlice);// MZR 18082021
}

WellBoreRepOnSlice::~WellBoreRepOnSlice() {
	if (m_layer!=nullptr) {
		delete m_layer;
		disconnect(dynamic_cast<Abstract2DInnerView*>(m_parent), &Abstract2DInnerView::viewAreaChanged,
				m_layer, &WellBoreLayerOnSlice::refreshLog);
	}
	if (m_propPanel!=nullptr) {
		delete m_propPanel;
	}
}

IData* WellBoreRepOnSlice::data() const {
	return m_data;
}

QString WellBoreRepOnSlice::name() const {
	return m_data->name();
}

bool WellBoreRepOnSlice::canBeDisplayed() const {
	bool canBeDisplayed = false;

	// TODO check if wellBore is compatible with inner view 
	canBeDisplayed = true;
	
	return canBeDisplayed;
}

//AbstractGraphicRep
QWidget* WellBoreRepOnSlice::propertyPanel() {
	if (m_propPanel == nullptr) {
		WellBorePropPanelOnSlice* propPanel = new WellBorePropPanelOnSlice(m_data, m_parent);
		m_propPanel = propPanel;
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
		connect(propPanel, &WellBorePropPanelOnSlice::originChanged,
				this, &WellBoreRepOnSlice::originChanged);
		connect(propPanel, &WellBorePropPanelOnSlice::widthChanged,
				this, &WellBoreRepOnSlice::widthChanged);
		connect(propPanel, &WellBorePropPanelOnSlice::logMinChanged,
				this, &WellBoreRepOnSlice::logMinChanged);
		connect(propPanel, &WellBorePropPanelOnSlice::logMaxChanged,
				this, &WellBoreRepOnSlice::logMaxChanged);
		originChanged(propPanel->origin());
		widthChanged(propPanel->width());
		logMinChanged(propPanel->logMin());
		logMaxChanged(propPanel->logMax());
	}
	return m_propPanel;
}

GraphicLayer* WellBoreRepOnSlice::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) {
	if (m_layer==nullptr) {
		m_layer = new WellBoreLayerOnSlice(this, scene, defaultZDepth, parent);
		m_layer->toggleLogDisplay(m_logDisplayParams.size()>0);

		AbstractSectionView* sectionView = dynamic_cast<AbstractSectionView*>(m_parent);
		if (sectionView) {
			connect(sectionView, &AbstractSectionView::signalWellSectionWidth, m_layer, &WellBoreLayerOnSlice::setPenWidth);
			m_layer->setPenWidth(sectionView->getWellSectionWidth());
		}

		connect(m_layer, &WellBoreLayerOnSlice::layerShownChanged, this, &WellBoreRepOnSlice::updatePicks);
		connect(dynamic_cast<Abstract2DInnerView*>(m_parent), &Abstract2DInnerView::viewAreaChanged,
						m_layer, &WellBoreLayerOnSlice::refreshLog);
	}
	return m_layer;
}

bool WellBoreRepOnSlice::setSampleUnit(SampleUnit type) {
	bool unitSetCorrect = true;
	if (type==SampleUnit::TIME && m_data->isWellCompatibleForTime(true)) {
		bool isValid = false;
		Deviations deviations = m_data->deviations();
		isValid = deviations.tvds.size()==deviations.xs.size() && deviations.tvds.size()==deviations.ys.size() && m_data->isTfpDefined();
		AbstractSectionView* sectionView = dynamic_cast<AbstractSectionView*>(view());
		isValid = isValid && sectionView && sectionView->inlineXLineToXY();
		if (isValid) {
			WellHead* wellHead = m_data->wellHead();
			m_trajectory.samples.resize(deviations.tvds.size());
			m_trajectory.traces.resize(deviations.tvds.size());
			m_trajectory.profils.resize(deviations.tvds.size());
			m_trajectory.xs.resize(deviations.tvds.size());
			m_trajectory.ys.resize(deviations.tvds.size());
			m_trajectory.mds.resize(deviations.tvds.size());
			bool isConversionValid;

			std::size_t fillIndex = 0;
			for (std::size_t i=0; i<deviations.tvds.size(); i++) {
				double sample = m_data->getTwtFromMd(deviations.mds[i], &isConversionValid);
				if (isConversionValid) {
					m_trajectory.samples[fillIndex] = sample;
					double imageX, imageY;

					sectionView->inlineXLineToXY()->worldToImage(deviations.xs[i], deviations.ys[i], imageX, imageY);
					m_trajectory.traces[fillIndex] = imageX;
					m_trajectory.profils[fillIndex] = imageY;
					m_trajectory.xs[fillIndex] = deviations.xs[i];
					m_trajectory.ys[fillIndex] = deviations.ys[i];
					m_trajectory.mds[fillIndex] = deviations.mds[i];
					fillIndex++;
				}
			}
			m_trajectory.samples.resize(fillIndex);
			m_trajectory.traces.resize(fillIndex);
			m_trajectory.profils.resize(fillIndex);
			m_trajectory.xs.resize(fillIndex);
			m_trajectory.ys.resize(fillIndex);
			m_trajectory.mds.resize(fillIndex);
			isValid = fillIndex>0;
		}
		if (isValid) {
			m_sectionType = type;
			m_boundingBox = computeBoundingBox();
		} else {
			setSampleUnit(SampleUnit::NONE);
			unitSetCorrect = false;
		}

	} else if (type==SampleUnit::DEPTH) {
		bool isValid = false;
		Deviations deviations = m_data->deviations();
		isValid = deviations.tvds.size()==deviations.xs.size() && deviations.tvds.size()==deviations.ys.size();
		AbstractSectionView* sectionView = dynamic_cast<AbstractSectionView*>(view());
		isValid = isValid && sectionView && sectionView->inlineXLineToXY();
		if (isValid) {
			WellHead* wellHead = m_data->wellHead();
			m_trajectory.samples.resize(deviations.tvds.size());
			m_trajectory.traces.resize(deviations.tvds.size());
			m_trajectory.profils.resize(deviations.tvds.size());
			m_trajectory.xs.resize(deviations.tvds.size());
			m_trajectory.ys.resize(deviations.tvds.size());
			m_trajectory.mds.resize(deviations.tvds.size());
			for (std::size_t i=0; i<deviations.tvds.size(); i++) {
				m_trajectory.samples[i] = deviations.tvds[i];
				double imageX, imageY;

				sectionView->inlineXLineToXY()->worldToImage(deviations.xs[i], deviations.ys[i], imageX, imageY);
				m_trajectory.traces[i] = imageX;
				m_trajectory.profils[i] = imageY;
				m_trajectory.xs[i] = deviations.xs[i];
				m_trajectory.ys[i] = deviations.ys[i];
				m_trajectory.mds[i] = deviations.mds[i];;
			}
		}

		if (isValid) {
			m_sectionType = type;
			m_boundingBox = computeBoundingBox();
		} else {
			setSampleUnit(NONE);
			unitSetCorrect = false;
		}
	} else {
		// clean
		m_sectionType = type;
		m_trajectory.samples.clear();
		m_trajectory.traces.clear();
		m_trajectory.profils.clear();
		m_trajectory.mds.clear();
		m_trajectory.xs.clear();
		m_trajectory.ys.clear();
		m_displayTrajectories.clear();
		m_displayLogTrajectories.clear();
		m_logDisplayParams.clear();
		m_boundingBox = QRectF();

		if (m_layer!=nullptr) {
			m_layer->toggleLogDisplay(false);
		}
	}
	setSliceIJPosition(0); // val is not taken into account
	return unitSetCorrect;
}

QList<SampleUnit> WellBoreRepOnSlice::getAvailableSampleUnits() const {
	QList<SampleUnit> list;
	if (m_data->isTfpDefined()) {
		list.push_back(SampleUnit::TIME);
	}
	list.push_back(SampleUnit::DEPTH);
	return list;
}

QString WellBoreRepOnSlice::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	if (list.contains(sampleUnit)) {
		return "Failure to load supported unit";
	} else{
		return "Unknown unit";
	}
}

QRectF WellBoreRepOnSlice::computeBoundingBox() {
	double sampleMin = std::numeric_limits<double>::max();
	double sampleMax = std::numeric_limits<double>::lowest();
	double axisMin = std::numeric_limits<double>::max();
	double axisMax = std::numeric_limits<double>::lowest();

	for (std::size_t index=0; index<m_trajectory.samples.size(); index++) {
		double& valSample = m_trajectory.samples[index];
		if (valSample<sampleMin) {
			sampleMin = valSample;
		}
		if (valSample>sampleMax) {
			sampleMax = valSample;
		}
		double valAxis;
		if (m_dir==SliceDirection::Inline) {
			valAxis = m_trajectory.traces[index];
		} else {
			valAxis = m_trajectory.profils[index];
		}
		if (valAxis<axisMin) {
			axisMin = valAxis;
		}
		if (valAxis>axisMax) {
			axisMax = valAxis;
		}
	}
	return QRectF(QPointF(axisMin, sampleMin), QPointF(axisMax, sampleMax));
}

void WellBoreRepOnSlice::setSliceIJPosition(int val) {
	m_displayTrajectories.clear();
	if (m_sectionType==SampleUnit::NONE) {
		return;
	}

	AbstractSectionView* sectionView = dynamic_cast<AbstractSectionView*>(view());
	if (sectionView==nullptr || !sectionView->isMapRelationSet()) {
		return;
	}
	m_currentWorldSlice = sectionView->getCurrentSliceWorldPosition();


	double displayDistance = m_displayDistance;

	std::vector<std::pair<double, double>> mdIntervals;
	//std::pair<QPointF, QPointF> segment;
	std::pair<QPointF, QPointF> segmentOnMap;
	if (m_dir==SliceDirection::Inline) {
		//segment = std::pair<QPointF, QPointF>(QPointF(0, m_currentWorldSlice), QPointF(1, m_currentWorldSlice));
		QPointF a(0, m_currentWorldSlice), b(1, m_currentWorldSlice);
		double aMapX, aMapY, bMapX, bMapY;
		sectionView->inlineXLineToXY()->imageToWorld(a.x(), a.y(), aMapX, aMapY);
		sectionView->inlineXLineToXY()->imageToWorld(b.x(), b.y(), bMapX, bMapY);
		segmentOnMap = std::pair<QPointF, QPointF>(QPointF(aMapX, aMapY), QPointF(bMapX, bMapY));
	} else {
		//segment = std::pair<QPointF, QPointF>(QPointF(m_currentWorldSlice, 0), QPointF(m_currentWorldSlice, 1));
		QPointF a(m_currentWorldSlice, 0), b(m_currentWorldSlice, 1);
		double aMapX, aMapY, bMapX, bMapY;
		sectionView->inlineXLineToXY()->imageToWorld(a.x(), a.y(), aMapX, aMapY);
		sectionView->inlineXLineToXY()->imageToWorld(b.x(), b.y(), bMapX, bMapY);
		segmentOnMap = std::pair<QPointF, QPointF>(QPointF(aMapX, aMapY), QPointF(bMapX, bMapY));
	}

	std::pair<double, QPointF> lastDistanceAndProjection;
	bool isAddingVect = false;
	for (std::size_t i=0; i<m_trajectory.samples.size(); i++) {
		bool ok;
		std::pair<double, QPointF> distanceAndProjection = getPointProjectionOnLine(QPointF(m_trajectory.xs[i], m_trajectory.ys[i]), segmentOnMap, &ok);

		if (i>0 && distanceAndProjection.first<displayDistance && lastDistanceAndProjection.first<displayDistance) {
			if (isAddingVect && m_displayTrajectories.size()>0) {
				if (m_dir==SliceDirection::Inline) {
					m_displayTrajectories.at(m_displayTrajectories.size()-1) << QPointF(m_trajectory.traces[i], m_trajectory.samples[i]);
				} else {
					m_displayTrajectories.at(m_displayTrajectories.size()-1) << QPointF(m_trajectory.profils[i], m_trajectory.samples[i]);
				}
				mdIntervals[mdIntervals.size()-1].second = m_trajectory.mds[i];
			} else {
				m_displayTrajectories.push_back(QPolygonF());
				if (m_dir==SliceDirection::Inline) {
					m_displayTrajectories.at(m_displayTrajectories.size()-1) << QPointF(m_trajectory.traces[i-1], m_trajectory.samples[i-1]);
					m_displayTrajectories.at(m_displayTrajectories.size()-1) << QPointF(m_trajectory.traces[i], m_trajectory.samples[i]);
				} else{
					m_displayTrajectories.at(m_displayTrajectories.size()-1) << QPointF(m_trajectory.profils[i-1], m_trajectory.samples[i-1]);
					m_displayTrajectories.at(m_displayTrajectories.size()-1) << QPointF(m_trajectory.profils[i], m_trajectory.samples[i]);
				}
				std::pair<double, double> mdPair(m_trajectory.mds[i-1], m_trajectory.mds[i]);
				mdIntervals.push_back(mdPair);
			}
			isAddingVect = true;
		} else {
			isAddingVect = false;
		}
		lastDistanceAndProjection = distanceAndProjection;
	}

	this->logPreprocessing(mdIntervals);

	if (m_layer) {
		m_layer->refresh();
	}
}

const std::vector<QPolygonF>& WellBoreRepOnSlice::displayTrajectories() const {
	return m_displayTrajectories;
}

QRectF WellBoreRepOnSlice::boundingBox() const {
	return m_boundingBox;
}

void WellBoreRepOnSlice::updatePicks() {
	bool isShown = m_layer!=nullptr && m_layer->isShown();

	const QList<AbstractGraphicRep*>& reps = m_parent->getVisibleReps();
	std::size_t index = 0;
	while (index<reps.size()) {
		WellPickRepOnSlice* pickRep = dynamic_cast<WellPickRepOnSlice*>(reps[index]);
		if (pickRep!=nullptr && dynamic_cast<WellPick*>(pickRep->data())->wellBore()==m_data) {
			pickRep->wellBoreLayerChanged(isShown, this);
		}
		index++;
	}
}

bool WellBoreRepOnSlice::isLayerShown() const {
	return m_layer!=nullptr && m_layer->isShown();
}

void WellBoreRepOnSlice::logChanged() {
	setSliceIJPosition(0);
}

void moy1d_double(const std::vector<WellBoreRepOnSlice::LogGraphicPoint>& tab1,QPolygonF& tab2,int nx,int fx,int opt)
{
	int i,j,ix,ind;
	double som ;
	tab2[0] = QPointF(0,0);
	for (ix=0;ix<=fx;ix++) { tab2[0] = tab2[0] +  tab1[ix].refPoint;}
	for (ix=1;ix<=fx;ix++)
		tab2[ix]=tab2[ix -1]+tab1[ix+fx].refPoint;
	/* Partie centrale */
	for (ix=fx+1;ix<nx-fx;ix++)
		tab2[ix]= tab2[ix -1]+ tab1[ ix + fx].refPoint- tab1[ix-fx-1].refPoint;

	/* Conditions finales */
	for (ix=nx-fx;ix<nx;ix++)
		tab2[ix]=tab2[ix-1]-tab1[ix-fx-1].refPoint;
	/* Renormalisations */
	if(opt != 0) {
		for (ix=0;ix<=fx;ix++) tab2[ix]/=(ix+fx+1);
		for (ix=fx+1;ix<nx-fx;ix++) tab2[ix] /= (2*fx+1);
		for (ix=nx-fx;ix<nx;ix++) tab2[ix]/=(nx-ix+fx);
	}
}


void WellBoreRepOnSlice::logPreprocessing(const std::vector<std::pair<double, double>>& mdIntervals) {
	m_displayLogTrajectories.clear();
	m_logDisplayParams.clear();

	if (!m_data->isLogDefined()) {
		if (m_layer!=nullptr) {
			m_layer->toggleLogDisplay(false);
		}
		return;
	}

	const Logs& log = m_data->currentLog();

	AbstractSectionView* sectionView = dynamic_cast<AbstractSectionView*>(view());
	if (sectionView->inlineXLineToXY()==nullptr) {
		if (m_layer!=nullptr) {
			m_layer->toggleLogDisplay(false);
		}
		return;
	}

	std::vector<std::pair<double, double>> croppedIntervals = fuseIntervals(mdIntervals, log.nonNullIntervals);

	std::pair<QPointF, QPointF> segment;
	if (m_dir==SliceDirection::Inline) {
		segment = std::pair<QPointF, QPointF>(QPointF(0, m_currentWorldSlice), QPointF(1, m_currentWorldSlice));
	} else {
		segment = std::pair<QPointF, QPointF>(QPointF(m_currentWorldSlice, 0), QPointF(m_currentWorldSlice, 1));
	}

	long logIndex = 0;
	for (const std::pair<double, double>& pair : croppedIntervals) {
		if (m_logDisplayParams.size()==0 || m_logDisplayParams[m_logDisplayParams.size()-1].size()!=0) {
			m_logDisplayParams.push_back(std::vector<LogGraphicPoint>());
		}

		// search beginning of interval
		bool indexNotFound = true;
		while (indexNotFound && logIndex<log.keys.size()) {
			bool conversionValid;
			double convertedMd = m_data->getMdFromWellUnit(log.keys[logIndex], log.unit, &conversionValid);
			indexNotFound = !conversionValid || convertedMd<pair.first;
			if (indexNotFound) {
				logIndex++;
			}
		}

		// extract params from intervals
		bool indexIsValid = !indexNotFound;
		while (indexIsValid && logIndex<log.keys.size()) {
			bool conversionValid;
			double convertedMd = m_data->getMdFromWellUnit(log.keys[logIndex], log.unit, &conversionValid);
			indexIsValid = conversionValid && convertedMd<pair.second;
			if (indexIsValid) {
				bool ok;
				LogGraphicPoint param;
				double mdVal = convertedMd;
				double x = m_data->getXFromMd(mdVal, &ok);
				double y;
				if (ok) {
					y = m_data->getYFromMd(mdVal, &ok);
				}
				double depth;
				if (ok && m_sectionType==DEPTH) {
					depth = m_data->getTvdFromMd(mdVal, &ok);
				} else if (ok && m_sectionType==TIME){
					depth = m_data->getTwtFromMd(mdVal, &ok);
				}
				if (ok) {
					param.logValue = log.attributes[logIndex];
					double trace, profil;
					sectionView->inlineXLineToXY()->worldToImage(x, y, trace, profil);
					std::pair<double, QPointF> distanceAndProjection = getPointProjectionOnLine(
							QPointF(trace, profil), segment, &ok);

					if (m_dir==SliceDirection::Inline) {
						param.refPoint = QPointF(distanceAndProjection.second.x(), depth);
					} else {
						param.refPoint = QPointF(distanceAndProjection.second.y(), depth);
					}
				}
				if (ok) { // skip point if not ok but keep same curve
					m_logDisplayParams[m_logDisplayParams.size()-1].push_back(param);
				}
			}
			logIndex++;
		}
		// if only one point clear
		if (m_logDisplayParams[m_logDisplayParams.size()-1].size()<2) {
			m_logDisplayParams[m_logDisplayParams.size()-1].clear();
		} else {
			// filter points
			QPolygonF polyTmp;
			polyTmp.resize(m_logDisplayParams[m_logDisplayParams.size()-1].size());
			moy1d_double(m_logDisplayParams[m_logDisplayParams.size()-1], polyTmp,
					m_logDisplayParams[m_logDisplayParams.size()-1].size(), 11, 1);

			// apply new points and extract normals
			for (std::size_t idx=0; idx<m_logDisplayParams[m_logDisplayParams.size()-1].size(); idx++) {
				m_logDisplayParams[m_logDisplayParams.size()-1][idx].refPoint = polyTmp[idx];

				if (idx>0) {
					QPointF a = m_logDisplayParams[m_logDisplayParams.size()-1][idx-1].refPoint;
					QPointF b = m_logDisplayParams[m_logDisplayParams.size()-1][idx].refPoint;
					QVector2D supportVect(b-a);
					QVector2D normal = getNormal(supportVect, true);
					QVector3D supportVect3d(supportVect);
					QVector3D normal3d(normal);
					QVector3D orientation = QVector3D::crossProduct(supportVect3d, normal3d);
					if (orientation.z()>0) {
						normal = -normal;
					}
					m_logDisplayParams[m_logDisplayParams.size()-1][idx].normal = normal;
					if (idx==1) {
						m_logDisplayParams[m_logDisplayParams.size()-1][0].normal = normal;
					}
				}
			}
		}
		if (m_logDisplayParams[m_logDisplayParams.size()-1].size()==1) {
			m_logDisplayParams[m_logDisplayParams.size()-1].clear();
		}
	}
	if (m_logDisplayParams.size()>0 && m_logDisplayParams[m_logDisplayParams.size()-1].size()==1) {
		m_logDisplayParams.pop_back();
	}
	if (m_layer!=nullptr) {
		m_layer->toggleLogDisplay(m_logDisplayParams.size()>0);
	}
}

std::vector<std::pair<double, double>> WellBoreRepOnSlice::fuseIntervals(const std::vector<std::pair<double, double>>& mdIntervals,
		const std::vector<std::pair<long, long>>& logIndexInterval) {
	std::vector<std::pair<double, double>> croppedIntervals;


	const Logs& log = m_data->currentLog();
	std::size_t indexLogIntervals = 0;
	std::size_t indexMdIntervals = 0;
	while (indexLogIntervals<logIndexInterval.size() && indexMdIntervals<mdIntervals.size()) {
		// turn logInterval (interval of indexes) into an interval of mds
		bool okMin, okMax;
		long intervalMinIndex = logIndexInterval[indexLogIntervals].first;
		long intervalMaxIndex = logIndexInterval[indexLogIntervals].second;
		double minMdLogInterval = m_data->getMdFromWellUnit(log.keys[intervalMinIndex], log.unit, &okMin);
		double maxMdLogInterval = m_data->getMdFromWellUnit(log.keys[intervalMaxIndex], log.unit, &okMax);

		if (okMin && !okMax) {
			// search new min <= max to replace out of bound min
			long index = intervalMaxIndex;
			bool maxNotFound = true;
			bool conversionValid;
			double mdVal;
			while (maxNotFound && index>intervalMinIndex) {
				mdVal = m_data->getMdFromWellUnit(log.keys[index], log.unit, &conversionValid);
				maxNotFound = !conversionValid;
				if (maxNotFound) {
					index--;
				}
			}
			if (!maxNotFound) {
				okMax = true;
				intervalMaxIndex = index;
				maxMdLogInterval = mdVal;
			}
		} else if (!okMin && okMax) {
			// search new max >= min to replace out of bound max
			long index = intervalMinIndex;
			bool minNotFound = true;
			bool conversionValid;
			double mdVal;
			while (minNotFound && index<intervalMaxIndex) {
				mdVal = m_data->getMdFromWellUnit(log.keys[index], log.unit, &conversionValid);
				minNotFound = !conversionValid;
				if (minNotFound) {
					index++;
				}
			}
			if (!minNotFound) {
				okMin = true;
				intervalMinIndex = index;
				minMdLogInterval = mdVal;
			}
		}
		if (okMin && okMax) {
			bool logIntervalEndReached = false;
			while (!logIntervalEndReached && indexMdIntervals<mdIntervals.size()) {
				// case md interval n log interval = 0 and md interval at the left of log interval
				if (minMdLogInterval>=mdIntervals[indexMdIntervals].second) {
					indexMdIntervals++;
				} else if (maxMdLogInterval>=mdIntervals[indexMdIntervals].second) {
					// case md interval n log interval != 0 and md interval not at the left

					double minMdResultInterval;
					double maxMdResultInterval = mdIntervals[indexMdIntervals].second;
					if (minMdLogInterval>mdIntervals[indexMdIntervals].first) {
						minMdResultInterval = minMdLogInterval;
					} else {
						minMdResultInterval = mdIntervals[indexMdIntervals].first;
					}
					croppedIntervals.push_back(std::pair<double, double>(minMdResultInterval, maxMdResultInterval));

					indexMdIntervals++;
				} else if (maxMdLogInterval<=mdIntervals[indexMdIntervals].first) {
					// case md interval n log interval = 0 and md interval at the right of log interval
					logIntervalEndReached = true;
				} else {
					// log interval include in md interval or union !=0 and log interval at the left of md interval

					double minMdResultInterval;
					double maxMdResultInterval = maxMdLogInterval;
					if (minMdLogInterval>=mdIntervals[indexMdIntervals].first) {
						// log interval in md interval
						minMdResultInterval = minMdLogInterval;
					} else {
						minMdResultInterval = mdIntervals[indexMdIntervals].first;
					}
					croppedIntervals.push_back(std::pair<double, double>(minMdResultInterval, maxMdResultInterval));
					logIntervalEndReached = true;
				}
			}
		}

		indexLogIntervals++;
	}

	// do union of touching intervals
	long idx = croppedIntervals.size()-1;
	while (idx>0) {
		std::pair<double, double>& pair1 = croppedIntervals[idx-1];
		std::pair<double, double>& pair2 = croppedIntervals[idx];

		if (std::fabs(pair1.second-pair2.first)<WELLBOREREPONSLICE_EPSILON) {
			pair1.second = pair2.second;
			croppedIntervals.erase(croppedIntervals.begin()+idx);
		}
		idx--;
	}
	return croppedIntervals;
}

const std::vector<std::vector<WellBoreRepOnSlice::LogGraphicPoint>>& WellBoreRepOnSlice::logDisplayParams() const {
	return m_logDisplayParams;
}


void WellBoreRepOnSlice::logMinChanged(double val) {
	if (m_layer) {
		m_layer->setLogMin(val);
	}
}

void WellBoreRepOnSlice::logMaxChanged(double val) {
	if (m_layer) {
		m_layer->setLogMax(val);
	}
}

void WellBoreRepOnSlice::widthChanged(double val) {
	if (m_layer) {
		m_layer->setWidth(val);
	}
}

void WellBoreRepOnSlice::originChanged(double val) {
	if (m_layer) {
		m_layer->setOrigin(val);
	}
}

double WellBoreRepOnSlice::displayDistance() const {
	return m_displayDistance;
}

void WellBoreRepOnSlice::setDisplayDistance(double val) {
	if (m_displayDistance!=val) {
		m_displayDistance = val;
		setSliceIJPosition(0);
	}
}

// MZR 18082021
void WellBoreRepOnSlice::buildContextMenu(QMenu *menu){
	QAction *deleteAction = new QAction(tr("Delete Wells"), this);
	menu->addAction(deleteAction);
	connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteWellBoreRepOnSlice()));

	QAction *viewWellsLogAction = new QAction(tr("View Wells Log"), this);
	menu->addAction(viewWellsLogAction);
	connect(viewWellsLogAction, SIGNAL(triggered()), this, SLOT(viewWellsLog()));
}

void WellBoreRepOnSlice::viewWellsLog() {
	WorkingSetManager *manager = m_data->workingSetManager();
	PlotWithMultipleKeys *w = new PlotWithMultipleKeys(manager);
	w->show();
}

void WellBoreRepOnSlice::deleteWellBoreRepOnSlice(){
	m_parent->hideRep(this);
	emit deletedRep(this);

	disconnect(m_data, nullptr, this, nullptr);
	m_data->deleteRep();

	if(m_layer != nullptr){
		m_layer->hide();
	}

	WorkingSetManager *manager = m_data->workingSetManager();
	manager->deleteWellHead(m_data->wellHead());

	this->deleteLater();
}

AbstractGraphicRep::TypeRep WellBoreRepOnSlice::getTypeGraphicRep() {
	return AbstractGraphicRep::Courbe;
}

void WellBoreRepOnSlice::reExtractDeviation() {
	setSampleUnit(m_sectionType);
	setSliceIJPosition(0);
}
