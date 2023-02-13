#include "wellborereponrandom.h"
#include "wellborelayeronrandom.h"
#include "wellbore.h"
#include "wellhead.h"
#include "randomlineview.h"
#include "geometry2dtoolbox.h"
#include "affine2dtransformation.h"
#include "wellpickreponrandom.h"
#include "wellpick.h"
#include "wellboreproppanelonslice.h"
#include "polygoninterpolator.h"
#include "../../../widget/WPetrophysics/PlotWithMultipleKeys.h"

#include "workingsetmanager.h"

#include <QMenu>
#include <QAction>

#include <cmath>

#define WELLBOREREPONRANDOM_EPSILON 1.0e-30

WellBoreRepOnRandom::WellBoreRepOnRandom(WellBore *wellBore, AbstractInnerView *parent) :
	AbstractGraphicRep(parent), m_data(wellBore) {
	m_layer = nullptr;
	m_propPanel = nullptr;
	//setSectionType(SectionType::DEPTH);

	RandomLineView* random = dynamic_cast<RandomLineView*>(parent);
	m_displayDistance = random->displayDistance();

	connect(m_data, &WellBore::logChanged, this, &WellBoreRepOnRandom::logChanged);
	connect(m_data, &WellBore::boreUpdated, this, &WellBoreRepOnRandom::reExtractDeviation);
	connect(random, &RandomLineView::displayDistanceChanged, this,
			&WellBoreRepOnRandom::setDisplayDistance);

	connect(m_data,&WellBore::deletedMenu,this,&WellBoreRepOnRandom::deleteWellBoreRepOnRandom); // MZR 18082021
}

WellBoreRepOnRandom::~WellBoreRepOnRandom() {
	if (m_layer!=nullptr) {
		delete m_layer;
		disconnect(dynamic_cast<Abstract2DInnerView*>(m_parent), &Abstract2DInnerView::viewAreaChanged,
				m_layer, &WellBoreLayerOnRandom::refreshLog);
	}
	if (m_propPanel!=nullptr) {
		delete m_propPanel;
	}
}

IData* WellBoreRepOnRandom::data() const {
	return m_data;
}

QString WellBoreRepOnRandom::name() const {
	return m_data->name();
}

bool WellBoreRepOnRandom::canBeDisplayed() const {
	bool canBeDisplayed = false;

	// TODO check if wellBore is compatible with inner view
	canBeDisplayed = true;

	return canBeDisplayed;
}

//AbstractGraphicRep
QWidget* WellBoreRepOnRandom::propertyPanel() {
	if (m_propPanel == nullptr) {
		WellBorePropPanelOnSlice* propPanel = new WellBorePropPanelOnSlice(m_data, m_parent);
		m_propPanel = propPanel;
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
		connect(propPanel, &WellBorePropPanelOnSlice::originChanged,
				this, &WellBoreRepOnRandom::originChanged);
		connect(propPanel, &WellBorePropPanelOnSlice::widthChanged,
				this, &WellBoreRepOnRandom::widthChanged);
		connect(propPanel, &WellBorePropPanelOnSlice::logMinChanged,
				this, &WellBoreRepOnRandom::logMinChanged);
		connect(propPanel, &WellBorePropPanelOnSlice::logMaxChanged,
				this, &WellBoreRepOnRandom::logMaxChanged);
		originChanged(propPanel->origin());
		widthChanged(propPanel->width());
		logMinChanged(propPanel->logMin());
		logMaxChanged(propPanel->logMax());
	}
	return m_propPanel;
}

GraphicLayer* WellBoreRepOnRandom::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) {
	if (m_layer==nullptr) {
		m_layer = new WellBoreLayerOnRandom(this, scene, defaultZDepth, parent);
		m_layer->toggleLogDisplay(m_logDisplayParams.size()>0);

		RandomLineView* sectionView = dynamic_cast<RandomLineView*>(m_parent);
		if (sectionView) {
			connect(sectionView, &RandomLineView::signalWellSectionWidth, m_layer, &WellBoreLayerOnRandom::setPenWidth);
			m_layer->setPenWidth(sectionView->getWellSectionWidth());
		}

		connect(m_layer, &WellBoreLayerOnRandom::layerShownChanged, this, &WellBoreRepOnRandom::updatePicks);
		connect(dynamic_cast<Abstract2DInnerView*>(m_parent), &Abstract2DInnerView::viewAreaChanged,
				m_layer, &WellBoreLayerOnRandom::refreshLog);
	}
	return m_layer;
}

// do not forget to clean up if return is false
bool WellBoreRepOnRandom::buildTrajectoriesOnRandom(SampleUnit type) {
	Deviations deviations = m_data->deviations();
	bool isValid = deviations.tvds.size()==deviations.xs.size() && deviations.tvds.size()==deviations.ys.size();
	RandomLineView* randomView = dynamic_cast<RandomLineView*>(view());
	if (isValid) {
		WellHead* wellHead = m_data->wellHead();
		QPolygonF randomPolyLine = randomView->polyLine();

//		std::size_t randomPanelIdx=0;
//		std::vector<std::vector<std::pair<double, double>>> cumulMdIntervals;
//		std::vector<std::vector<double>> cumulMdDistance;
//		for (std::size_t randomPanelIdx=0; randomPanelIdx<randomPolyLine.size()-1; randomPanelIdx++) {
//			std::pair<QPointF, QPointF> randomSegment = std::pair<QPointF, QPointF>(randomPolyLine[randomPanelIdx], randomPolyLine[randomPanelIdx+1]);
//			bool ok;
//			bool isAddingVect = false;
//			std::pair<double, QPointF> lastDistanceAndProjection;
//			bool lastConversionValid = false;
//			double lastSample;
//			std::vector<std::pair<double, double>> mdIntervals;
//			std::vector<double> minSegmentDist;
//			for (std::size_t i=0; i<deviations.tvds.size(); i++) {
//				double sample;
//				bool conversionValid;
//				if (type==SampleUnit::DEPTH) {
//					sample = deviations.tvds[i];
//					conversionValid = true;
//				} else if (type==SampleUnit::TIME) {
//					sample = m_data->getTwtFromMd(deviations.mds[i], &conversionValid);
//				} else{
//					conversionValid = false;
//				}
//				std::pair<double, QPointF> distanceAndProjection = getPointProjectionOnSegment(QPointF(deviations.xs[i], deviations.ys[i]), randomSegment, &ok);
//				if (i!=0 && ok && conversionValid && lastDistanceAndProjection.first<m_displayDistance && lastConversionValid) {
//					if (isAddingVect && m_displayTrajectories.size()>0 && conversionValid) {
//						QPointF worldPos(distanceAndProjection.second);
//						std::tuple<long, double, bool> projection = randomView->getDiscreatePolyLineIndexFromWorldPos(worldPos);
//						if (!std::get<2>(projection)) {
//							qDebug() << "Point in random panel invalid but deviation valid ? How come ?";
//						}
//						m_displayTrajectories.at(m_displayTrajectories.size()-1) << QPointF(std::get<0>(projection), sample);
//						//mdIntervals[mdIntervals.size()-1].second = deviations.mds[i];
//						std::pair<double, double> mdPair(deviations.mds[i-1], deviations.mds[i]);
//						mdIntervals.push_back(mdPair);
//					} else {
//						m_displayTrajectories.push_back(QPolygonF());
//
//						QPointF worldPos1(lastDistanceAndProjection.second);
//						std::tuple<long, double, bool> projection1 = randomView->getDiscreatePolyLineIndexFromWorldPos(worldPos1);
//						if (!std::get<2>(projection1)) {
//							qDebug() << "Point in random panel invalid but deviation valid ? How come ?";
//						}
//						QPointF worldPos2(distanceAndProjection.second);
//						std::tuple<long, double, bool> projection2 = randomView->getDiscreatePolyLineIndexFromWorldPos(worldPos2);
//						if (!std::get<2>(projection2)) {
//							qDebug() << "Point in random panel invalid but deviation valid ? How come ?";
//						}
//						m_displayTrajectories.at(m_displayTrajectories.size()-1) << QPointF(std::get<0>(projection1), lastSample);
//						m_displayTrajectories.at(m_displayTrajectories.size()-1) << QPointF(std::get<0>(projection2), sample);
//
//						std::pair<double, double> mdPair(deviations.mds[i-1], deviations.mds[i]);
//						mdIntervals.push_back(mdPair);
//					}
//					isAddingVect = true;
//				} else {
//					isAddingVect = false;
//				}
//				lastDistanceAndProjection = distanceAndProjection;
//				lastConversionValid = conversionValid;
//				lastSample = sample;
//			}
//			// add new intervals and keep lowest distance segment if well segment already there
//
//		}
		std::pair<Deviations, std::vector<std::size_t>> dvs = polygonInterpolator(deviations, 10.0);
		std::size_t dvsIdx = 0;
		long paramsIdx = -1;
		for (std::size_t i=0; i<deviations.tvds.size()-1; i++) {
			double minDistanceA = std::numeric_limits<double>::max();
			double minDistanceB = std::numeric_limits<double>::max();
			std::size_t indexMin;
			QPointF startPoint;
			QPointF endPoint;


			for (std::size_t randomPanelIdx=0; randomPanelIdx<randomPolyLine.size()-1; randomPanelIdx++) {
				std::pair<QPointF, QPointF> randomSegment = std::pair<QPointF, QPointF>(randomPolyLine[randomPanelIdx], randomPolyLine[randomPanelIdx+1]);
				bool conversionValid, conversionValid2, conversionValid3, conversionValid4;
				std::pair<double, QPointF> distanceAndProjection = getPointProjectionOnSegment(QPointF(deviations.xs[i], deviations.ys[i]), randomSegment, &conversionValid);
				double sample;
				if (type==SampleUnit::DEPTH) {
					sample = deviations.tvds[i];
					conversionValid3 = true;
				} else if (type==SampleUnit::TIME) {
					sample = m_data->getTwtFromMd(deviations.mds[i], &conversionValid3);
				} else{
					conversionValid3 = false;
				}
				std::pair<double, QPointF> distanceAndProjection2 = getPointProjectionOnSegment(QPointF(deviations.xs[i+1], deviations.ys[i+1]), randomSegment, &conversionValid2);
				double sample2;
				if (type==SampleUnit::DEPTH) {
					sample2 = deviations.tvds[i+1];
					conversionValid4 = true;
				} else if (type==SampleUnit::TIME) {
					sample2 = m_data->getTwtFromMd(deviations.mds[i+1], &conversionValid4);
				} else{
					conversionValid4 = false;
				}

				bool distanceCheck = distanceAndProjection.first<minDistanceA && distanceAndProjection2.first<minDistanceB;

				if (conversionValid && conversionValid2 && conversionValid3 && conversionValid4 &&  distanceCheck) {
					QPointF worldPos1(distanceAndProjection.second);
					std::tuple<long, double, bool> projection1 = randomView->getDiscreatePolyLineIndexFromWorldPos(worldPos1);
					if (!std::get<2>(projection1)) {
						qDebug() << "Point in random panel invalid but deviation valid ? How come ?";
					}
					QPointF worldPos2(distanceAndProjection2.second);
					std::tuple<long, double, bool> projection2 = randomView->getDiscreatePolyLineIndexFromWorldPos(worldPos2);
					if (!std::get<2>(projection2)) {
						qDebug() << "Point in random panel invalid but deviation valid ? How come ?";
					}
					startPoint = QPointF(std::get<0>(projection1), sample);
					endPoint =  QPointF(std::get<0>(projection2), sample2);
					indexMin = randomPanelIdx;
					minDistanceA = distanceAndProjection.first;
					minDistanceB = distanceAndProjection2.first;
				}
			}

			if (minDistanceA<m_displayDistance && minDistanceB<m_displayDistance) {
				QPolygonF segmentPoly;
				segmentPoly << startPoint << endPoint;
				m_displayTrajectories.push_back(segmentPoly);

				std::vector<std::pair<double, double>> mdIntervals;
				std::pair<double, double> pair(deviations.mds[i], deviations.mds[i+1]);
				mdIntervals.push_back(pair);

				std::pair<QPointF, QPointF> randomSegment = std::pair<QPointF, QPointF>(randomPolyLine[indexMin], randomPolyLine[indexMin+1]);
				logPreprocessing(mdIntervals, randomSegment);
			}
			if (dvs.second.size()>dvsIdx+1 && dvs.second.at(dvsIdx+1)==i) {
				long n = m_logDisplayParams.size()-1 - paramsIdx;
				if (n>0) {
					QPointF start = m_logDisplayParams[paramsIdx+1][0].refPoint;
					QPointF end = m_logDisplayParams[paramsIdx+n][m_logDisplayParams[paramsIdx+n].size()-1].refPoint;
					QVector2D supportVect(end-start);
					QVector2D normal = getNormal(supportVect, true);
					QVector3D supportVect3d(supportVect);
					QVector3D normal3d(normal);
					QVector3D orientation = QVector3D::crossProduct(supportVect3d, normal3d);
					if (orientation.z()>0) {
						normal = -normal;
					}

					for (std::size_t itPoly=paramsIdx+1; itPoly<m_logDisplayParams.size(); itPoly++) {
						for (std::size_t itPt=0; itPt<m_logDisplayParams[itPoly].size(); itPt++) {
							m_logDisplayParams[itPoly][itPt].normal = normal;
						}
					}

					paramsIdx = m_logDisplayParams.size()-1;
				}
				dvsIdx++;
			}
		}
		if (dvs.second.size()>dvsIdx+1 && dvs.second.at(dvsIdx+1)==deviations.tvds.size()-1) {
			long n = m_logDisplayParams.size()-1 - paramsIdx;
			if (n>0) {
				QPointF start = m_logDisplayParams[paramsIdx+1][0].refPoint;
				QPointF end = m_logDisplayParams[paramsIdx+n][m_logDisplayParams[paramsIdx+n].size()-1].refPoint;
				QVector2D supportVect(end-start);
				QVector2D normal = getNormal(supportVect, true);
				QVector3D supportVect3d(supportVect);
				QVector3D normal3d(normal);
				QVector3D orientation = QVector3D::crossProduct(supportVect3d, normal3d);
				if (orientation.z()>0) {
					normal = -normal;
				}

				for (std::size_t itPoly=paramsIdx+1; itPoly<m_logDisplayParams.size(); itPoly++) {
					for (std::size_t itPt=0; itPt<m_logDisplayParams[itPoly].size(); itPt++) {
						m_logDisplayParams[itPoly][itPt].normal = normal;
					}
				}

				paramsIdx = m_logDisplayParams.size()-1;
			}
			dvsIdx++;
		}

//		for (std::size_t randomPanelIdx=0; randomPanelIdx<randomPolyLine.size()-1; randomPanelIdx++) {
//			std::pair<QPointF, QPointF> randomSegment = std::pair<QPointF, QPointF>(randomPolyLine[randomPanelIdx], randomPolyLine[randomPanelIdx+1]);
//			const std::vector<std::pair<double, double>>& mdIntervals = cumulMdIntervals[randomPanelIdx];
//			logPreprocessing(mdIntervals, randomSegment);
//		}

		if (m_layer!=nullptr) {
			m_layer->toggleLogDisplay(m_logDisplayParams.size()>0);
		}
	}
	return isValid;
}

bool WellBoreRepOnRandom::setSampleUnit(SampleUnit type) {
	bool isCorrect = true;
	m_displayTrajectories.clear();
	m_displayLogTrajectories.clear();
	m_logDisplayParams.clear();
	if ((type==SampleUnit::TIME && m_data->isWellCompatibleForTime(true)) || type==SampleUnit::DEPTH) {
		bool isValid = buildTrajectoriesOnRandom(type);
		if (isValid) {
			m_sectionType = type;
			m_boundingBox = computeBoundingBox();
		} else {
			setSampleUnit(SampleUnit::NONE);
			isCorrect = false;
		}
	} else {
		// clean
		m_sectionType = type;
		m_displayTrajectories.clear();
		m_displayLogTrajectories.clear();
		m_logDisplayParams.clear();
		m_boundingBox = QRectF();
	}
	return isCorrect;
}

QList<SampleUnit> WellBoreRepOnRandom::getAvailableSampleUnits() const {
	QList<SampleUnit> list;
	if (m_data->isTfpDefined()) {
		list.push_back(SampleUnit::TIME);
	}
	list.push_back(SampleUnit::DEPTH);
	return list;
}

QString WellBoreRepOnRandom::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	if (list.contains(sampleUnit)) {
		return "Failure to load supported unit";
	} else{
		return "Unknown unit";
	}
}

QRectF WellBoreRepOnRandom::computeBoundingBox() {
	QRectF out;
	for (std::size_t i=0; i<m_displayTrajectories.size(); i++) {
		QRectF rect = m_displayTrajectories[i].boundingRect();
		if (i==0) {
			out = rect;
		} else {
			out = out.united(rect);
		}
	}

	return out;
}


const std::vector<QPolygonF>& WellBoreRepOnRandom::displayTrajectories() const {
	return m_displayTrajectories;
}

QRectF WellBoreRepOnRandom::boundingBox() const {
	return m_boundingBox;
}

void WellBoreRepOnRandom::updatePicks() {
	bool isShown = m_layer!=nullptr && m_layer->isShown();

	const QList<AbstractGraphicRep*>& reps = m_parent->getVisibleReps();
	std::size_t index = 0;
	while (index<reps.size()) {
		WellPickRepOnRandom* pickRep = dynamic_cast<WellPickRepOnRandom*>(reps[index]);
		if (pickRep!=nullptr && dynamic_cast<WellPick*>(pickRep->data())->wellBore()==m_data) {
			pickRep->wellBoreLayerChanged(isShown, this);
		}
		index++;
	}
}

void WellBoreRepOnRandom::logChanged() {
	setSampleUnit(m_sectionType);
}

bool WellBoreRepOnRandom::isLayerShown() const {
	return m_layer!=nullptr && m_layer->isShown();
}

void WellBoreRepOnRandom::logMinChanged(double val) {
	if (m_layer) {
		m_layer->setLogMin(val);
	}
}

void WellBoreRepOnRandom::logMaxChanged(double val) {
	if (m_layer) {
		m_layer->setLogMax(val);
	}
}

void WellBoreRepOnRandom::widthChanged(double val) {
	if (m_layer) {
		m_layer->setWidth(val);
	}
}

void WellBoreRepOnRandom::originChanged(double val) {
	if (m_layer) {
		m_layer->setOrigin(val);
	}
}

double WellBoreRepOnRandom::displayDistance() const {
	return m_displayDistance;
}

void WellBoreRepOnRandom::setDisplayDistance(double val) {
	if (m_displayDistance!=val) {
		m_displayDistance = val;
		setSampleUnit(m_sectionType);
	}
}

const std::vector<std::vector<WellBoreRepOnRandom::LogGraphicPoint>>& WellBoreRepOnRandom::logDisplayParams() const {
	return m_logDisplayParams;
}

void random_moy1d_double(const std::vector<WellBoreRepOnRandom::LogGraphicPoint>& tab1,QPolygonF& tab2,int nx,int fx,int opt)
{
	int i,j,ix,ind;
	double som ;
	tab2[0] = QPointF(0,0);
	if (fx>nx/2) {
		fx = nx/2;
	}
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


void WellBoreRepOnRandom::logPreprocessing(const std::vector<std::pair<double, double>>& mdIntervals,
		std::pair<QPointF, QPointF> randomSegment) {
	if (!m_data->isLogDefined()) {
		if (m_layer!=nullptr) {
			m_layer->toggleLogDisplay(false);
		}
		return;
	}

	const Logs& log = m_data->currentLog();

	RandomLineView* randomView = dynamic_cast<RandomLineView*>(view());

	std::vector<std::pair<double, double>> croppedIntervals = fuseIntervals(mdIntervals, log.nonNullIntervals);

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
					std::tuple<long, double, bool> projection = randomView->getDiscreatePolyLineIndexFromWorldPos(QPointF(x, y));
	//				double trace, profil;
	//				sectionView->inlineXLineToXY()->worldToImage(x, y, trace, profil);
	//				std::pair<double, QPointF> distanceAndProjection = getPointProjectionOnLine(
	//						QPointF(trace, profil), randomSegment, &ok);
	//
	//				if (m_dir==SliceDirection::Inline) {
	//					param.refPoint = QPointF(distanceAndProjection.second.x(), depth);
	//				} else {
	//					param.refPoint = QPointF(distanceAndProjection.second.y(), depth);
	//				}
					param.refPoint = QPointF(std::get<0>(projection), depth);
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
			random_moy1d_double(m_logDisplayParams[m_logDisplayParams.size()-1], polyTmp,
					m_logDisplayParams[m_logDisplayParams.size()-1].size(), 11, 1);

			// apply new points and extract normals
			for (std::size_t idx=0; idx<m_logDisplayParams[m_logDisplayParams.size()-1].size(); idx++) {
				m_logDisplayParams[m_logDisplayParams.size()-1][idx].refPoint = polyTmp[idx];

//				if (idx>0) {
//					QPointF a = m_logDisplayParams[m_logDisplayParams.size()-1][idx-1].refPoint;
//					QPointF b = m_logDisplayParams[m_logDisplayParams.size()-1][idx].refPoint;
//					QVector2D supportVect(b-a);
//					QVector2D normal = getNormal(supportVect, true);
//					QVector3D supportVect3d(supportVect);
//					QVector3D normal3d(normal);
//					QVector3D orientation = QVector3D::crossProduct(supportVect3d, normal3d);
//					if (orientation.z()>0) {
//						normal = -normal;
//					}
//					// apply modification to normal highest component win, if egual check last
//					if (std::fabs(normal.x()-normal.y()<WELLBOREREPONRANDOM_EPSILON)) {
//						if (idx>1) {
//							normal = m_logDisplayParams[m_logDisplayParams.size()-1][idx-1].normal;
//						} else {
//							normal = QVector2D(1, 0);
//						}
//					} else if (normal.x()>normal.y()) {
//						normal = QVector2D(1, 0);
//					} else {
//						normal = QVector2D(0, 1);
//					}
//					m_logDisplayParams[m_logDisplayParams.size()-1][idx].normal = normal;
//					if (idx==1) {
//						m_logDisplayParams[m_logDisplayParams.size()-1][0].normal = normal;
//					}
//				}
			}
		}
		if (m_logDisplayParams[m_logDisplayParams.size()-1].size()==1) {
			m_logDisplayParams[m_logDisplayParams.size()-1].clear();
		}/* else if (m_logDisplayParams[m_logDisplayParams.size()-1].size()>1) {
			std::size_t N = m_logDisplayParams[m_logDisplayParams.size()-1].size();
			QPointF a = m_logDisplayParams[m_logDisplayParams.size()-1][0].refPoint;
			QPointF b = m_logDisplayParams[m_logDisplayParams.size()-1][N-1].refPoint;
			QVector2D supportVect(b-a);
			QVector2D normal = getNormal(supportVect, true);
			QVector3D supportVect3d(supportVect);
			QVector3D normal3d(normal);
			QVector3D orientation = QVector3D::crossProduct(supportVect3d, normal3d);
			if (orientation.z()>0) {
				normal = -normal;
			}
			for (std::size_t idxNorm=0; idxNorm<N; idxNorm++) {
				m_logDisplayParams[m_logDisplayParams.size()-1][idxNorm].normal = normal;
			}
		}*/
	}
	if (m_logDisplayParams.size()>0 && m_logDisplayParams[m_logDisplayParams.size()-1].size()<=1) {
		m_logDisplayParams.pop_back();
	}
}

std::vector<std::pair<double, double>> WellBoreRepOnRandom::fuseIntervals(const std::vector<std::pair<double, double>>& mdIntervals,
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

		if (std::fabs(pair1.second-pair2.first)<WELLBOREREPONRANDOM_EPSILON) {
			pair1.second = pair2.second;
			croppedIntervals.erase(croppedIntervals.begin()+idx);
		}
		idx--;
	}
	return croppedIntervals;
}

// MZR 19082021
void WellBoreRepOnRandom::buildContextMenu(QMenu *menu){
	QAction *deleteAction = new QAction(tr("Delete Wells 4"), this);
	menu->addAction(deleteAction);
	connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteWellBoreRepOnRandom()));

	QAction* viewWellsLogAction = new QAction(tr("View Wells Log"), this);
	menu->addAction(viewWellsLogAction);
	connect(viewWellsLogAction, SIGNAL(triggered()), this, SLOT(viewWellsLogRepOnRandom()));
}
void WellBoreRepOnRandom::viewWellsLogRepOnRandom() {
	WorkingSetManager* manager = m_data->workingSetManager();
	PlotWithMultipleKeys* w = new PlotWithMultipleKeys(manager);
	w->show();

}
void WellBoreRepOnRandom::deleteWellBoreRepOnRandom(){
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

AbstractGraphicRep::TypeRep WellBoreRepOnRandom::getTypeGraphicRep() {
	return AbstractGraphicRep::Courbe;
}

void WellBoreRepOnRandom::deleteLayer(){
    if (m_layer!=nullptr) {
        delete m_layer;
        m_layer = nullptr;
    }

    if (m_propPanel!=nullptr) {
        delete m_propPanel;
        m_propPanel = nullptr;
    }
}

void WellBoreRepOnRandom::reExtractDeviation() {
	setSampleUnit(m_sectionType);
	if (m_layer) {
		m_layer->refresh();
	}
}
