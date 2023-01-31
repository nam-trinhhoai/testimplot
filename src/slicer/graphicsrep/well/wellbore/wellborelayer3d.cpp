#include "wellborelayer3d.h"

#include "wellborerepon3d.h"
#include "wellbore.h"
#include "wellhead.h"
#include "viewqt3d.h"
#include "qt3dhelpers.h"

#include <QRandomGenerator>

#include <Qt3DCore/QEntity>
#include <Qt3DCore/QTransform>
#include <Qt3DRender/QCamera>
#include <chrono>
#include "polygoninterpolator.h"

using namespace std::chrono;




WellBoreLayer3D::WellBoreLayer3D(WellBoreRepOn3D *rep, QWindow * parent,
		Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera)  :
		Graphic3DLayer(parent, root, camera) {
    m_rep = rep;
    m_selected=false;

    m_isShown=false;

    m_camera = camera;
    m_colorA = QVector3D(0, 0, 0);
    m_colorB = QVector3D(0, 0, 1);

    connect(this,SIGNAL(showNameSignal(const IToolTipProvider*,QString,int,int,QVector3D)),dynamic_cast<ViewQt3D*>(m_rep->view()),SLOT(showTooltipWell(const IToolTipProvider*,QString,int,int,QVector3D)));
    connect(this,SIGNAL(hideNameSignal()),dynamic_cast<ViewQt3D*>(m_rep->view()),SLOT(hideTooltipWell()));

    connect(this,SIGNAL(selectSignal(WellBoreLayer3D*)),dynamic_cast<ViewQt3D*>(m_rep->view()),SLOT(selectWell(WellBoreLayer3D*)));
}

WellBoreLayer3D::~WellBoreLayer3D() {

}


QVector3D WellBoreLayer3D::applyPalette(float ratio) const {
	float r = std::max(0.0f, std::min(1.0f, m_colorA.x() * ratio + (1-ratio) * m_colorB.x()));
	float g = std::max(0.0f, std::min(1.0f, m_colorA.y() * ratio + (1-ratio) * m_colorB.y()));
	float b = std::max(0.0f, std::min(1.0f, m_colorA.z() * ratio + (1-ratio) * m_colorB.z()));
	return QVector3D(r, g, b);
}
/*
QVector3D WellBoreLayer3D::getPosFromMd(double md, bool* ok) const {
	const Logs& log = m_rep->wellBore()->currentLog();

	double x= m_rep->wellBore()->getXFromMd(md, ok);
	double y, depth;
	if (*ok) {
		y = m_rep->wellBore()->getYFromMd(md, ok);
	}
	if (*ok) {
		if (m_rep->sampleUnit()==SampleUnit::TIME) {
			depth = m_rep->wellBore()->getTwtFromMd(md, ok);
		} else {
			depth = m_rep->wellBore()->getTvdFromMd(md, ok);
		}
	}
	QVector3D pos = QVector3D(x, depth, y);
	return pos;
}*/

Vector3dD WellBoreLayer3D::getPosFromWellUnitD(double unitIndex, WellUnit wellUnit, bool* ok) const {
	const Logs& log = m_rep->wellBore()->currentLog();

	double x= m_rep->wellBore()->getXFromWellUnit(unitIndex, wellUnit, ok);
	double y, depth;
	if (*ok) {
		y = m_rep->wellBore()->getYFromWellUnit(unitIndex, wellUnit, ok);
	}
	if (*ok) {
		depth = m_rep->wellBore()->getDepthFromWellUnit(unitIndex, wellUnit, m_rep->sampleUnit(), ok);
	}
	Vector3dD pos = Vector3dD(x, depth, y);
	return pos;
}

void WellBoreLayer3D::updateLog()
{
	if( m_lineEntities.count() >0 &&  m_lineEntities[0].entitylog != nullptr)  m_lineEntities[0].entitylog->deleteLater();

	showLog();

}

void WellBoreLayer3D::showLog()
{
	float minWidth = m_minimalWidth;
		float maxWidth = m_maximalWidth;

		double logMin = m_logMin;
		double logMax = m_logMax;
	const Logs& log = m_rep->wellBore()->currentLog();
	bool isLogDefined = m_rep->wellBore()->isLogDefined() && log.nonNullIntervals.size()>0;

		QMatrix4x4 transform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();

	QVector<QVector<QVector3D>> listePosVecLog;
	QVector<QVector<float>> listeWidthVecLog;


	QVector<QVector3D> posVecLog;
	QVector<float> widthVecLog;
	//boucle log
	if(isLogDefined)
	{
		for(int i=0;i< log.nonNullIntervals.size();i++)
		{
			int start = log.nonNullIntervals[i].first;
			int end = log.nonNullIntervals[i].second;

			//QVector<QVector3D> posVecLog;
			//QVector<float> widthVecLog;
			for(int index=start;index<=end;index+= m_incrLog)
			{
				double logKey = log.keys[index];
				bool ok;

				//QVector3D logPos = getPosFromMd(md, &ok);//(x, depth, y);
				Vector3dD logPosd = getPosFromWellUnitD(logKey, log.unit, &ok);
				logPosd = logPosd.multiply(transform);
				QVector3D logPos = logPosd.convert();

				float ratio = (log.attributes[index] - logMin) / (logMax - logMin);
				ratio = std::max(std::min(1.0f, ratio), 0.0f);
				float width = ratio * (maxWidth - minWidth) + minWidth;

				if(ok)
				{
					posVecLog.push_back(logPos);
					widthVecLog.push_back(width);
				}
			}

		//	containerExt.entitylog =Qt3DHelpers::drawLog(posVecLog,widthVecLog,Qt::green,m_root,1);

			//listePosVecLog.push_back(posVecLog);
			//listeWidthVecLog.push_back(widthVecLog);

		}

	}

	 if(isLogDefined && m_lineEntities.size() > 0)
	{
		 m_lineEntities[0].entitylog =Qt3DHelpers::drawLog(posVecLog,widthVecLog,m_colorLog,m_root,m_thicknessLog);
	}
	else
	{
		if(m_lineEntities.count()>0) m_lineEntities[0].entitylog =nullptr;
	}


}

void WellBoreLayer3D::show() {
	if (m_rep->sampleUnit()!=SampleUnit::TIME && m_rep->sampleUnit()!=SampleUnit::DEPTH) {
		return;
	}

	QMatrix4x4 transform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();

	m_isShown = true;

	const Deviations& deviations = wellBore()->deviations();

	std::pair<Deviations, std::vector<std::size_t>>  res = polygonInterpolator(deviations,m_distanceSimplification);

	long i=0;

	QVector<QVector3D> posVec;
	QVector<float> widthVec;
	QVector<QVector3D> colorVec;


	float defaultWidth = m_defaultWidth;
	QVector3D defaultColor(m_defaultColor.red() / 255.0,
							m_defaultColor.green() / 255.0,
							m_defaultColor.blue() / 255.0);


	float minWidth = m_minimalWidth;
	float maxWidth = m_maximalWidth;

	double logMin = m_logMin;
	double logMax = m_logMax;


	const Logs& log = m_rep->wellBore()->currentLog();
	bool isLogDefined = m_rep->wellBore()->isLogDefined() && log.nonNullIntervals.size()>0;


	long nextLogIndex = 0;
	if (isLogDefined) {
		nextLogIndex = log.nonNullIntervals[0].first;
	}
	long currentLogInterval = 0;
	double lastMd = std::numeric_limits<double>::lowest();

	const Deviations& deviationsSimple =res.first;


	//boucle well

	for (long i=0; i<deviationsSimple.xs.size(); i++) {
		QVector3D start;

				bool isValid = true;
		if (m_rep->sampleUnit()==SampleUnit::TIME)
		{
			double twt1 = m_rep->wellBore()->getTwtFromMd(deviationsSimple.mds[i], &isValid);

			if (isValid) {
				start = QVector3D(deviationsSimple.xs[i], twt1, deviationsSimple.ys[i]);
			}
		} else{
			start = QVector3D(deviationsSimple.xs[i], deviationsSimple.tvds[i], deviationsSimple.ys[i]);
		}
		if(isValid)
		{
			start = transform * start;
			posVec.push_back(start);
			widthVec.push_back(defaultWidth);
			colorVec.push_back(defaultColor);
		}

	}

	/*QVector<QVector<QVector3D>> listePosVecLog;
	QVector<QVector<float>> listeWidthVecLog;


	QVector<QVector3D> posVecLog;
	QVector<float> widthVecLog;
	//boucle log
	if(isLogDefined)
	{
		for(int i=0;i< log.nonNullIntervals.size();i++)
		{
			int start = log.nonNullIntervals[i].first;
			int end = log.nonNullIntervals[i].second;

			//QVector<QVector3D> posVecLog;
			//QVector<float> widthVecLog;
			for(int index=start;index<=end;index+= m_incrLog)
			{
				double md = log.keys[index];
				bool ok;

				//QVector3D logPos = getPosFromMd(md, &ok);//(x, depth, y);
				Vector3dD logPosd = getPosFromMdD(md, &ok);
				logPosd = logPosd.multiply(transform);
				QVector3D logPos = logPosd.convert();

				float ratio = (log.attributes[index] - logMin) / (logMax - logMin);
				ratio = std::max(std::min(1.0f, ratio), 0.0f);
				float width = ratio * (maxWidth - minWidth) + minWidth;

				if(ok)
				{
					posVecLog.push_back(logPos);
					widthVecLog.push_back(width);
				}
			}

		//	containerExt.entitylog =Qt3DHelpers::drawLog(posVecLog,widthVecLog,Qt::green,m_root,1);

			//listePosVecLog.push_back(posVecLog);
			//listeWidthVecLog.push_back(widthVecLog);

		}

	}*/




	//for (long i=0; i<deviations.xs.size()-1; i++) {
/*	for (long i=0; i<deviationsSimple.xs.size()-1; i++) {
		QVector3D start, end;

		bool isValid = true;

		if (m_rep->sampleUnit()==SampleUnit::TIME) {
			double twt1 = m_rep->wellBore()->getTwtFromMd(deviationsSimple.mds[i], &isValid);
			double twt2;
			if (isValid) {
				twt2 = m_rep->wellBore()->getTwtFromMd(deviationsSimple.mds[i+1], &isValid);
			}
			if (isValid) {
				start = QVector3D(deviationsSimple.xs[i], twt1, deviationsSimple.ys[i]);
				end = QVector3D(deviationsSimple.xs[i+1], twt2, deviationsSimple.ys[i+1]);
			}
		} else{
			start = QVector3D(deviationsSimple.xs[i], deviationsSimple.tvds[i], deviationsSimple.ys[i]);
			end = QVector3D(deviationsSimple.xs[i+1], deviationsSimple.tvds[i+1], deviationsSimple.ys[i+1]);
		}
		if (isValid) {
			start = transform * start;
			end = transform * end;

			bool isStartInLogInterval = isLogDefined && deviationsSimple.mds[i]>=log.keys[log.nonNullIntervals[currentLogInterval].first] &&
					deviationsSimple.mds[i]<=log.keys[log.nonNullIntervals[currentLogInterval].second];
			// add as many point as needed for logs

			if (!isStartInLogInterval)
			{

				posVec.push_back(start);
				widthVec.push_back(defaultWidth);
				colorVec.push_back(defaultColor);



				lastMd = deviations.mds[i];
			}

			bool isLogIntervalActive = isStartInLogInterval;
			QVector3D lastValidPos;
			bool isLastValidPosSet = false;

			while(currentLogInterval<log.nonNullIntervals.size() && nextLogIndex<log.keys.size() && (log.keys[log.nonNullIntervals[currentLogInterval].second] < deviations.mds[i+1] ||
					log.keys[nextLogIndex]<deviationsSimple.mds[i+1])) {
				double md = log.keys[nextLogIndex];
				bool ok;

				//QVector3D logPos = getPosFromMd(md, &ok);//(x, depth, y);
				Vector3dD logPosd = getPosFromMdD(md, &ok);
				logPosd = logPosd.multiply(transform);
				QVector3D logPos = logPosd.convert();


				//logPos = transform * logPos;

				if (ok) {
					lastValidPos = logPos;
					isLastValidPosSet = true;

					// get md step
					if (!isLogIntervalActive) {
						double stepMd;
						double stepBefore = md - lastMd;
						double nextIndex = nextLogIndex + 1;
						if (nextIndex>log.nonNullIntervals[currentLogInterval].second) {
							stepMd = stepBefore / 100;
						} else {
							double nextMd = log.keys[nextIndex];
							stepMd = std::min(nextMd - md, stepBefore / 2);
						}
						double newMd = md - stepMd;
						bool newOk;
						Vector3dD newLogPosd = getPosFromMdD(newMd, &newOk);
						newLogPosd = newLogPosd.multiply(transform);
						QVector3D newLogPos = newLogPosd.convert();
					//	QVector3D newLogPos = getPosFromMd(newMd, &newOk);
						//newLogPos = transform * newLogPos;

						posVec.push_back(newLogPos);
						widthVec.push_back(defaultWidth);
						colorVec.push_back(defaultColor);

						isLogIntervalActive = true;
					}
					posVec.push_back(logPos);

					float ratio = (log.attributes[nextLogIndex] - logMin) / (logMax - logMin);
					ratio = std::max(std::min(1.0f, ratio), 0.0f);
					float width = ratio * (maxWidth - minWidth) + minWidth;
					widthVec.push_back(width);
					colorVec.push_back(applyPalette(ratio));

					//myPositionsVec.push_back(logPos);
					//myWithVec.push_back(width);
				}
				nextLogIndex= nextLogIndex+ m_incrLog;
				if (nextLogIndex>log.nonNullIntervals[currentLogInterval].second) {
					// add point at the end to close interval
					if (isLastValidPosSet) {
						double stepBefore = md - lastMd;
						double nextMd = deviationsSimple.mds[i+1];
						if (currentLogInterval+1<log.nonNullIntervals.size()) {
							nextMd = std::min(nextMd, log.keys[log.nonNullIntervals[currentLogInterval+1].first]);
						}
						double stepNext = nextMd - md;
						double stepMd = std::min(stepBefore, stepNext/2);
						double newMd = md + stepMd;
						bool newOk;
						Vector3dD newLogPosd = getPosFromMdD(newMd, &newOk);
						//QVector3D newLogPos = getPosFromMd(newMd, &newOk);
						if(newOk)
						{
							newLogPosd = newLogPosd.multiply(transform);
							QVector3D newLogPos = newLogPosd.convert();
							//newLogPos = transform * newLogPos;

							posVec.push_back(newLogPos);
							widthVec.push_back(defaultWidth);
							colorVec.push_back(defaultColor);
						}
						else
						{
							qDebug()<<" error newLogPos ";
						}
					}
					isLastValidPosSet = false;
					isLogIntervalActive = false;

					currentLogInterval++;
					if (currentLogInterval<log.nonNullIntervals.size()) {
						nextLogIndex = log.nonNullIntervals[currentLogInterval].first;
					}
				}
				lastMd = md;
			}
		}
    }

	if ((currentLogInterval>=log.nonNullIntervals.size() || nextLogIndex==log.nonNullIntervals[currentLogInterval].first) &&
			deviationsSimple.mds.size()>0) {
		// add end
		bool isValid = true;
		double depth;
		long index = deviationsSimple.mds.size()-1;

		if (m_rep->sampleUnit()==SampleUnit::TIME) {
			depth = m_rep->wellBore()->getTwtFromMd(deviationsSimple.mds[index], &isValid);
		} else {
			depth = deviationsSimple.tvds[index];
		}

		QVector3D pos(deviationsSimple.xs[index], depth, deviationsSimple.ys[index]);
		if (isValid) {
			pos = transform * pos;
			posVec.push_back(pos);
			widthVec.push_back(defaultWidth);
			colorVec.push_back(defaultColor);
		}
	}
*/
//	steady_clock::time_point middle = steady_clock::now();
	QString nameall = generateToolTipInfo();


	//QVector<QVector3D> myPositionsVec;
    Container containerExt;
    containerExt.entity = Qt3DHelpers::drawExtruders(posVec, colorVec, widthVec, m_root, m_camera,nameall,m_modeWireframe,m_showNormals);

//	steady_clock::time_point nearEnd = steady_clock::now();
    containerExt.transform = new Qt3DCore::QTransform();
    containerExt.entity->addComponent(containerExt.transform);
    containerExt.mat = new Qt3DExtras::QPhongMaterial();
    //containerExt.mat->setAlpha(1.0f);
    containerExt.mat->setDiffuse(m_defaultColor);
    QColor colamb(m_defaultColor.red()*0.25,m_defaultColor.green()*0.25,m_defaultColor.blue()*0.25);
    containerExt.mat->setAmbient(colamb);
    containerExt.entity->addComponent(containerExt.mat);

  /*  if(isLogDefined)
    {
    	containerExt.entitylog =Qt3DHelpers::drawLog(posVecLog,widthVecLog,m_colorLog,m_root,m_thicknessLog);
	}
	else
	{
		containerExt.entitylog =nullptr;
	}*/
 //   containerExt.mat = containerExt.entity->componentsOfType<QMaterial>()



    //qDebug()<<" nb points log : "<<myPositionsVec.size();

 //   containerExt.entitylog = Qt3DHelpers::drawLines(myPositionsVec,Qt::green,m_root,1);

    m_lineEntities.push_back(containerExt);

    showLog();

    Qt3DRender::QObjectPicker *sPicker = new Qt3DRender::QObjectPicker(containerExt.entity);
    Qt3DRender::QPickingSettings * sPickingSettings = new Qt3DRender::QPickingSettings(sPicker);
   sPickingSettings->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);
   sPickingSettings->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick);
   sPickingSettings->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontAndBackFace);
   sPickingSettings->setEnabled(true);
   sPicker->setEnabled(true);
   sPicker->setHoverEnabled(true);
   containerExt.entity->addComponent(sPicker);



       connect(sPicker, &Qt3DRender::QObjectPicker::pressed, [&](Qt3DRender::QPickEvent* e) {
		 // qDebug() << "======= Qt3DRender::QObjectPicker::clicked =======";
		  // activate on left mouse button
		  if(e->button() == Qt3DRender::QPickEvent::Buttons::RightButton)
		  {
			 m_selected = ! m_selected;
			// qDebug()<<" pressed "<<m_lineEntities.count();

			// QVector3D dirCam = m_camera->position()- e->worldIntersection();
			// dirCam = dirCam.normalized();

			  if(m_selected) this->selectWell(e->entity()->objectName(),e->position().x(),e->position().y(),e->worldIntersection());
			 else deselectWell();

		  }
		  if(e->button() == Qt3DRender::QPickEvent::Buttons::LeftButton)
		  {
			  auto p = dynamic_cast<Qt3DRender::QPickTriangleEvent*>(e);
			if(p) {
				QVector3D pos = p->worldIntersection();
				QPropertyAnimation* animation = new QPropertyAnimation(m_camera,"viewCenter");
				animation->setDuration(2000);
				animation->setStartValue(m_camera->viewCenter());
				animation->setEndValue(pos);
				animation->start();

				float coefZoom = 0.7f;
				QVector3D dirDest = (pos - m_camera->position()) * coefZoom;
				QVector3D  newpos = m_camera->position() + dirDest;

				QPropertyAnimation* animation2 = new QPropertyAnimation(m_camera,"position");
				animation2->setDuration(2000);
				animation2->setStartValue(m_camera->position());
				animation2->setEndValue(newpos);
				animation2->start();
			}

		 }
	  });


	//steady_clock::time_point end = steady_clock::now();

//	duration<double> time_span1 = duration_cast<duration<double>>(begin - middle);
//	duration<double> time_span2 = duration_cast<duration<double>>(middle - nearEnd);
//	duration<double> time_span3 = duration_cast<duration<double>>(nearEnd - end);

//	duration<double> time_spanTot = duration_cast<duration<double>>(end - begin);

//	qDebug() << "It took me " << time_spanTot.count() << " seconds.";
//	qDebug() << "Details : " << time_span1.count() << " " << time_span2.count() << " " << time_span3.count();
    emit layerShownChanged(true);
}

/*

void WellBoreLayer3D::show() {
	if (m_rep->sampleUnit()!=SampleUnit::TIME && m_rep->sampleUnit()!=SampleUnit::DEPTH) {
		return;
	}


	//steady_clock::time_point begin = steady_clock::now();

	//QColor color(Qt::green);
	QMatrix4x4 transform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();

	m_isShown = true;
	//bool isValid = true;
	const Deviations& deviations = wellBore()->deviations();

	std::pair<Deviations, std::vector<std::size_t>>  res = polygonInterpolator(deviations,m_distanceSimplification);

	long i=0;	//QVector<QVector3D> startPos;
	//QVector<QVector3D> endPos;
	QVector<QVector3D> posVec;
	QVector<float> widthVec;
	QVector<QVector3D> colorVec;

	float defaultWidth = m_defaultWidth;
	QVector3D defaultColor(m_defaultColor.red() / 255.0,
							m_defaultColor.green() / 255.0,
							m_defaultColor.blue() / 255.0);

	float minWidth = m_minimalWidth;
	float maxWidth = m_maximalWidth;

	double logMin = m_logMin;
	double logMax = m_logMax;


	const Logs& log = m_rep->wellBore()->currentLog();
	bool isLogDefined = m_rep->wellBore()->isLogDefined() && log.unit==WellUnit::MD && log.nonNullIntervals.size()>0;


	long nextLogIndex = 0;
	if (isLogDefined) {
		nextLogIndex = log.nonNullIntervals[0].first;
	}
	long currentLogInterval = 0;
	double lastMd = std::numeric_limits<double>::lowest();

	const Deviations& deviationsSimple =res.first;
	//for (long i=0; i<deviations.xs.size()-1; i++) {
	for (long i=0; i<deviationsSimple.xs.size()-1; i++) {
		QVector3D start, end;

		bool isValid = true;

		if (m_rep->sampleUnit()==SampleUnit::TIME) {
			double twt1 = m_rep->wellBore()->getTwtFromMd(deviationsSimple.mds[i], &isValid);
			double twt2;
			if (isValid) {
				twt2 = m_rep->wellBore()->getTwtFromMd(deviationsSimple.mds[i+1], &isValid);
			}
			if (isValid) {
				start = QVector3D(deviationsSimple.xs[i], twt1, deviationsSimple.ys[i]);
				end = QVector3D(deviationsSimple.xs[i+1], twt2, deviationsSimple.ys[i+1]);
			}
		} else{
			start = QVector3D(deviationsSimple.xs[i], deviationsSimple.tvds[i], deviationsSimple.ys[i]);
			end = QVector3D(deviationsSimple.xs[i+1], deviationsSimple.tvds[i+1], deviationsSimple.ys[i+1]);
		}
		if (isValid) {
			start = transform * start;
			end = transform * end;

			bool isStartInLogInterval = isLogDefined && deviationsSimple.mds[i]>=log.keys[log.nonNullIntervals[currentLogInterval].first] &&
					deviationsSimple.mds[i]<=log.keys[log.nonNullIntervals[currentLogInterval].second];
			// add as many point as needed for logs

			if (!isStartInLogInterval) {
				posVec.push_back(start);
				widthVec.push_back(defaultWidth);
				colorVec.push_back(defaultColor);
				lastMd = deviations.mds[i];
			}

			bool isLogIntervalActive = isStartInLogInterval;
			QVector3D lastValidPos;
			bool isLastValidPosSet = false;

			while(currentLogInterval<log.nonNullIntervals.size() && nextLogIndex<log.keys.size() && (log.keys[log.nonNullIntervals[currentLogInterval].second] < deviations.mds[i+1] ||
					log.keys[nextLogIndex]<deviationsSimple.mds[i+1])) {
				double md = log.keys[nextLogIndex];
				bool ok;

				//QVector3D logPos = getPosFromMd(md, &ok);//(x, depth, y);
				Vector3dD logPosd = getPosFromMdD(md, &ok);
				logPosd = logPosd.multiply(transform);
				QVector3D logPos = logPosd.convert();


				//logPos = transform * logPos;

				if (ok) {
					lastValidPos = logPos;
					isLastValidPosSet = true;

					// get md step
					if (!isLogIntervalActive) {
						double stepMd;
						double stepBefore = md - lastMd;
						double nextIndex = nextLogIndex + 1;
						if (nextIndex>log.nonNullIntervals[currentLogInterval].second) {
							stepMd = stepBefore / 100;
						} else {
							double nextMd = log.keys[nextIndex];
							stepMd = std::min(nextMd - md, stepBefore / 2);
						}
						double newMd = md - stepMd;
						bool newOk;
						Vector3dD newLogPosd = getPosFromMdD(newMd, &newOk);
						newLogPosd = newLogPosd.multiply(transform);
						QVector3D newLogPos = newLogPosd.convert();
					//	QVector3D newLogPos = getPosFromMd(newMd, &newOk);
						//newLogPos = transform * newLogPos;

						posVec.push_back(newLogPos);
						widthVec.push_back(defaultWidth);
						colorVec.push_back(defaultColor);

						isLogIntervalActive = true;
					}
					posVec.push_back(logPos);

					float ratio = (log.attributes[nextLogIndex] - logMin) / (logMax - logMin);
					ratio = std::max(std::min(1.0f, ratio), 0.0f);
					float width = ratio * (maxWidth - minWidth) + minWidth;
					widthVec.push_back(width);
					colorVec.push_back(applyPalette(ratio));

					//myPositionsVec.push_back(logPos);
					//myWithVec.push_back(width);
				}
				nextLogIndex= nextLogIndex+ m_incrLog;
				if (nextLogIndex>log.nonNullIntervals[currentLogInterval].second) {
					// add point at the end to close interval
					if (isLastValidPosSet) {
						double stepBefore = md - lastMd;
						double nextMd = deviationsSimple.mds[i+1];
						if (currentLogInterval+1<log.nonNullIntervals.size()) {
							nextMd = std::min(nextMd, log.keys[log.nonNullIntervals[currentLogInterval+1].first]);
						}
						double stepNext = nextMd - md;
						double stepMd = std::min(stepBefore, stepNext/2);
						double newMd = md + stepMd;
						bool newOk;
						Vector3dD newLogPosd = getPosFromMdD(newMd, &newOk);
						//QVector3D newLogPos = getPosFromMd(newMd, &newOk);
						if(newOk)
						{
							newLogPosd = newLogPosd.multiply(transform);
							QVector3D newLogPos = newLogPosd.convert();
							//newLogPos = transform * newLogPos;

							posVec.push_back(newLogPos);
							widthVec.push_back(defaultWidth);
							colorVec.push_back(defaultColor);
						}
						else
						{
							qDebug()<<" error newLogPos ";
						}
					}
					isLastValidPosSet = false;
					isLogIntervalActive = false;

					currentLogInterval++;
					if (currentLogInterval<log.nonNullIntervals.size()) {
						nextLogIndex = log.nonNullIntervals[currentLogInterval].first;
					}
				}
				lastMd = md;
			}
		}
    }

	if ((currentLogInterval>=log.nonNullIntervals.size() || nextLogIndex==log.nonNullIntervals[currentLogInterval].first) &&
			deviationsSimple.mds.size()>0) {
		// add end
		bool isValid = true;
		double depth;
		long index = deviationsSimple.mds.size()-1;

		if (m_rep->sampleUnit()==SampleUnit::TIME) {
			depth = m_rep->wellBore()->getTwtFromMd(deviationsSimple.mds[index], &isValid);
		} else {
			depth = deviationsSimple.tvds[index];
		}

		QVector3D pos(deviationsSimple.xs[index], depth, deviationsSimple.ys[index]);
		if (isValid) {
			pos = transform * pos;
			posVec.push_back(pos);
			widthVec.push_back(defaultWidth);
			colorVec.push_back(defaultColor);
		}
	}

//	steady_clock::time_point middle = steady_clock::now();
	QString nameWell = m_rep->name();

	QString status= wellBore()->getStatus();
	QString uwi= wellBore()->getUwi();
	QString domain = wellBore()->getDomain();
	QString elev = wellBore()->getElev();
	QString datum = wellBore()->getDatum();
	QString velocity = wellBore()->getVelocity();
	QString ihs = wellBore()->getIhs();
	QString date = wellBore()->wellHead()->getDate();

	QString nameall = nameWell+"|"+status+"|"+date+"|"+uwi+"|"+domain+"|"+elev+"|"+datum+"|"+velocity+"|"+ihs;


	QVector<QVector3D> myPositionsVec;
    Container containerExt;
    containerExt.entity = Qt3DHelpers::drawExtruders(posVec, colorVec, widthVec, m_root, m_camera,nameall,m_modeWireframe,m_showNormals,myPositionsVec);

//	steady_clock::time_point nearEnd = steady_clock::now();
    containerExt.transform = new Qt3DCore::QTransform();
    containerExt.entity->addComponent(containerExt.transform);
    containerExt.mat = new Qt3DExtras::QPhongMaterial();
    //containerExt.mat->setAlpha(1.0f);
    containerExt.mat->setDiffuse(m_defaultColor);
    QColor colamb(m_defaultColor.red()*0.25,m_defaultColor.green()*0.25,m_defaultColor.blue()*0.25);
    containerExt.mat->setAmbient(colamb);
    containerExt.entity->addComponent(containerExt.mat);

 //   containerExt.mat = containerExt.entity->componentsOfType<QMaterial>()



    //qDebug()<<" nb points log : "<<myPositionsVec.size();

    containerExt.entitylog = Qt3DHelpers::drawLines(myPositionsVec,Qt::green,m_root,1);

    m_lineEntities.push_back(containerExt);

    Qt3DRender::QObjectPicker *sPicker = new Qt3DRender::QObjectPicker(containerExt.entity);
    Qt3DRender::QPickingSettings * sPickingSettings = new Qt3DRender::QPickingSettings(sPicker);
   sPickingSettings->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);
   sPickingSettings->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick);
   sPickingSettings->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontAndBackFace);
   sPickingSettings->setEnabled(true);
   sPicker->setEnabled(true);
   sPicker->setHoverEnabled(true);
   containerExt.entity->addComponent(sPicker);



       connect(sPicker, &Qt3DRender::QObjectPicker::pressed, [&](Qt3DRender::QPickEvent* e) {
		 // qDebug() << "======= Qt3DRender::QObjectPicker::clicked =======";
		  // activate on left mouse button
		  if(e->button() == Qt3DRender::QPickEvent::Buttons::RightButton)
		  {
			 m_selected = ! m_selected;
			// qDebug()<<" pressed "<<m_lineEntities.count();

			// QVector3D dirCam = m_camera->position()- e->worldIntersection();
			// dirCam = dirCam.normalized();

			  if(m_selected) this->selectWell(e->entity()->objectName(),e->position().x(),e->position().y(),e->worldIntersection());
			 else deselectWell();

		  }
		  if(e->button() == Qt3DRender::QPickEvent::Buttons::LeftButton)
		  {
			  auto p = dynamic_cast<Qt3DRender::QPickTriangleEvent*>(e);
			if(p) {
				QVector3D pos = p->worldIntersection();
				QPropertyAnimation* animation = new QPropertyAnimation(m_camera,"viewCenter");
				animation->setDuration(2000);
				animation->setStartValue(m_camera->viewCenter());
				animation->setEndValue(pos);
				animation->start();

				float coefZoom = 0.7f;
				QVector3D dirDest = (pos - m_camera->position()) * coefZoom;
				QVector3D  newpos = m_camera->position() + dirDest;

				QPropertyAnimation* animation2 = new QPropertyAnimation(m_camera,"position");
				animation2->setDuration(2000);
				animation2->setStartValue(m_camera->position());
				animation2->setEndValue(newpos);
				animation2->start();
			}

		 }
	  });


	//steady_clock::time_point end = steady_clock::now();

//	duration<double> time_span1 = duration_cast<duration<double>>(begin - middle);
//	duration<double> time_span2 = duration_cast<duration<double>>(middle - nearEnd);
//	duration<double> time_span3 = duration_cast<duration<double>>(nearEnd - end);

//	duration<double> time_spanTot = duration_cast<duration<double>>(end - begin);

//	qDebug() << "It took me " << time_spanTot.count() << " seconds.";
//	qDebug() << "Details : " << time_span1.count() << " " << time_span2.count() << " " << time_span3.count();
    emit layerShownChanged(true);
}
*/

void WellBoreLayer3D::selectWell(QString name,int posX,int posY, QVector3D posGlobal)
{
	if( m_lineEntities.count() >0)
	 {
		if(m_selected)
		{
			emit selectSignal(this);
			m_lineEntities[0].mat->setDiffuse(m_selectedColor);
		}
		else m_lineEntities[0].mat->setDiffuse(m_defaultColor);
	 }
	emit showNameSignal(this,name,posX,posY,posGlobal);
}

void WellBoreLayer3D::deselectWell()
{
	if( m_lineEntities.count() >0) m_lineEntities[0].mat->setDiffuse(m_defaultColor);
	//wellBore()->setDisplayPreference(false);
	emit hideNameSignal();
}

void WellBoreLayer3D::deselectLastWell()
{
	m_selected =false;
	if( m_lineEntities.count() >0) m_lineEntities[0].mat->setDiffuse(m_defaultColor);
	//wellBore()->setDisplayPreference(false);

}

void WellBoreLayer3D::setDistanceSimplification( double value)
{
	m_distanceSimplification  = value;
}
void WellBoreLayer3D::setIncrementLogs(int value)
{
	m_incrLog = value;
	updateLog();
}

void WellBoreLayer3D::setWireframe(bool value)
{
	m_modeWireframe = value;

}

void WellBoreLayer3D::setShowNormals(bool value)
{
	m_showNormals= value;
}

void WellBoreLayer3D::setThicknessLog(int value)
{
	if(m_thicknessLog != value)
	{
		m_thicknessLog = value;
		updateLog();
	}
}

void WellBoreLayer3D::setColorLog(QColor value)
{

	if(m_colorLog != value)
	{
		m_colorLog = value;
		updateLog();
	}
}

void WellBoreLayer3D::setColorWell(QColor value)
{

	if(m_defaultColor != value)
	{
		m_defaultColor = value;
		refresh();

	}
}



void WellBoreLayer3D::setColorSelectedWell(QColor value)
{

	if(m_selectedColor != value)
	{
		m_selectedColor = value;
	}
}

void WellBoreLayer3D::setDiameterWell(int value)
{
	if(m_defaultWidth != value)
	{
		m_defaultWidth = value;
		refresh();
	}
}

void WellBoreLayer3D::hideLastWell()
{
	wellBore()->setAllDisplayPreference(false);
}

void WellBoreLayer3D::hide() {
	m_isShown = false;
	for (Container container : m_lineEntities) {
		container.entity->setParent((Qt3DCore::QEntity*) nullptr);
		container.entity->deleteLater();
		if( container.entitylog != nullptr) container.entitylog->deleteLater();
	}
	m_lineEntities.clear();
	emit layerShownChanged(false);
}

QRect3D WellBoreLayer3D::boundingRect() const {
	if (m_rep->sampleUnit()!=SampleUnit::TIME && m_rep->sampleUnit()!=SampleUnit::DEPTH) {
		return QRect3D(0, 0, 0, 0, 0, 0);
	}

	WellBore* wellBore = dynamic_cast<WellBore*>(m_rep->data());

	const Deviations& deviations = wellBore->deviations();

	// fill list
	double xmin = std::numeric_limits<double>::max();
	double xmax = std::numeric_limits<double>::lowest();
	double ymin = std::numeric_limits<double>::max();
	double ymax = std::numeric_limits<double>::lowest();
	double zmin = std::numeric_limits<double>::max();
	double zmax = std::numeric_limits<double>::lowest();

	QMatrix4x4 transform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();

	bool isValid = true;
	bool atLeastOnValid = false;

	for (long i=0; i<deviations.xs.size(); i++) { // xline
		// apply transform
		double iWorld, jWorld, kWorld;
		iWorld = deviations.xs[i];
		kWorld = deviations.ys[i];

		if (m_rep->sampleUnit()==SampleUnit::TIME) {
			jWorld = m_rep->wellBore()->getTwtFromMd(deviations.mds[i], &isValid);
		} else {
			jWorld = deviations.tvds[i];
		}

		if (isValid) {
			QVector3D oriPt(iWorld, jWorld, kWorld);
			QVector3D newPoint = transform*oriPt;

			// get min max
			if (xmin>newPoint.x()) {
				xmin = newPoint.x();
			}
			if (xmax<newPoint.x()) {
				xmax = newPoint.x();
			}
			if (ymin>newPoint.y()) {
				ymin = newPoint.y();
			}
			if (ymax<newPoint.y()) {
				ymax = newPoint.y();
			}
			if (zmin>newPoint.z()) {
				zmin = newPoint.z();
			}
			if (zmax<newPoint.z()) {
				zmax = newPoint.z();
			}
			atLeastOnValid = true;
		}
	}

	QRect3D worldBox;
	if (atLeastOnValid) {
		worldBox = QRect3D(xmin, ymin, zmin, xmax-xmin, ymax-ymin, zmax-zmin);
	} else {
		worldBox = QRect3D(0, 0, 0, 0, 0, 0);
	}

	return worldBox;
}

void WellBoreLayer3D::refresh() {
	if(m_isShown)
	{
		hide();
		show();
	}
}


void WellBoreLayer3D::zScale(float val) {
	for (Container container : m_lineEntities) {
		container.transform->setScale3D(QVector3D(1, val, 1));
	}
}

WellBore * WellBoreLayer3D::wellBore() const {
	return dynamic_cast<WellBore*>(m_rep->data());
}

bool WellBoreLayer3D::isShown() const {
	return m_isShown;
}

void WellBoreLayer3D::setDefaultWidth(long defaultWidth) {
	m_defaultWidth = defaultWidth;
}

void WellBoreLayer3D::setMinimalWidth(long minimalWidth) {
	m_minimalWidth = minimalWidth;
}

void WellBoreLayer3D::setMaximalWidth(long maximalWidth) {
	m_maximalWidth = maximalWidth;
}

void WellBoreLayer3D::setLogMin(double logMin) {
	m_logMin = logMin;
}

void WellBoreLayer3D::setLogMax(double logMax) {
	m_logMax = logMax;
}

void WellBoreLayer3D::setDefaultColor(QColor defaultColor) {
	m_defaultColor = defaultColor;
}

QString WellBoreLayer3D::generateToolTipInfo() const {
	QString nameWell = m_rep->name();

	QString status= wellBore()->getStatus();
	QString uwi= wellBore()->getUwi();
	QString domain = wellBore()->getDomain();
	QString elev = wellBore()->getElev();
	QString datum;
	if (dynamic_cast<ViewQt3D*>(m_rep->parent())) {
		datum = wellBore()->getConvertedDatum(dynamic_cast<ViewQt3D*>(m_rep->parent())->depthLengthUnit());
	} else {
		datum = wellBore()->getDatum();
	}
	QString velocity = wellBore()->getVelocity();
	QString ihs = wellBore()->getIhs();
	QString date = wellBore()->wellHead()->getDate();

	QString nameall = nameWell+"|"+status+"|"+date+"|"+uwi+"|"+domain+"|"+elev+"|"+datum+"|"+velocity+"|"+ihs;
	return nameall;
}
