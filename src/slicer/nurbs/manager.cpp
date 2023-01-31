#include "manager.h"
#include "pointslinesgeometry.h"
#include "meshgeometry.h"
#include "workingsetmanager.h"
#include "cpuimagepaletteholder.h"
//#include "layerstablemodel.h"
//#include "implicits/triangulate.h"
//#include "help.h"

#include <QMessageBox>
#include <QTimer>
#include <QJsonObject>
#include <QJsonDocument>
#include <QJsonArray>
#include <QTime>
#include <QFile>

#include "pointslinesgeometry.h"
#include "curvelistener.h"

#include "nurbsentity.h"
#include "randomlineview.h"

//Manager* Manager::thesingleton = nullptr;

Manager::Manager(WorkingSetManager* workingset, QString name, int precision,QVector3D posCam, Qt3DCore::QEntity* root,IsoSurfaceBuffer surface,int timeLayer, QObject* parent)
{
   // thesingleton = this;
	m_nameId = name;
	m_timeLayer = timeLayer;
	m_nurbsData = new NurbsDataset(workingset,this,m_nameId,this);

	m_bufferIso = surface;
	m_modeEditable=true;




    m_nurbs = new NurbsEntity(precision,posCam,root);
    m_curveInstantiator = new CurveInstantiator(root);


    setinbetweenxsectionalpha(1.0f);
    setNuminbetweens(40);
  //  setinbetweenxsectionalpha(0.5);
    setLinearInterpolateInbetweens(true);


    m_cloneNurbs = new NurbsClone();


    // picker
	Qt3DRender::QObjectPicker* spicker = new Qt3DRender::QObjectPicker();
	Qt3DRender::QPickingSettings *pickingSettings = new Qt3DRender::QPickingSettings(spicker);
	pickingSettings->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);
	pickingSettings->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick);
	pickingSettings->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontAndBackFace);
	pickingSettings->setEnabled(true);
	spicker->setEnabled(true);
	spicker->setDragEnabled(true);
	m_nurbs->addComponent(spicker);//m_ghostEntity

	connect(spicker, &Qt3DRender::QObjectPicker::pressed, [&](Qt3DRender::QPickEvent* e) {
		m_actifRayon= true;
		});
		connect(spicker, &Qt3DRender::QObjectPicker::moved, [&](Qt3DRender::QPickEvent* e) {
		m_actifRayon= false;
		});

	connect(spicker, &Qt3DRender::QObjectPicker::clicked, [&](Qt3DRender::QPickEvent* e) {

	if(m_actifRayon== true)// && e->button() == Qt3DRender::QPickEvent::Buttons::LeftButton)
	{
		int bouton = e->button();
		auto p = dynamic_cast<Qt3DRender::QPickTriangleEvent*>(e);
		if(p) {
			QVector3D pos = p->worldIntersection();
			emit sendAnimationCam(bouton,pos);
		}
	}

	});

}


Manager::Manager(WorkingSetManager* workingset, QString name,QVector3D posCam, Qt3DCore::QEntity* root, QObject* parent)
{
   // thesingleton = this;
	m_nameId = name;
	m_nurbsData = new NurbsDataset(workingset,this,m_nameId,this);



	m_modeEditable = false;

	//m_bufferIso = nul;

	//qDebug()<< "m_nameId : "<<m_nameId;



  //  m_nurbs = new NurbsEntity(precision,posCam,root);
 //   m_curveInstantiator = new CurveInstantiator(root);


 //   setinbetweenxsectionalpha(1.0f);
  //  setNuminbetweens(40);
  //  setinbetweenxsectionalpha(0.5);
//    setLinearInterpolateInbetweens(true);


 //   m_cloneNurbs = new NurbsClone();


    // picker
	/*Qt3DRender::QObjectPicker* spicker = new Qt3DRender::QObjectPicker();
	Qt3DRender::QPickingSettings *pickingSettings = new Qt3DRender::QPickingSettings(spicker);
	pickingSettings->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);
	pickingSettings->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick);
	pickingSettings->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontAndBackFace);
	pickingSettings->setEnabled(true);
	spicker->setEnabled(true);
	spicker->setDragEnabled(true);
	m_nurbs->addComponent(spicker);//m_ghostEntity

	connect(spicker, &Qt3DRender::QObjectPicker::pressed, [&](Qt3DRender::QPickEvent* e) {
		m_actifRayon= true;
		});
		connect(spicker, &Qt3DRender::QObjectPicker::moved, [&](Qt3DRender::QPickEvent* e) {
		m_actifRayon= false;
		});

	connect(spicker, &Qt3DRender::QObjectPicker::clicked, [&](Qt3DRender::QPickEvent* e) {

	if(m_actifRayon== true)// && e->button() == Qt3DRender::QPickEvent::Buttons::LeftButton)
	{
		int bouton = e->button();
		auto p = dynamic_cast<Qt3DRender::QPickTriangleEvent*>(e);
		if(p) {
			QVector3D pos = p->worldIntersection();
			emit sendAnimationCam(bouton,pos);
		}
	}

	});*/

}


void Manager::objToEditable(IsoSurfaceBuffer surface,QVector3D posCam, Qt3DCore::QEntity* root)
{

	m_bufferIso = surface;

	 m_nurbs = new NurbsEntity(20.0f,posCam,root);
	 m_cloneNurbs = new NurbsClone();

	 Qt3DRender::QObjectPicker* spicker = new Qt3DRender::QObjectPicker();
	 	Qt3DRender::QPickingSettings *pickingSettings = new Qt3DRender::QPickingSettings(spicker);
	 	pickingSettings->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);
	 	pickingSettings->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick);
	 	pickingSettings->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontAndBackFace);
	 	pickingSettings->setEnabled(true);
	 	spicker->setEnabled(true);
	 	spicker->setDragEnabled(true);
	 	m_nurbs->addComponent(spicker);//m_ghostEntity

	 	connect(spicker, &Qt3DRender::QObjectPicker::pressed, [&](Qt3DRender::QPickEvent* e) {
	 		m_actifRayon= true;
	 		});
	 		connect(spicker, &Qt3DRender::QObjectPicker::moved, [&](Qt3DRender::QPickEvent* e) {
	 		m_actifRayon= false;
	 		});

	 	connect(spicker, &Qt3DRender::QObjectPicker::clicked, [&](Qt3DRender::QPickEvent* e) {

	 	if(m_actifRayon== true)// && e->button() == Qt3DRender::QPickEvent::Buttons::LeftButton)
	 	{
	 		int bouton = e->button();
	 		auto p = dynamic_cast<Qt3DRender::QPickTriangleEvent*>(e);
	 		if(p) {
	 			QVector3D pos = p->worldIntersection();
	 			emit sendAnimationCam(bouton,pos);
	 		}
	 	}

	 	});


	if(m_entityObj!= nullptr)
	{
		m_entityObj->deleteLater();
		m_entityObj=nullptr;
	}
	if(m_parameterLightPosition != nullptr)
	{
		delete m_parameterLightPosition;
		m_parameterLightPosition= nullptr;
	}
	if(m_parameterColor != nullptr)
	{
		delete m_parameterColor;
		m_parameterColor= nullptr;
	}

	m_modeEditable = true;
}


void Manager::editerNurbs(/*QString path*/)
{
	//TODO sylvain
	// via NurbsParams

/*	bool valid;
	Manager::NurbsParams params = Manager::read(path, &valid);
	if (valid)
	{

	}*/
}



void Manager::setRandom3D(RandomView3D* rand)
{
	connect(rand,SIGNAL(destroy(RandomView3D*)), this, SLOT(destroyRandom3D(RandomView3D*)));
	if(rand != nullptr && rand->getRandomLineView() != nullptr)
		rand->getRandomLineView()->m_isoBuffer = m_bufferIso;

	//m_nurbs->setOrtho(rand->getRandomLineView());
	//qDebug()<<" stockage du buffer";
	m_random3d=rand;

}

void Manager::destroyRandom3D(RandomView3D* r)
{
	m_random3d =nullptr;
	//m_nurbs->setOrtho(nullptr);
}

void Manager::show()
{
	if(m_nurbs != nullptr)
		m_nurbs->setEnabled(true);
	if(m_random3d != nullptr)
			m_random3d->show();

	if(m_entityObj!= nullptr)
	{
		m_entityObj->setEnabled(true);
	}
}

void Manager::hide()
{
	if(m_nurbs != nullptr)
		m_nurbs->setEnabled(false);

	if(m_random3d != nullptr)
		m_random3d->hide();

	if(m_entityObj!= nullptr)
	{
		m_entityObj->setEnabled(false);
	}
}

void Manager::exportNurbsObj(QString path)
{
	//QString heure = QTime::currentTime().toString("hh_mm_ss");
	m_nurbs->exportObj(path );//m_nameId+"_"+heure+".obj"

	QString path2 = path.replace(".obj",".txt");
	saveNurbs(path2);
}

void Manager::importNurbsObj(QString path, Qt3DCore::QEntity* root,QVector3D posCam, QColor col)
{
	m_colorObj = col;
	m_entityObj = Qt3DHelpers::loadObj(path.toStdString().c_str(),root);


	Qt3DRender::QMaterial * m_material = new Qt3DRender::QMaterial();
				m_material->setEffect(
							Qt3DHelpers::generateImageEffect("qrc:/shaders/nurbsmaterial.frag",
									"qrc:/shaders/nurbsmaterial.vert"));


	m_parameterColor= new Qt3DRender::QParameter(QStringLiteral("colorObj"),col);
	m_parameterLightPosition= new Qt3DRender::QParameter(QStringLiteral("lightPosition"),posCam);
	m_material->addParameter(m_parameterColor);
	m_material->addParameter(m_parameterLightPosition);


	m_entityObj->addComponent(m_material);




	// picker
		Qt3DRender::QObjectPicker* spicker = new Qt3DRender::QObjectPicker();
		Qt3DRender::QPickingSettings *pickingSettings = new Qt3DRender::QPickingSettings(spicker);
		pickingSettings->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);
		pickingSettings->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick);
		pickingSettings->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontAndBackFace);
		pickingSettings->setEnabled(true);
		spicker->setEnabled(true);
		spicker->setDragEnabled(true);
		m_entityObj->addComponent(spicker);//m_ghostEntity

		connect(spicker, &Qt3DRender::QObjectPicker::pressed, [&](Qt3DRender::QPickEvent* e) {
			m_actifRayon= true;
			});
			connect(spicker, &Qt3DRender::QObjectPicker::moved, [&](Qt3DRender::QPickEvent* e) {
			m_actifRayon= false;
			});

		connect(spicker, &Qt3DRender::QObjectPicker::clicked, [&](Qt3DRender::QPickEvent* e) {

		if(m_actifRayon== true)// && e->button() == Qt3DRender::QPickEvent::Buttons::LeftButton)
		{

			int bouton = e->button();
			auto p = dynamic_cast<Qt3DRender::QPickTriangleEvent*>(e);
			if(p) {
				QVector3D pos = p->worldIntersection();
				emit sendAnimationCam(bouton,pos);
			}
		}

		});




	m_entityObj->setEnabled(true);


}



Manager::~Manager()
{
	if(m_cloneNurbs != nullptr) delete m_cloneNurbs;
//	if(m_nurbs != nullptr) delete m_nurbs;
	//if(m_curveInstantiator != nullptr) delete m_curveInstantiator;


	//if(m_cloneNurbs != nullptr) m_cloneNurbs->deleteLater();
	if(m_nurbs != nullptr) m_nurbs->deleteLater();
	if(m_curveInstantiator != nullptr) m_curveInstantiator->deleteLater();

	//workingset->removeNurbs(m_nurbsData);
}

QColor Manager::getColorNurbs()
{
	if(m_nurbs!= nullptr)
		return m_nurbs->getColorNurbs();

	return m_colorObj;
}

QColor Manager::getColorDirectrice()
{
	if(m_nurbs!= nullptr)
		return m_nurbs->getColorDirectrice();
	return QColor(0,0,0);
}

void Manager::setColorNurbs(QColor col)
{
	if(m_parameterColor) m_parameterColor->setValue(col);
	if(m_nurbs)m_nurbs->setColorNurbs(col);
	if(m_modeEditable == true)
	{
		setColorDirectrice(col);
		if(m_random3d)m_random3d->setColorCross(col);
	}
}

void Manager::setColorDirectrice(QColor col)
{
	if(m_nurbs)m_nurbs->setColorDirectrice(col);
	if(m_directriceBezier != nullptr)
	{
		m_directriceBezier->setColor(col);
		dynamic_cast<GraphicSceneEditor*>(m_directriceBezier->scene())->applyColor(m_directriceBezier->getNameNurbs(),col);
	}



}


void Manager::setPositionLight(QVector3D pos)
{
	if(m_nurbs)m_nurbs->setPositionLight( pos);
	if(m_parameterLightPosition) m_parameterLightPosition->setValue(pos);
}

void Manager::resetNurbs()
{
	 if (m_currentlyDrawnCurve!=nullptr) m_currentlyDrawnCurve->clear();
}

void Manager::setPrecision(int value)
{
	if(m_nurbs)m_nurbs->setPrecision(value);
}

int Manager::getPrecision()
{
	if(m_nurbs)
	   return m_nurbs->getPrecision();

	return -1;
}

void Manager::updateDirectriceCurve(QVector<QVector3D> listepoints)
{

	//qDebug()<<"updateDirectriceCurve listepoints :"<<listepoints[0];
	 if (m_nurbs->hasActiveXsection() )
	  {
		 // deleteXSection();
		 // qDebug()<<"==> hasActiveXsection ==true ";
		 // return;
	  }
	 /* if (m_curveDirectrice!=nullptr)
	  {

		  for(int i=0;i<listepoints.count();i++)
		  {
			  int sizeCurrent = m_curveDirectrice->getSize();
			  if(i<sizeCurrent)
			  {
				  if(!qFuzzyCompare(listepoints[i],m_curveDirectrice->getPosition(i)))
				  {
					  m_curveDirectrice->setPoint(i,listepoints[i]);
				  }
			  }
			  else
			  {
				  m_curveDirectrice->insertBack(listepoints[i]);
			  }
		  }
		  m_cloneNurbs->updateDirectrice(listepoints);
		  m_curveDirectrice->emitModelUpdated(true);
	  }*/
	 if (m_curveDirectrice!=nullptr)
	{
		 	 m_curveDirectrice->clear();

		 	 for(int i=0;i<listepoints.count();i++)
		 	 {
		 		m_curveDirectrice->insertBack(listepoints[i]);
		 	 }

		 	 m_cloneNurbs->updateDirectrice(listepoints);
		 	m_currentlyDrawnCurve = m_curveDirectrice;
		 	 m_curveDirectrice->emitModelUpdated(true);
	}

	 	 float posi = m_nurbs->getXSectionPos();
		 QVector3D normal = m_nurbs->setXsectionPos(posi);
		 QVector3D normal2 = m_cloneNurbs->nurbs()->setXsectionPos(posi);
		// qDebug()<<" slider X ==>count "<<listepoints.count();
		 if(m_directriceOk) emit sendNurbsY(m_nurbs->getPlanePosition(), normal);

}

void Manager::createDirectriceFromTangent(std::vector<QVector3D> points)
{
	qDebug()<<"attention OBSOLETE";
	m_isTangent = true;
	m_listePosBezier= points;
//	m_nurbs->createDirectriceFromTangent(points);

}

void Manager::createDirectriceFromTangent(GraphEditor_ListBezierPath* path ,QMatrix4x4 transform,QColor col)
{


	m_isTangent = true;
	m_directriceBezier = path;
	m_transformScene =  transform;
	m_listePosBezier.clear();
	m_listePosBezier.reserve(path->getPrecision());
	float coef = 0.0f;
	//float lastY = 0.0f;
	for(int i=0;i<path->getPrecision();i++)
	{
		coef = i/(path->getPrecision()-1.0);
		QPointF pos2D = path->getPosition(coef);

		if(!m_bufferIso.isValid())
		{
			qDebug()<<" Manager::createDirectriceFromTangent BUFFFER INVALIDE !!!!";
		}
		float altY = m_bufferIso.getAltitude(pos2D);

	/*	if( altY >-0.01f && altY < 0.01f &&  i>0)
		{
			altY=lastY;
		}


		lastY = altY;*/


		QVector3D pos3D(pos2D.x(),altY,pos2D.y());
	//	qDebug()<<i<<" pos3D :"<<pos3D;

		QVector3D pos = transform * pos3D;

	//	qDebug()<<i<<" pos :"<<pos;
		m_listePosBezier.push_back(pos);
	}



	QColor colCurr= NurbsWidget::getCurrentColor();
	if( m_nurbs != nullptr)
		m_nurbs->createDirectriceFromTangent(m_listePosBezier,path,transform,m_bufferIso,colCurr);
	else
		qDebug()<<"ERROR: nurbs entity est nullptr !!!!!!!!!!!!!!!!!!!";

}

void Manager::createNurbsGeneric(GraphEditor_ListBezierPath* path,RandomTransformation* transfo)
{
	// PAS UTILISABLE!!
	if(m_directriceOk == false)
	{
		//createDirectriceFromTangent(path,transfo);
		m_directriceOk=true;
	}
	else
	{
		createGeneratriceFromTangent(path,transfo,true);
	}
}

//create directrice
void Manager::curveDrawMouseBtnDownGeomIsect(QVector3D point)
{
    if (m_nurbs->hasActiveXsection() ) return;

    if (m_currentlyDrawnCurve==nullptr)
    {
        std::shared_ptr<CurveModel> curve = newCurve();
        m_nurbs->extrudeCurve()->addCurve(curve);
        m_currentlyDrawnCurve = curve.get();
        m_curveDirectrice = curve.get();
        m_curveInstantiator->selectCurve(m_currentlyDrawnCurve);
        m_curveInstantiator->enablePicking(false);
    }
    m_currentlyDrawnCurve->insertBack(point);
    m_cloneNurbs->createDirectrice(point);
}

// Intersection of cross section (xsect) is performed and the point is projected to the cross section <plane>
void Manager::curveDrawMouseBtnDownXsectIsect(int mouseX, int mouseY)
{
    if ( !m_nurbs->hasActiveXsection() ) return;

    // only the first xsection can be made by drawing a curve, the following must be made using makeinbetweenXsection(float)
    if ((m_nurbs->numXsections()>=1) && (m_currentlyDrawnCurve==nullptr))
    {
        qDebug("Only the first xsection can be made by drawing a curve, the following must be made by clicking make X-section button");
        return;
    }

    if (m_currentlyDrawnCurve==nullptr)
    {
        helperqt3d::IsectPlane plane = m_nurbs->getXsectionPlane(m_nurbs->getXSectionPos());
        std::shared_ptr<CurveModel> curve = newCurve(plane);



        m_nurbs->addUserDrawnCurveandXsection(m_nurbs->getXSectionPos(),  curve, plane);

        m_currentlyDrawnCurve = curve.get();
        m_curveInstantiator->selectCurve(m_currentlyDrawnCurve);
        m_curveInstantiator->enablePicking(false);
    }

    QVector3D point = m_curveInstantiator->unprojectFromScreen(mouseX, mouseY);
    m_currentlyDrawnCurve->insertBack(point);
}


void Manager::createGeneratriceFromTangent(QVector<PointCtrl> listeCtrls,QVector<QVector3D> listepoints,QVector<QVector3D> listeCtrl3D,QVector<QVector3D>  listeTangent3D,QVector3D cross3d, int index,bool isopen, QPointF cross)
{
	qDebug()<<"CreateGeneratrice from tangent old";
	m_nurbs->createGeneratriceFromTangent(listeCtrls,m_listePosBezier,listepoints,listeCtrl3D,listeTangent3D,cross3d,index,isopen,cross);
}

void Manager::createGeneratriceFromTangent(QVector<PointCtrl> listeCtrls,RandomTransformation* transfo,bool open,bool compute)
{

/*	QVector<QVector3D> listepoints;
	float coef = 0.0f;
	for(int i=0;i<path->getPrecision();i++)
	{
		coef = i/(path->getPrecision()-1.0);
		QPointF pos2D = path->getPosition(coef);
		QVector3D pos3D(pos2D.x(),m_bufferIso.getAltitude(pos2D),pos2D.y());
		QVector3D pos = m_transformScene * pos3D;
		listepoints.push_back(pos);
	}

	QVector<QVector3D> listeCtrl3D;
	QVector<QVector3D> listeTangentes3D;

	QVector<PointCtrl> listeCtrls = path->GetListeCtrls();

	for(int i=0;i<listeCtrls.count();i++)
	{
		QVector3D posTr =m_random3d->getRandomLineView()->viewWorldTo3dWordExtended(listeCtrls[i].m_position);
		QVector3D pos3D(posTr.x(),m_bufferIso.getAltitude(listeCtrls[i].m_position),posTr.y());
		QVector3D pos = m_transformScene * pos3D;
		listeCtrl3D.push_back(pos);

	}*/

	m_nurbs->createGeneratriceFromTangent(listeCtrls,transfo, m_listePosBezier,open,compute);
}


void Manager::createGeneratriceFromTangent(GraphEditor_ListBezierPath* path,RandomTransformation* transfo,bool compute,float coef)
{
	m_random3d->getRandomLineView()->setBezierItem(path);
	m_nurbs->createGeneratriceFromTangent(path,transfo, m_listePosBezier,compute,coef);
}

void Manager::deleteGeneratrice()
{
	m_nurbs->deleteGeneratrice();
}

void Manager::curveDrawSection(QVector<QVector3D> listepoints, int index,bool isopen)
{

	//qDebug()<<" ==>m_listePosBezier.size() : "<<m_listePosBezier.size();
	/*if( m_listePosBezier.size()> 0)
	{
		m_nurbs->createGeneratriceFromTangent(m_listePosBezier,listepoints,  index, isopen );
	}
	else
	{*/
	m_nurbs->setOpenorClosed(isopen);
	m_cloneNurbs->setOpenorClosed(isopen);



	m_nurbs->setTriangulateResolution(2 * m_nbPtsDirectrice);
	float pos  =m_nurbs->getXSectionPos();
	bool exist = m_nurbs->existsXsectionatParam(pos);
	if(exist)
	{

		m_curveInstantiator->selectCurve(m_nurbs->getCurveModel(pos));

		//get index current
		int indice =index;

		if(indice>=0)
		{

			QVector3D newPos = listepoints[indice];
			//qDebug()<<indice<<"curveDrawSection newPos "<<newPos;
			m_curveInstantiator->getSelectedCurve()->setPoint((uint)indice,newPos);
		}
		else
		{
			m_curveInstantiator->getSelectedCurve()->setAllPoints(listepoints);
		}

		m_cloneNurbs->addGeneratrice(listepoints,index);


		return ;
	}

	if (m_directriceOk ==false)
	{
		qDebug()<<"not found directrice nurbs";
		return;
	}


	helperqt3d::IsectPlane plane = m_nurbs->getXsectionPlane(m_nurbs->getXSectionPos());

	  std::shared_ptr<CurveModel> curve = newCurve(plane);

	 m_nurbs->addUserDrawnCurveandXsection(m_nurbs->getXSectionPos(),  curve, plane);

	 m_currentlyDrawnCurve = curve.get();
	 m_curveInstantiator->selectCurve(m_currentlyDrawnCurve);




	 for(int i=0;i<listepoints.count();i++)
	 {

		 m_currentlyDrawnCurve->insertBack(listepoints[i]);

	 }

	 m_cloneNurbs->addGeneratrice(listepoints,index);

	//}


}

QVector3D Manager::ptsInPlane(helperqt3d::IsectPlane plane, QVector3D points)
{
	QVector3D normalPlan = QVector3D::crossProduct(plane.xaxis.normalized(),plane.yaxis.normalized());

	float dist = points.distanceToPlane(plane.pointinplane,normalPlan);

	return (points-dist *normalPlan);
}

void Manager::curveDrawSection()
{


	//OBSOLETE
	  helperqt3d::IsectPlane plane = m_nurbs->getXsectionPlane(m_nurbs->getXSectionPos());

	  std::shared_ptr<CurveModel> curve = newCurve(plane);

	 m_nurbs->addUserDrawnCurveandXsection(m_nurbs->getXSectionPos(),  curve, plane);



	 m_currentlyDrawnCurve = curve.get();
	 m_curveInstantiator->selectCurve(m_currentlyDrawnCurve);
	 // qDebug()<<"x axis :"<<plane.xaxis<<" , y axis :"<<plane.yaxis;

	  QVector3D pos1 = plane.pointinplane -100.0f * plane.xaxis - 100.0f * plane.yaxis;
	  QVector3D pos2 = plane.pointinplane -100.0f *plane.xaxis + 100.0f *plane.yaxis;
	  QVector3D pos3 = plane.pointinplane +100.0f *plane.xaxis + 100.0f *plane.yaxis;
	  QVector3D pos4 = plane.pointinplane +100.0f *plane.xaxis - 100.0f *plane.yaxis;
	  QVector3D pos5 = plane.pointinplane -100.0f * plane.xaxis - 100.0f * plane.yaxis;


	  m_currentlyDrawnCurve->insertBack(pos1);
	  m_currentlyDrawnCurve->insertBack(pos2);
	  m_currentlyDrawnCurve->insertBack(pos3);
	  m_currentlyDrawnCurve->insertBack(pos4);
	  m_currentlyDrawnCurve->insertBack(pos5);


	  setinbetweenxsectionalpha(1.0f);
	  setNuminbetweens(12);
	  setinbetweenxsectionalpha(0.5);
	  setLinearInterpolateInbetweens(false);





}

void Manager::endCurve()
{



    if (m_currentlyDrawnCurve==nullptr) return;
    if (m_currentlyDrawnCurve->getSize()<3) return;

    emit m_currentlyDrawnCurve->modelUpdated(true); // force a geometry refresh

    if (m_nurbs->hasActiveXsection())
        m_nurbs->addCurveandXsectionEnd();




    if (m_cloneNurbs->nurbs()->hasActiveXsection())
    	m_cloneNurbs->nurbs()->addCurveandXsectionEnd();

   // m_currentlyDrawnCurve = nullptr;

}

void Manager::setOpenorClosed(bool isopen)
{
	m_nurbs->setOpenorClosed(isopen);
	m_cloneNurbs->setOpenorClosed(isopen);

}

float Manager::getXSection()
{
	return m_nurbs->getXSectionPos();
}

void Manager::setSliderXsectPos(float pos)
{
	if( m_isTangent ==true)
	{
		QVector3D normal = m_nurbs->setXsectionPos(pos);
		//if(m_directriceOk) emit sendNurbsY(m_nurbs->getPlanePosition(), normal);



		//QVector3D position(0.0f,0.0f,0.0f);
		//QVector3D normal (1.0f,0.0f,0.0f);
		//emit sendNurbsY(position, normal);
	}
	else
	{
		QVector3D normal = m_nurbs->setXsectionPos(pos);
		QVector3D normal2 = m_cloneNurbs->nurbs()->setXsectionPos(pos);
		qDebug()<<m_directriceOk<<" , " <<m_nurbs->getPlanePosition();
		if(m_directriceOk) emit sendNurbsY(m_nurbs->getPlanePosition(), normal);
	}

	//addinbetweenXsection(pos);
}

QVector3D Manager::setSliderXsectPos(float pos,QVector3D position, QVector3D normal)
{


	QVector3D normalTmp = m_nurbs->setXsectionPosWithTangent(pos,position,normal);


	//return m_nurbs->plane.pointinplane;
	return position;
//	return m_nurbs->m_planeCurrent.pointinplane;


}


QVector3D  Manager::getPositionDirectrice(float pos)
{
	if(m_directriceBezier != nullptr&& m_random3d->getRandomLineView() != nullptr )
	{
		QPointF pos2D = m_directriceBezier->getPosition(pos);
		QPointF nor2D = m_directriceBezier->getNormal(pos);
		float altY = m_random3d->getRandomLineView()->m_isoBuffer.getAltitude(pos2D);
		QVector3D pos3D(pos2D.x(),altY,pos2D.y());


		return pos3D;
	}
	return QVector3D();
}

QPointF  Manager::getNormalDirectrice(float pos)
{
	if(m_directriceBezier != nullptr)
	{

		QPointF nor2D = m_directriceBezier->getNormal(pos);
		return nor2D;

	}
	return QPointF();
}


void Manager::addSelectedCurvepoint()
{
    CurveModel* curve = getSelectedCurve();
    if (!curve) {qDebug() << "addSelectedCurvepoint no curve selected"; return;}

    int index = m_curveInstantiator->getPointSelectedIndex();
    m_curveInstantiator->unselectSelectedPoint();
    if (index==-1){ qDebug() << "addSelectedCurvepoint no point selected";  return;}

    if (curve->getSize()-1==index) return; // index is at last point of curve.

    QVector3D insertVertex = (curve->getPosition(index) + curve->getPosition(index+1)) / 2;
    curve->insertAfter(index+1,insertVertex);
    curve->emitModelUpdated(true);
}

void Manager::removeSelectedCurvepoint()
{
    CurveModel* curve = getSelectedCurve();
    if (!curve) {qDebug() << "removeCurvepoint no curve selected"; return;}

    int index = m_curveInstantiator->getPointSelectedIndex();
    m_curveInstantiator->unselectSelectedPoint();
    if (index==-1){ qDebug() << "removeCurvepoint no point selected";  return;}

    qDebug() << "removeCurvepoint point index selected" << index;

    if (curve->getSize()==1) return; // then delete curve instead

    curve->eraseAt(index);
    curve->emitModelUpdated(true);
}

void Manager::addinbetweenXsection(float pos)
{
	//qDebug()<<" Manager::addinbetweenXsection "<<pos;
    if (m_nurbs->existsXsectionatParam(pos))
    {qDebug() << "Already curve at this position/xsection"; return;}

    if (m_nurbs->numXsections()==0)
    {qDebug() << "The first xsection must be drawn using a curve"; return;}

    helperqt3d::IsectPlane plane = m_nurbs->getXsectionPlane(pos);
    std::shared_ptr<CurveModel> curve = newCurve(plane);
    m_nurbs->addinbetweenXsection(pos,curve);
   m_curveInstantiator->selectCurve(curve.get());

 /*  if(m_isTangent)
   {
	   QVector<PointCtrl*> pts = m_nurbs->getCurrentCurve(pos);
	   emit sendCurveDataTangent(pts,m_nurbs->isOpen());
   }
   else
   {*/
	   emit sendCurveData(curve->data(),m_nurbs->isOpen());
//   }
}


void Manager::addinbetweenXsectionClone(float pos, bool emitted)
{

	//qDebug()<<"addinbetweenXsectionClone "<< pos;
    if(m_isTangent)
     {
    	if(m_nurbs->getNbCurves()> 0)
    	{
		   //QVector<PointCtrl> pts = m_nurbs->getCurrentCurve(pos);

		   QVector<QVector3D> listeptsCtrls3D =m_nurbs->getCurrentPosition3D(pos);

		   QVector<QVector3D> listeTangente3D =m_nurbs->getCurrentTangente3D(pos);

		   QVector<QVector3D> globalTangente3D;
		   QVector<QVector2D> listeptslocal;
		   QVector<QVector3D> listeptsglobal;

		   for(int i=0;i<listeptsCtrls3D.count();i++)
		   {
			   listeptslocal.push_back(m_nurbs->getCurrentPLane(pos).getLocalPosition(listeptsCtrls3D[i]));
		   }

		   for(int i=0;i<listeptslocal.count();i++)
		   {
			   listeptsglobal.push_back(m_nurbs->plane.getWorldPosition(listeptslocal[i].x(), listeptslocal[i].y()));

		   }


		  for(int i=0;i<listeTangente3D.count();i++)
		   {
			  QVector2D local1 =  m_nurbs->getCurrentPLane(pos).getLocalPosition(listeTangente3D[i]);

				globalTangente3D.push_back(m_nurbs->plane.getWorldPosition(local1.x(), local1.y()));
		   }





		   emit sendCurveDataTangent2(listeptsglobal,globalTangente3D,m_nurbs->isOpen(),m_nurbs->getCrossPts(pos));
		   //emit sendCurveDataTangent(pts,m_nurbs->isOpen(),m_nurbs->getCrossPts(pos));
    	}
    	if(m_nurbs->getNbCurvesOpt()>0)
    	{
    		//GraphEditor_ListBezierPath* path =m_nurbs->getCurrentPath(pos);

    	/*	 QVector<PointCtrl> pts = path->GetListeCtrls();

    		 QVector<QPointF> listePts;
    		 QVector<QPointF> listeTangentes;
    		 for(int i = 0;i<pts.count();i++)
    		 {
    			 listePts.push_back(pts[i].m_position);
    			 listeTangentes.push_back(pts[i].m_ctrl1);
    			 listeTangentes.push_back(pts[i].m_ctrl2);
    		 }*/

    		QVector<QVector3D> listeptsCtrls3D =m_nurbs->getCurrentPosition3D(pos);

    		QVector<QVector3D> listeTangente3D =m_nurbs->getCurrentTangente3D(pos);

		   QVector<QVector3D> globalTangente3D;
		   QVector<QVector2D> listeptslocal;
		   QVector<QVector3D> listeptsglobal;



		   for(int i=0;i<listeptsCtrls3D.count();i++)
		   {

			   listeptslocal.push_back(m_nurbs->getCurrentPLane(pos).getLocalPosition(listeptsCtrls3D[i]));
		   }

		   for(int i=0;i<listeptslocal.count();i++)
		   {
			   listeptsglobal.push_back(m_nurbs->plane.getWorldPosition(listeptslocal[i].x(), listeptslocal[i].y()));

		   }


		  for(int i=0;i<listeTangente3D.count();i++)
		   {
			  QVector2D local1 =  m_nurbs->getCurrentPLane(pos).getLocalPosition(listeTangente3D[i]);

				globalTangente3D.push_back(m_nurbs->plane.getWorldPosition(local1.x(), local1.y()));
		   }

		  QPointF crosspts;
		  if(emitted)
		  {
			//  qDebug()<<"emit sendCurveDataTangent2 "<<listeptsglobal.size();
			  emit sendCurveDataTangent2(listeptsglobal,globalTangente3D,m_nurbs->isOpen(),crosspts);
		  }

    		//emit sendCurveDataTangentOpt( path);


    	}
     }
     else
     {
    	 if (m_cloneNurbs->nurbs()->existsXsectionatParam(pos))
		{qDebug() << "Already curve at this position/xsection"; return;}

		if (m_cloneNurbs->nurbs()->numXsections()==0)
		{qDebug() << "The first xsection must be drawn using a curve"; return;}

		helperqt3d::IsectPlane plane = m_cloneNurbs->nurbs()->getXsectionPlane(pos);
		std::shared_ptr<CurveModel> curve = newCurve(plane);
		m_cloneNurbs->nurbs()->addinbetweenXsection(pos,curve);

    	emit sendCurveData(curve->data(),m_nurbs->isOpen());
     }

}

void Manager::setNuminbetweens(int numinbetweens)
    {m_nurbs->setNuminbetweens(numinbetweens);}

void Manager::deleteXSection()
    {m_nurbs->deleteXSection();}

void Manager::setLinearInterpolateInbetweens(bool val)
    {m_nurbs->setLinearInterpolate(val);}

void Manager::setinbetweenxsectionalpha(float a)
    {m_nurbs->setinbetweenxsectionalpha(a);}

void Manager::setSliderInsertPointPos(float param)
    {m_nurbs->setInsertPointPos(param);}

void Manager::addSelectedCurveNurbspoint()
{
    CurveModel* curve = getSelectedCurve();
    if (!curve) {qDebug() << "addSelectedCurvepoint no curve selected"; return;}
    m_nurbs->insertCurveNurbspoint();
}

void Manager::setWireframeRendering(bool wireframe)
    {m_nurbs->setWireframRendering(wireframe);}

void Manager::setShowNurbsPoints(bool show)
    {m_nurbs->setShowNurbsPoints(show);}

void Manager::setTriangulateResolution(int resolution)
    {m_nurbs->setTriangulateResolution(resolution);}


CurveModel* Manager::getSelectedCurve()
{
    return m_curveInstantiator->getSelectedCurve();
}

// Create a new curve and register it to events and for rendering and receiving pick and move events
std::shared_ptr<CurveModel> Manager::newCurve(helperqt3d::IsectPlane plane)
{
    std::shared_ptr<CurveModel> curve = std::make_shared<CurveModel>();
    m_curves.push_back(curve);
    m_curveInstantiator->createNew(curve.get(),plane);

    return curve;
}

void Manager::deleteCurve(CurveModel* curve)
{
    auto position=find_if(m_curves.begin(), m_curves.end(),
                            [&] (std::shared_ptr<CurveModel> c) { return c.get() == curve; } );

    // if position!= curves.end()

    std::shared_ptr<CurveModel> foundcurve = *position;
    foundcurve->emitModeltobeDeleted();
    m_curves.erase(position);
}

float Manager::getAltitude(QPointF pt)
{
	if( m_bufferIso.buffer != nullptr)
	{

		double heightValue = 0.0;
		int i = 0;
		int j = 0;
		bool res = m_bufferIso.buffer->value(pt.x(),pt.y(),i,j,heightValue);
		if( !res ) return 0.0f;

		float newHeightValue =  m_bufferIso.originSample + m_bufferIso.stepSample *heightValue;
		return newHeightValue;
	}


	return 0.0f;
}



void Manager::saveNurbs(QString path)
{

//	qDebug()<<"Manager::saveNurbs OBSOLETE";
	QFile file(path);
	if(!file.open(QIODevice::WriteOnly | QIODevice::Text))
	{
		qDebug()<<" ouverture du fichier impossible "<<path;
		return;
	}
	QTextStream out(&file);

	out<<"color"<<"|"<<m_nurbs->getColorNurbs().red()<<"-"<<m_nurbs->getColorNurbs().green()<<"-"<<m_nurbs->getColorNurbs().blue()
				<<"|"<<m_nurbs->getColorDirectrice().red()<<"-"<<m_nurbs->getColorDirectrice().green()<<"-"<<m_nurbs->getColorDirectrice().blue()<<"\n";

	out<<"precision"<<"|"<<getPrecision()<<"|"<<getTimerLayer()<<"\n";


	QVector<PointCtrl> listeCtrl = m_directriceBezier->GetListeCtrls();

	out<<"directrice"<<"|"<<listeCtrl.size()<<"|"<<m_directriceBezier->isClosedPath()<<"|";
	for(int i=0;i<listeCtrl.size();i++)
	{
		out<<listeCtrl[i].m_position.x()<<"|"<<listeCtrl[i].m_position.y()<<"|"
		<<listeCtrl[i].m_ctrl1.x()<<"|"<<listeCtrl[i].m_ctrl1.y()<<"|"
				<<listeCtrl[i].m_ctrl2.x()<<"|"<<listeCtrl[i].m_ctrl2.y()<<"|";
	}

	out<<"\n";


	int nbcurve= m_nurbs->getManagerCurveOpt()->getNbCurves();
	out<<"nbcurve|"<<nbcurve<<"\n";
	for(int j=0;j<nbcurve;j++)
	{
		CurveBezierOpt curve = m_nurbs->getManagerCurveOpt()->getCurves(j);
		QVector<PointCtrl> listeCtrlCurve = curve.m_path->GetListeCtrls();
		out<<"plane|"<<curve.m_plane.pointinplane.x()<<"|"<<curve.m_plane.pointinplane.y()<<"|"<<curve.m_plane.pointinplane.z()<<"|"<<
				curve.m_plane.xaxis.x()<<"|"<<curve.m_plane.xaxis.y()<<"|"<<curve.m_plane.xaxis.z()<<"|"<<
				curve.m_plane.yaxis.x()<<"|"<<curve.m_plane.yaxis.y()<<"|"<<curve.m_plane.yaxis.z()<<"\n";

		out<<"nbpts|"<<listeCtrlCurve.size()<<"|"<<curve.m_path->isClosedPath()<<"|";
		for(int i=0;i<listeCtrlCurve.size();i++)
		{
			out<<listeCtrlCurve[i].m_position.x()<<"|"<<listeCtrlCurve[i].m_position.y()<<"|"
					<<listeCtrlCurve[i].m_ctrl1.x()<<"|"<<listeCtrlCurve[i].m_ctrl1.y()<<"|"
							<<listeCtrlCurve[i].m_ctrl2.x()<<"|"<<listeCtrlCurve[i].m_ctrl2.y()<<"|";
		}

		out<<"\n";
		out<<"coef|"<<curve.m_coef<<"\n";
		RandomTransformation* transfo = curve.m_randomTransfo;
		QPolygonF polygon = transfo->getPoly();
		out<<"randomTransformation|"<<transfo->width()<<"|"<<transfo->height()<<"\n";
		out<<"poly|"<<polygon.count()<<"|";
		for(int i=0;i<polygon.count();i++)
		{
			out<<polygon[i].x()<<"|"<<polygon[i].y()<<"|";
		}
		out<<"\n";

		std::array<double,6> direct = transfo->getAffineTransformation().direct();
		out<<"affine|"<<transfo->getAffineTransformation().width()<<"|"<<transfo->getAffineTransformation().height()<<"\n";
		out<<"direct|"<<direct[0]<<"|"<<direct[1]<<"|"<<direct[2]<<"|"<<direct[3]<<"|"<<direct[4]<<"|"<<direct[5];
		out<<"\n";

	}
	out<<"\n";

/*	for(int i=0;i<m_nurbs->getManagerCurve()->getNbCurves();i++)
	{
		CurveBezier curve = m_nurbs->getManagerCurve()->getCurves(i);

		out<<"positions"<<"|"<<i<<"|"<<curve.m_positions3D.size()<<"|";
		for(int j =0;j<curve.m_positions3D.size();j++)
		{
			out<<curve.m_positions3D[j].x()<<"|"<<curve.m_positions3D[j].y()<<"|"<<curve.m_positions3D[j].z();
		}
		out<<"\n";
		out<<"tangentes"<<"|"<<i<<"|"<<curve.m_tangente3D.size()<<"|";
		for(int j =0;j<curve.m_tangente3D.size();j++)
		{
			out<<curve.m_tangente3D[j].x()<<"|"<<curve.m_tangente3D[j].y()<<"|"<<curve.m_tangente3D[j].z();
		}
		out<<"\n";
	}*/
	file.close();


}

void Manager::loadNurbs(QString path,QMatrix4x4 transform)
{

	qDebug()<<" Manager::loadNurbs obsolete ";
	return;
	QFile file(path);
	if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		qDebug()<<"Manager Load nurbs ouverture du fichier impossible :"<<path;
		return;
	}


	QVector<PointCtrl> listeDirectrice;
	int widthTr;
	int heightTr;
	QColor colorDirectrice;
	QColor colorNurbs;
	QPolygonF polygonTr;
	QTextStream in(&file);

	while(!in.atEnd())
	{
		QString line = in.readLine();
		QStringList linesplit = line.split("|");
		if(linesplit.count()> 0)
		{
			if(linesplit[0] =="color")
			{
				QStringList color1 = linesplit[1].split("-");
				QStringList color2 = linesplit[2].split("-");
				colorNurbs = QColor(color1[0].toInt(),color1[1].toInt(),color1[2].toInt());
				colorDirectrice = QColor(color2[0].toInt(),color2[1].toInt(),color2[2].toInt());
			}
			else if(linesplit[0] =="precision")
			{
				int precis = linesplit[1].toInt();
			}
			else if(linesplit[0] =="directrice")
			{
				int nbptsDir = linesplit[1].toInt();
				int closed = linesplit[2].toInt();

				bool open = true;
				if(closed==1 ) open = false;

				qDebug()<<" nb points directrice :"<<nbptsDir;

				for(int i=0;i<nbptsDir*6;i+=6)
				{
					QPointF pos(linesplit[i+3].toFloat(),linesplit[i+4].toFloat());
					QPointF ctrl1(linesplit[i+5].toFloat(),linesplit[i+6].toFloat());
					QPointF ctrl2(linesplit[i+7].toFloat(),linesplit[i+8].toFloat());

					listeDirectrice.push_back(PointCtrl(pos,ctrl1,ctrl2));

				}
				qDebug()<<" listeDirectrice.count():"<<listeDirectrice.count();
				emit generateDirectrice(listeDirectrice,colorDirectrice,open);
			}
			else if(linesplit[0] =="nbcurve")
			{
				//pas necessaire
			}
			else if(linesplit[0] =="plane")
			{
				QVector3D pos(linesplit[1].toFloat(),linesplit[2].toFloat(),linesplit[3].toFloat());
				QVector3D axeX(linesplit[4].toFloat(),linesplit[5].toFloat(),linesplit[6].toFloat());
				QVector3D axeY(linesplit[7].toFloat(),linesplit[8].toFloat(),linesplit[9].toFloat());
			}
			else if(linesplit[0] =="nbpts")
			{

				QVector<PointCtrl> listeGeneratrice;
				int nbptsGene = linesplit[1].toInt();
				int closedG = linesplit[2].toInt();
				for(int i=0;i<nbptsGene*6;i+=6)
				{
					QPointF pos(linesplit[i+3].toFloat(),linesplit[i+4].toFloat());
					QPointF ctrl1(linesplit[i+5].toFloat(),linesplit[i+6].toFloat());
					QPointF ctrl2(linesplit[i+7].toFloat(),linesplit[i+8].toFloat());

					listeGeneratrice.push_back(PointCtrl(pos,ctrl1,ctrl2));
				}

			}
			else if(linesplit[0] =="randomTransformation")
			{

				widthTr = linesplit[1].toInt();
				heightTr = linesplit[2].toInt();
			}
			else if(linesplit[0] =="poly")
			{
				polygonTr.clear();
				int nbpoly = linesplit[1].toInt();
				for(int i=0;i<nbpoly*2;i+=2)
				{
					QPointF pos(linesplit[i+2].toFloat(),linesplit[i+3].toFloat());
					polygonTr<<pos;

				}
			}
			else if(linesplit[0] =="direct")
			{
				std::array<double,6> direct;
				direct[0] = linesplit[1].toDouble();
				direct[1] = linesplit[2].toDouble();
				direct[2] = linesplit[3].toDouble();
				direct[3] = linesplit[4].toDouble();
				direct[4] = linesplit[5].toDouble();
				direct[5] = linesplit[6].toDouble();

				Affine2DTransformation affine(widthTr,heightTr,direct);

				RandomTransformation* transfo = new RandomTransformation(heightTr,polygonTr,affine);




			}
		}
	}
	file.close();



	qDebug()<<" import nurbs ok";

}

Manager::NurbsParams Manager::read(QString path, bool* ok)
{
	NurbsParams params;

	QFile file(path);
	if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		qDebug()<<"Manager Load nurbs ouverture du fichier impossible :"<<path;
		if (ok!=nullptr)
		{
			*ok = false;
		}
		return params;
	}


	params.timerLayer = -1;
	int widthTr;
	int heightTr;
	QPolygonF polygonTr;
	int widthAffine,heightAffine;
	bool openGene = true;
	double coef = 0;
	QVector3D posplane;
	QVector3D xaxis;
	QVector3D yaxis;
	QVector<PointCtrl> listeGeneratrice;
	QTextStream in(&file);

	int nbCurve = 0;
	int cptCurve =0;
	while(!in.atEnd())
	{
		QString line = in.readLine();
		QStringList linesplit = line.split("|");
		if(linesplit.count()> 1)
		{
			if(linesplit[0] =="color")
			{
				QStringList color1 = linesplit[1].split("-");
				params.color = QColor(color1[0].toInt(),color1[1].toInt(),color1[2].toInt());
			}
			else if(linesplit[0] =="precision")
			{
				params.precision = linesplit[1].toInt();
				if(linesplit.count() >=3) params.timerLayer = linesplit[2].toInt();
			}
			else if(linesplit[0] =="directrice" && linesplit.count()> 2)
			{
				int nbptsDir = linesplit[1].toInt();
				int closed = linesplit[2].toInt();

				params.directrixClosed = false;
				if (linesplit.count()>2+nbptsDir*6)
				{
					if(closed==1 ) params.directrixClosed = true;

					for(int i=0;i<nbptsDir*6;i+=6)
					{
						QPointF pos(linesplit[i+3].toFloat(),linesplit[i+4].toFloat());
						QPointF ctrl1(linesplit[i+5].toFloat(),linesplit[i+6].toFloat());
						QPointF ctrl2(linesplit[i+7].toFloat(),linesplit[i+8].toFloat());

						params.directrix.push_back(PointCtrl(pos,ctrl1,ctrl2));
					}
				}
				else
				{
					qDebug() << "Directrix with bad number of points or tangents : " << path;
				}
			}
			else if (linesplit[0] =="directrice")
			{
				qDebug() << "Directrix without a number of points or a closed information : " << path;
			}
			else if(linesplit[0] =="nbcurve")
			{
				nbCurve =linesplit[1].toInt();
			}
			else if (linesplit[0] =="coef")
			{
				coef = linesplit[1].toDouble();
			}
			else if(linesplit[0] =="plane")
			{
				if(linesplit.count()>9)
				{
					posplane = QVector3D(linesplit[1].toFloat(),linesplit[2].toFloat(),linesplit[3].toFloat());
					xaxis = QVector3D(linesplit[4].toFloat(),linesplit[5].toFloat(),linesplit[6].toFloat());
					yaxis = QVector3D(linesplit[7].toFloat(),linesplit[8].toFloat(),linesplit[9].toFloat());
				}
				else
				{
					qDebug() << "Directrix with plane ortho not defined : " << path;
				}
			}
			else if(linesplit[0] =="nbpts" && linesplit.count()> 2)
			{
				listeGeneratrice.clear();
				openGene = true;

				int nbptsGene = linesplit[1].toInt();
				int closedG = linesplit[2].toInt();
				if (linesplit.count()> 2+nbptsGene*6)
				{
					if(closedG==1 ) openGene = false;
					for(int i=0;i<nbptsGene*6;i+=6)
					{
						QPointF pos(linesplit[i+3].toFloat(),linesplit[i+4].toFloat());
						QPointF ctrl1(linesplit[i+5].toFloat(),linesplit[i+6].toFloat());
						QPointF ctrl2(linesplit[i+7].toFloat(),linesplit[i+8].toFloat());


						listeGeneratrice.push_back(PointCtrl(pos,ctrl1,ctrl2));
					}
				}
				else
				{
					qDebug() << "Generatrix n°" << cptCurve << " with bad number of points or tangents : " << path;
				}
			}
			else if (linesplit[0] =="nbpts")
			{
				qDebug() << "Genetratix n°" << cptCurve << " without a number of points or a closed information : " << path;
			}
			else if(linesplit[0] =="randomTransformation" && linesplit.count()> 2)
			{

				widthTr = linesplit[1].toInt();
				heightTr = linesplit[2].toInt();
			}
			else if (linesplit[0] =="randomTransformation")
			{
				qDebug() << "Generatrix n°" << cptCurve << " with bad random transformation width or height : " << path;
			}
			else if(linesplit[0] =="poly")
			{
				polygonTr.clear();
				int nbpoly = linesplit[1].toInt();
				if (linesplit.count()> 1+2*nbpoly)
				{
					for(int i=0;i<nbpoly*2;i+=2)
					{
						QPointF pos(linesplit[i+2].toFloat(),linesplit[i+3].toFloat());
						polygonTr.push_back(pos);

					}
				}
				else
				{
					qDebug() << "Generatrix n°" << cptCurve << " random polygon with bad number of points : " << path;
				}
			}
			else if(linesplit[0] =="affine" && linesplit.count()> 2)
			{
				widthAffine = linesplit[1].toInt();
				heightAffine = linesplit[2].toInt();
			}
			else if (linesplit[0] =="affine")
			{
				qDebug() << "Generatrix n°" << cptCurve << " affine transformation with bad width or height : " << path;
			}
			else if(linesplit[0] =="direct" && linesplit.count()> 6)
			{
				std::array<double,6> direct;
				direct[0] = linesplit[1].toDouble();
				direct[1] = linesplit[2].toDouble();
				direct[2] = linesplit[3].toDouble();
				direct[3] = linesplit[4].toDouble();
				direct[4] = linesplit[5].toDouble();
				direct[5] = linesplit[6].toDouble();

				Generatrix generatrix;
				generatrix.widthAffine = widthAffine;
				generatrix.heightAffine = heightAffine;
				generatrix.directAffine = direct;

				generatrix.widthRandom = widthTr;
				generatrix.heightRandom = heightTr;
				generatrix.polygonRandom = polygonTr;

				generatrix.generatrix = listeGeneratrice;
				generatrix.generatrixClosed = !openGene;
				generatrix.pos = coef;

				generatrix.positionPlane = posplane;
				generatrix.axeXPlane = xaxis;
				generatrix.axeYPlane = yaxis;

				params.generatrices.append(generatrix);

				cptCurve++;
			}
			else if (linesplit[0] =="direct")
			{
				qDebug() << "Generatrix n°" << cptCurve << " affine with bad direct transformation : " << path;
			}
		}
	}
	file.close();

	if (ok!=nullptr)
	{
		*ok = true;
	}
	return params;
}

bool Manager::write(QString path, NurbsParams params)
{

	qDebug()<<" Save nurbs : "<<path;
	QFile file(path);
	if(!file.open(QIODevice::WriteOnly | QIODevice::Text))
	{
		qDebug()<<" ouverture du fichier impossible "<<path;
		return false;
	}
	QTextStream out(&file);

	out<<"color"<<"|"<<params.color.red()<<"-"<<params.color.green()<<"-"<<params.color.blue()
				<<"|"<<params.color.red()<<"-"<<params.color.green()<<"-"<<params.color.blue()<<"\n";

	out<<"precision"<<"|"<<params.precision<<"|"<<params.timerLayer<<"\n";


	out<<"directrice"<<"|"<<params.directrix.size()<<"|"<<params.directrixClosed<<"|";
	for(int i=0;i<params.directrix.size();i++)
	{
		out<<params.directrix[i].m_position.x()<<"|"<<params.directrix[i].m_position.y()<<"|"
		<<params.directrix[i].m_ctrl1.x()<<"|"<<params.directrix[i].m_ctrl1.y()<<"|"
				<<params.directrix[i].m_ctrl2.x()<<"|"<<params.directrix[i].m_ctrl2.y()<<"|";
	}

	out<<"\n";


	int nbcurve= params.generatrices.size();
	out<<"nbcurve|"<<nbcurve<<"\n";
	for(int j=0;j<nbcurve;j++)
	{
		Generatrix generatrix = params.generatrices[j];
		out<<"coef|"<<generatrix.pos;
		out<<"\n";
		out<<"plane|"<<generatrix.positionPlane.x()<<"|"<<generatrix.positionPlane.y()<<"|"<<generatrix.positionPlane.z()<<"|"<<
				generatrix.axeXPlane.x()<<"|"<<generatrix.axeXPlane.y()<<"|"<<generatrix.axeXPlane.z()<<"|"<<
				generatrix.axeYPlane.x()<<"|"<<generatrix.axeYPlane.y()<<"|"<<generatrix.axeYPlane.z()<<"\n";

		out<<"nbpts|"<<generatrix.generatrix.size()<<"|"<<generatrix.generatrixClosed<<"|";
		for(int i=0;i<generatrix.generatrix.size();i++)
		{
			out<<generatrix.generatrix[i].m_position.x()<<"|"<<generatrix.generatrix[i].m_position.y()<<"|"
					<<generatrix.generatrix[i].m_ctrl1.x()<<"|"<<generatrix.generatrix[i].m_ctrl1.y()<<"|"
							<<generatrix.generatrix[i].m_ctrl2.x()<<"|"<<generatrix.generatrix[i].m_ctrl2.y()<<"|";
		}

		out<<"\n";
		QPolygonF polygon = generatrix.polygonRandom;
		out<<"randomTransformation|"<<generatrix.widthRandom<<"|"<<generatrix.heightRandom<<"\n";
		out<<"poly|"<<polygon.count()<<"|";
		for(int i=0;i<polygon.count();i++)
		{
			out<<polygon[i].x()<<"|"<<polygon[i].y()<<"|";
		}
		out<<"\n";

		std::array<double,6> direct = generatrix.directAffine;
		out<<"affine|"<<generatrix.widthAffine<<"|"<<generatrix.heightAffine<<"\n";
		out<<"direct|"<<direct[0]<<"|"<<direct[1]<<"|"<<direct[2]<<"|"<<direct[3]<<"|"<<direct[4]<<"|"<<direct[5];
		out<<"\n";

	}
	out<<"\n";

	file.close();
	return true;
}


