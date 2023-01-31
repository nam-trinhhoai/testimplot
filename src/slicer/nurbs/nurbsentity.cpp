#include "nurbsentity.h"
#include <QSceneLoader>
#include <QEntity>
#include <QTransform>
#include "meshgeometry.h"

#include <QVector>
#include <Qt3DRender/QEffect>
#include <Qt3DRender/QMaterial>
#include <Qt3DRender/QTechnique>
#include <Qt3DRender/QRenderPass>
#include <Qt3DRender/QShaderProgram>
#include <Qt3DRender/QGraphicsApiFilter>
#include <Qt3DRender/QParameter>
#include <Qt3DExtras>
#include <QPhongMaterial>

#include <tinynurbs/tinynurbs.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/vec3.hpp>
#include <glm/gtx/io.hpp>

#include "manager.h"
#include "curvemodel.h"

#include "helperqt3d.h"
#include "curvelistener.h"
#include <functional>

#include "meshgeometry.h"
#include "pointslinesgeometry.h"
#include "splinemath.h"
#include "qt3dhelpers.h"

using namespace Qt3DRender;


NurbsEntity::NurbsEntity(int precision,QVector3D posCam, Qt3DCore::QNode *parent) : Qt3DCore::QEntity(parent),
    m_extrudecurvelistener(new CurveListener(this)) ,
    m_nurbscurvelistener(new CurveListener(this))
{

	m_parent = parent;
	m_material = new Qt3DRender::QMaterial();
			m_material->setEffect(
						Qt3DHelpers::generateImageEffect("qrc:/shaders/nurbsmaterial.frag",
								"qrc:/shaders/nurbsmaterial.vert"));



/*	Qt3DExtras::QPhongMaterial* material = new Qt3DExtras::QPhongMaterial(parent);
	material->setAmbient(QColor(0, 0, 0, 50));
	material->setDiffuse(QColor(0,0, 255, 255));
	material->setSpecular(QColor(255, 255, 255, 255));
*/

	QVector3D colorObj(0,0,1);

	m_managerCurves = new CurveBezierManager();
	m_managerCurvesOpt = new CurveBezierOptManager();

	m_parameterColor= new Qt3DRender::QParameter(QStringLiteral("colorObj"),colorObj);
	m_parameterLightPosition= new Qt3DRender::QParameter(QStringLiteral("lightPosition"),posCam);
	m_material->addParameter(m_parameterColor);
	m_material->addParameter(m_parameterLightPosition);

    m_currentMesh = new MeshGeometry2();//  type Qt3DRender::QGeometryRenderer
    m_matNurbs= helperqt3d::makeSimpleMaterial2(QColorConstants::White);
//   QMaterial* m_matNurb = helperqt3d::makeShaderMaterial(QStringLiteral("qrc:/shaders/nurbsmaterial.vert"),QStringLiteral("qrc:/shaders/nurbsmaterial.frag"));
   // QMaterial* mymat = helperqt3d::makeShaderMaterial(QStringLiteral("qrc:/shaders/nurbsmaterial.vert"),QStringLiteral("qrc:/shaders/nurbsmaterial.frag"));
    Qt3DCore::QEntity* meshEntity =  new Qt3DCore::QEntity(this);
    helperqt3d::makeEntity(meshEntity,m_material,m_currentMesh);

    m_splineGeometry = new PointsLinesGeometry();//  type Qt3DRender::QGeometryRenderer
    Qt3DCore::QEntity* curveEntity =  new Qt3DCore::QEntity(this);
    Qt3DExtras::QDiffuseSpecularMaterial* mat = helperqt3d::makeSimpleMaterial(QColorConstants::Red);
    helperqt3d::makeEntity(curveEntity, mat, m_splineGeometry, true);

    m_extrudeSplineGeometry = new PointsLinesGeometry();//  type Qt3DRender::QGeometryRenderer
    Qt3DCore::QEntity* extrudeEntity =  new Qt3DCore::QEntity(this);
    m_matdirectrice = helperqt3d::makeSimpleMaterial(QColorConstants::Yellow);
    helperqt3d::makeEntity(extrudeEntity, m_matdirectrice, m_extrudeSplineGeometry, true);

    m_xSectionframeGeometry = new PointsLinesGeometry();//  type Qt3DRender::QGeometryRenderer
    Qt3DCore::QEntity* xsectEntity =  new Qt3DCore::QEntity(this);
    Qt3DExtras::QDiffuseSpecularMaterial* xmat = helperqt3d::makeSimpleMaterial(QColorConstants::Blue);
    helperqt3d::makeEntity(xsectEntity, xmat, m_xSectionframeGeometry, true);

    m_pointsGeometry = new PointsLinesGeometry();//  type Qt3DRender::QGeometryRenderer
    Qt3DCore::QEntity* pointsEntity =  new Qt3DCore::QEntity(this);
    Qt3DExtras::QDiffuseSpecularMaterial* ptsmat = helperqt3d::makeSimpleMaterial(QColorConstants::White);
    helperqt3d::makeEntity(pointsEntity, ptsmat, m_pointsGeometry, false);

    m_triangulateResolution = precision;

   // if (Manager::singleton()==nullptr)    qDebug("NULLPTR");

   // Manager::singleton()->setNurbs(this);

    using namespace std::placeholders;
    m_extrudecurvelistener->setCallbackCurveUpdated( std::bind(&NurbsEntity::extrudecurveUpdated,     this , _1));
    m_extrudecurvelistener->setCallbackCurveDeleted( std::bind(&NurbsEntity::extrudecurveDeleted,     this));
    //  m_extrudecurvelistener->setCallbackCurveSelected(std::bind(&NurbsEntity::extrudecurveSetSelected, this, _1, _2));

    m_nurbscurvelistener->setCallbackCurveUpdated( std::bind(&NurbsEntity::nurbscurveUpdated,     this , _1, _2));
 //   m_nurbscurvelistener->setCallbackCurveDeleted( std::bind(&NurbsEntity::nurbscurveDeleted,     this));
    m_nurbscurvelistener->setCallbackCurveSelected(std::bind(&NurbsEntity::nurbscurveSetSelected, this, _1, _2));


}


NurbsEntity::~NurbsEntity()
{
	if(m_nurbscurvelistener != nullptr)
	{
		m_nurbscurvelistener->deleteLater();
		m_nurbscurvelistener = nullptr;
	}
	if(m_extrudecurvelistener != nullptr)
	{
		m_extrudecurvelistener->deleteLater();
		m_extrudecurvelistener = nullptr;
	}

    if(m_managerCurves != nullptr)
    {
    	delete m_managerCurves;
    	m_managerCurves= nullptr;
    }
    if(m_managerCurvesOpt != nullptr)
	{
		delete m_managerCurvesOpt;
		m_managerCurvesOpt= nullptr;
	}
}

QColor NurbsEntity::getColorNurbs()
{
	return m_colorNurbs;
}

QColor NurbsEntity::getColorDirectrice()
{
	 return m_colorDirectrice;
}


void NurbsEntity::setColorNurbs(QColor col)
{
	 m_colorNurbs= col;

	 m_matNurbs->setAmbient(QColor(col.red()*0.15f,col.green()*0.15f,col.blue()*0.15f,255));
	 m_matNurbs->setDiffuse(col);



	 for(int i =0; i<m_managerCurvesOpt->getNbCurves();i++)
	 {
		 CurveBezierOpt curve = m_managerCurvesOpt->getCurves(i);
		 curve.setColor(col);


	 }

	QVector3D vecColor(col.redF(),col.greenF(),col.blueF());
	m_parameterColor->setValue(vecColor);
//	m_matNurbs->setAmbient(QColor(col.red()*0.3f,col.green()*0.3f,col.blue()*0.3f,255));
//	m_matNurbs->setDiffuse(col);
}

void NurbsEntity::setColorDirectrice(QColor col)
{
	m_colorDirectrice = col;
	m_matdirectrice->setAmbient(col);
	m_matdirectrice->setDiffuse(col);
}

void NurbsEntity::setPositionLight(QVector3D pos)
{
	//qDebug()<<"position light :"<<pos;
	m_parameterLightPosition->setValue(pos);
}

helperqt3d::IsectPlane NurbsEntity::getXsectionPlane(float param)
{
    float param2=    m_extrudeSpline.todistance(param);

    glm::vec3 start =  m_extrudeSpline.getPosition(param2);
    glm::vec3 tang =   m_extrudeSpline.getTangent(param2);



    //glm::vec3 startbezier =  m_extrudeBezier.getPosition(param2);

   // qDebug()<<" param2 , "<<startbezier.x << " ,  "<<startbezier.z;
 /*   float param2=    m_extrudeBezier.todistance(param);

       glm::vec3 start =  m_extrudeBezier.getPosition(param2);
       glm::vec3 tang =   m_extrudeBezier.getTangent(param2);*/

    QVector3D pos = QVector3D(start.x,start.y,start.z);
    QVector3D vector1InPlane, vector2InPlane;
    helperqt3d::getPlanevectorsFromNormal(QVector3D(tang.x, tang.y, tang.z),vector1InPlane,vector2InPlane);

    return helperqt3d::IsectPlane{pos,vector1InPlane,vector2InPlane};
}

void NurbsEntity::nurbscurveDeleted()
{
    qDebug("NurbsEntity curveDeleted");
    recalculateAndUpdateGeometry();
}


// returns two vectors that are different and orthogonal to normal, .i.e if normal describes a plane then the two vectors are in the plane
/*void NurbsEntity::getPlanevectorsFromNormal(const QVector3D& normal, QVector3D& vector1InPlane, QVector3D& vector2InPlane)
{
    QVector3D upvector(0,-1,0);
    float angle = acos(QVector3D::dotProduct(normal,upvector));

    if (angle<0.1) qDebug() << "tangent vector almost paralell with upvector, crossproduct might be inaccurate";

    vector1InPlane = QVector3D::crossProduct(-upvector,normal);   // plane should be tangential to curve as seen from above
    vector2InPlane = upvector;                                    // plane should be vertical

    vector1InPlane.normalize();
    vector2InPlane.normalize();
}*/

/*
// returns two vectors that are different and orthogonal to normal, .i.e if normal describes a plane then the two vectors are in the plane
void NurbsEntity::getPlanevectorsFromNormal(const QVector3D& normal, QVector3D& vector1InPlane, QVector3D& vector2InPlane)
{

    QVector3D differentfromNormal(0,0,1);

    float angle = acos(QVector3D::dotProduct(normal,differentfromNormal));
    if (angle<0.1) differentfromNormal=QVector3D(1,0,0); // "tangent (almost) paralell with initial <differentfromNormal>, results may be inaccurate so chose other

    vector1InPlane = QVector3D::crossProduct(-differentfromNormal,normal);      // basisvector x axis
    vector2InPlane = QVector3D::crossProduct(vector1InPlane,normal);            // basisvector y axis

    vector1InPlane.normalize();
    vector2InPlane.normalize();
}
*/

QVector3D NurbsEntity::setXsectionPosWithTangent(float pos,QVector3D position, QVector3D normal)
{
	 m_xSectionParam = pos;

	 QVector3D dir1(0.0f,-1.0f,0.0f);
	 normal = normal.normalized();
	 QVector3D dir2= QVector3D::crossProduct(normal,dir1).normalized();


	// position.setY(0.0f);



	 plane.pointinplane = position;
	 plane.xaxis = dir1;
	 plane.yaxis = -dir2;


	// qDebug()<<"set slider getnormal : "<<plane.getNormal();




   /* if (!m_extrudeSpline.isValid()) return QVector3D();

    m_xSectionParam = pos;

    helperqt3d::IsectPlane plane = getXsectionPlane(m_xSectionParam);
    QVector3D start=plane.pointinplane,ex=plane.xaxis,ey=plane.yaxis;

    m_positionPlane = start;
   // qDebug()<<" START :"<<plane.getNormal();
    CurveModel lines;
    lines.insertBack(start-ex-ey);
    lines.insertBack(start-ex+ey);
    lines.insertBack(start+ex+ey);
    lines.insertBack(start+ex-ey);
    lines.insertBack(start-ex-ey);

    m_xSectionframeGeometry->updateData(lines.data());
*/

	// qDebug()<<"plane.getNormal  :"<<plane.getNormal();
    return plane.getNormal();


}

QVector3D NurbsEntity::setXsectionPos(float pos)
{

	//qDebug()<<"NurbsEntity::setXsectionPos "<<pos;
    if (!m_extrudeSpline.isValid()) return QVector3D();

    m_xSectionParam = pos;

    helperqt3d::IsectPlane plane = getXsectionPlane(m_xSectionParam);
    QVector3D start=plane.pointinplane,ex=plane.xaxis,ey=plane.yaxis;

    m_positionPlane = start;
   // qDebug()<<" START :"<<plane.getNormal();
    CurveModel lines;
    lines.insertBack(start-ex-ey);
    lines.insertBack(start-ex+ey);
    lines.insertBack(start+ex+ey);
    lines.insertBack(start+ex-ey);
    lines.insertBack(start-ex-ey);

    m_xSectionframeGeometry->updateData(lines.data());



    return plane.getNormal();

}

void NurbsEntity::setInsertPointPos(float param)
{
    m_insertPointParam = param;
    //qDebug() << "setInsertPointPos" << m_insertPointParam;

    if (m_xsections.size()==0) return;

    auto position=find_if(m_xsections.begin(), m_xsections.end(),    // find the curvecontroller that is associated with cm
                            [&] (const auto&  c) {return c.second.param == m_xSectionParam; } );

    if (position!=m_xsections.end())
    {   // render the insertPoint position
        CurveModel cm;
        glm::vec3 p = position->second.splinemath.getPosition(m_insertPointParam);

        cm.insertBack(QVector3D(p[0],p[1],p[2]));
        m_pointsGeometry->updateData(cm.data());
    }

}

void NurbsEntity::setShowNurbsPoints(bool show)
{
    m_showNURBSPoints = show;
    recalculateAndUpdateGeometry();
}

void NurbsEntity::createDirectriceFromTangent(std::vector<QVector3D> points,GraphEditor_ListBezierPath* path,QMatrix4x4 transformScene,IsoSurfaceBuffer bufferIso, QColor col)
{

	//m_extrudeSplineGeometry->updateData(points);


	 setColorNurbs(col);
	 setColorDirectrice(col);

	// todo
	int nbCurve = m_managerCurvesOpt->getNbCurves();
	CurveBezierOpt directrice(path);

	m_managerCurvesOpt->setDirectrice(directrice,transformScene,bufferIso);


	QVector<QVector3D> dir= m_managerCurvesOpt->getDirectriceInterpolated(getPrecision());

	std::vector<QVector3D> points2;
	for(int i=0;i<getPrecision();i++)
	{
		points2.push_back(dir[i]);
	}
	m_extrudeSplineGeometry->updateData(points2);



	if(nbCurve> 0)
	{

	//	UpdateGeometryWithTangent2(points,m_isOpenNurbs);
		UpdateGeometryWithTangentOpt(m_isOpenNurbs);
	}

}

void NurbsEntity::deleteGeneratrice()
{
	m_currentMesh->setEmptyData();

}

void NurbsEntity::createGeneratriceFromTangent(GraphEditor_ListBezierPath* path,RandomTransformation* transfo,std::vector<QVector3D> directrice,bool compute, float coef)
{
	m_isOpenNurbs = !path->isClosedPath();

	if(m_managerCurvesOpt->getNbCurves()== 0 )
	{
		if(m_xSectionParam <0.05f)
		{
			QVector3D normal= (directrice[1] -directrice[0]).normalized();
			QVector3D dir1(0.0f,-1.0f,0.0f);
			QVector3D dir2= QVector3D::crossProduct(normal,dir1).normalized();

			m_planeCurrent.pointinplane = directrice[0];
			m_planeCurrent.xaxis =dir1;
			m_planeCurrent.yaxis = dir2;

		   plane = m_planeCurrent;
		}



	}

	//if(coef >= 0.0f) m_xSectionParam =coef;

	int indexSection = m_managerCurvesOpt->exist(m_xSectionParam);
	QVector<PointCtrl> pts = path->GetListeCtrls();

		QVector<QPointF> listePts;
		 QVector<QPointF> listeTangentes;
		 for(int i = 0;i<pts.count();i++)
		 {

			 listePts.push_back(pts[i].m_position);
			 listeTangentes.push_back(pts[i].m_ctrl1);
			 listeTangentes.push_back(pts[i].m_ctrl2);
		 }

	   QVector<QVector3D> listeptsCtrls3D =m_managerCurvesOpt->get3DWorld(listePts,transfo);
	   QVector<QVector3D> listeTangente3D =m_managerCurvesOpt->get3DWorld(listeTangentes,transfo);


	//   qDebug()<<" NurbsEntity::createGeneratriceFromTangent ; indexSection :"<<indexSection<<" , m_xSectionParam :"<<m_xSectionParam;
	//   qDebug()<<" plane :"<<plane.pointinplane<<" normal :"<<plane.getNormal()<<" , coef :"<<coef;
	if(indexSection>= 0)
	{
		m_managerCurvesOpt->supprimer(indexSection);
		CurveBezierOpt curveBB(m_xSectionParam,path,plane,listeptsCtrls3D,listeTangente3D,transfo,false);
		m_managerCurvesOpt->add(curveBB);

		if(compute)UpdateGeometryWithTangentOpt(m_isOpenNurbs);
	}
	else
	{

		CurveBezierOpt curveB(m_xSectionParam,path,plane,listeptsCtrls3D,listeTangente3D,transfo,true);
	//	CurveBezier curveB(m_xSectionParam,listeCtrls,listepoints,cross,plane,listeCtrl3D,listeTangent3D,cross3d);
		m_managerCurvesOpt->add(curveB);

		if(compute)UpdateGeometryWithTangentOpt(m_isOpenNurbs);

	}

}

void NurbsEntity::createGeneratriceFromTangent(QVector<PointCtrl> pts,RandomTransformation* transfo,std::vector<QVector3D> directrice,bool open,bool compute)
{
	m_isOpenNurbs = open;
	if(m_managerCurvesOpt->getNbCurves()== 0 )
	{
		QVector3D normal= (directrice[1] -directrice[0]).normalized();
		QVector3D dir1(0.0f,-1.0f,0.0f);
		QVector3D dir2= QVector3D::crossProduct(normal,dir1).normalized();

		m_planeCurrent.pointinplane = directrice[0];
		m_planeCurrent.xaxis =dir1;
	    m_planeCurrent.yaxis = dir2;

	   plane = m_planeCurrent;
	}


	int indexSection = m_managerCurvesOpt->exist(m_xSectionParam);
//	QVector<PointCtrl> pts = path->GetListeCtrls();

		QVector<QPointF> listePts;
		 QVector<QPointF> listeTangentes;
		 for(int i = 0;i<pts.count();i++)
		 {
			 listePts.push_back(pts[i].m_position);
			 listeTangentes.push_back(pts[i].m_ctrl1);
			 listeTangentes.push_back(pts[i].m_ctrl2);
		 }
	   QVector<QVector3D> listeptsCtrls3D =m_managerCurvesOpt->get3DWorld(listePts,transfo);
	   QVector<QVector3D> listeTangente3D =m_managerCurvesOpt->get3DWorld(listeTangentes,transfo);



	if(indexSection>= 0)
	{
		m_managerCurvesOpt->supprimer(indexSection);
		CurveBezierOpt curveBB(m_xSectionParam,nullptr,plane,listeptsCtrls3D,listeTangente3D,transfo,false);
		m_managerCurvesOpt->add(curveBB);

		if(compute)UpdateGeometryWithTangentOpt(m_isOpenNurbs);
	}
	else
	{


		CurveBezierOpt curveB(m_xSectionParam,nullptr,plane,listeptsCtrls3D,listeTangente3D,transfo,true);
	//	CurveBezier curveB(m_xSectionParam,listeCtrls,listepoints,cross,plane,listeCtrl3D,listeTangent3D,cross3d);
		m_managerCurvesOpt->add(curveB);

		if(compute)UpdateGeometryWithTangentOpt(m_isOpenNurbs);

	}

}


void NurbsEntity::createGeneratriceFromTangent(QVector<PointCtrl> listeCtrls,std::vector<QVector3D> directrice, QVector<QVector3D> listepoints, QVector<QVector3D> listeCtrl3D,QVector<QVector3D>  listeTangent3D,QVector3D cross3d, int index,bool isopen, QPointF cross)
{
	m_isOpenNurbs = isopen;
	if(m_managerCurves->getNbCurves()== 0 )
	{
		if(m_xSectionParam <0.1f)
		{
			QVector3D normal= (directrice[1] -directrice[0]).normalized();
			QVector3D dir1(0.0f,-1.0f,0.0f);
			QVector3D dir2= QVector3D::crossProduct(normal,dir1).normalized();

			m_planeCurrent.pointinplane = directrice[0];
			m_planeCurrent.xaxis =dir1;
			m_planeCurrent.yaxis = dir2;


		   plane = m_planeCurrent;
		}
	}


	int indexSection = m_managerCurves->exist(m_xSectionParam);



	if(indexSection>= 0)
	{
		m_managerCurves->supprimer(indexSection);
		CurveBezier curveBB(m_xSectionParam,listeCtrls,listepoints,cross,plane,listeCtrl3D,listeTangent3D,cross3d);
		m_managerCurves->add(curveBB);

		UpdateGeometryWithTangent2(directrice,isopen);
	}
	else
	{
		CurveBezier curveB(m_xSectionParam,listeCtrls,listepoints,cross,plane,listeCtrl3D,listeTangent3D,cross3d);
		m_managerCurves->add(curveB);

		UpdateGeometryWithTangent2(directrice,isopen);

	}

}


GraphEditor_ListBezierPath* NurbsEntity::getCurrentPath(float pos)
{
	return m_managerCurvesOpt->getCurves(pos).m_path;
}

const QVector<PointCtrl>&  NurbsEntity::getCurrentCurve(float pos)
{

	return m_managerCurves->getCurves(pos).m_listCtrls;
}

helperqt3d::IsectPlane NurbsEntity::getCurrentPLane(float pos)
{
    return m_managerCurvesOpt->getCurves(pos).m_plane;
}

QVector<QVector3D>  NurbsEntity::getCurrentTangente3D(float pos)
{
	return m_managerCurvesOpt->getCurves(pos).m_tangente3D;
}


QVector<QVector3D>  NurbsEntity::getCurrentPosition3D(float pos)
{
	//qDebug()<<"cross position original "<<m_managerCurves->getCurves(pos).m_cross2D;
	return m_managerCurvesOpt->getCurves(pos).m_positions3D;
}
void NurbsEntity::extrudecurveUpdated(bool finished)
{
   // if (!finished) return;  // use this line if you want refresh only when control point is released after move and not continuosly



    CurveModel& controlpts = *extrudeCurve()->getCurves()[0].get();
    int numctrlpts=controlpts.getSize();
    // qDebug() << "extrudecurveUpdated" << numctrlpts;
    if (numctrlpts<4) return;

    m_extrudeSpline.setOpen(true);
    m_extrudeSpline.setPoints(&controlpts);

    CurveModel samplednurbs;
    m_extrudeSpline.sampletheSpline(&samplednurbs);
    m_extrudeSplineGeometry->updateData(samplednurbs.data());

    //bezier
  //  m_extrudeBezier.setOpen(true);
  //  m_extrudeBezier.setPoints(&controlpts);

 //   float precision = 100.0f;
 //   m_ptsCtrlBezier.clear();
 /*   for(int i=0;i<precision;i++)
    {
    	float eps = (float)i/(float)(precision);
    	m_ptsCtrlBezier.push_back(QVector3D(m_extrudeBezier.getPosition(eps).x ,0.0f,m_extrudeBezier.getPosition(eps).z));
    }
    */


   /* for(int i=0;i<numctrlpts ;i++)
    {
    	m_ptsCtrlBezier.push_back(QVector3D(controlpts.data()[i].x() ,0.0f,controlpts.data()[i].z()));
    }
    qDebug()<<finished<<"extrudecurveUpdated :"<<m_extrudeBezier.getNbCtrlPts();
*/

    recalculateAndUpdateGeometry();
};


void NurbsEntity::addinbetweenXsection(float param, std::shared_ptr<CurveModel> curve)
{
    //qDebug() << "addinbetweenXsection" << param;

    if (m_xsections.size()==0) return;

    XSectionInfo beforeParam = m_xsections.begin()->second;
    XSectionInfo  afterParam = m_xsections.begin()->second;

    if (m_xsections.size()>=2)
    {   // find the two closest xsections to param, i.e. two consecutive xsections where first has parameter<=param and second is larger
        // if param has gone past the first or the last xsection then beforeParam == afterParam == the closest xsection
        for (auto xsect = m_xsections.begin() ;xsect!=m_xsections.end(); xsect++)
        {
            afterParam  = xsect->second;
            if (xsect->second.param>param) break;
            beforeParam = afterParam;
        }
    }

    // qDebug() << "param " << param << " before " << beforeParam.param << "   after " << afterParam.param;
    XSectionInfo result{param, curve.get(), helperqt3d::IsectPlane()};

    result.plane = getXsectionPlane(result.param);
    addUserDrawnCurveandXsection(result.param,  curve, result.plane); // add curve
    getInterpolatedCurve(result,  beforeParam, afterParam);  // fill in values of the curve.
    addCurveandXsectionEnd();
}

void NurbsEntity::deleteXSection()
{
    auto it =  m_xsections.find(m_xSectionParam);
    if (it!=m_xsections.end())
    {
        it->second.curve->emitModeltobeDeleted();
        m_xsections.erase(it);

        recalculateAndUpdateGeometry();
        m_splineGeometry->clearData();
    }
}

void NurbsEntity::insertCurveNurbspoint()
{
    m_nurbscurvelistener->mute();
    for (auto& xsect : m_xsections) // add the knot point and update the control points for all curves
        xsect.second.splinemath.addShapepreservingPoint(m_insertPointParam);

    m_nurbscurvelistener->unmute();
}

void NurbsEntity::nurbscurveUpdated(bool finished, CurveModel* cm)
{
    //if (!finished) return;  // use this line if you want refresh only when control point is released after move and not continuosly

    // Find cm in m_xsections. Wouldn't need to do find_if if CurveModel had pointer to Controller, but I don't want them too tightly coupled
    auto position=find_if(m_xsections.begin(), m_xsections.end(),    // find the curvecontroller that is associated with cm
                            [&] (const auto&  c) {return c.second.curve == cm; } );
    XSectionInfo& xsect =position->second;

    redrawandUpdateNurbsSpline(xsect);
    recalculateAndUpdateGeometry();
}


// move the xsection frame to the curve that was selected
void NurbsEntity::nurbscurveSetSelected(bool selected, CurveModel* cm)
{
    if (!selected) return;
    auto position=find_if(m_xsections.begin(), m_xsections.end(),    // find the curvecontroller that is associated with cm
                            [&] (const auto&  c) {return c.second.curve == cm; } );

    setXsectionPos(position->first);
    redrawandUpdateNurbsSpline(position->second);
}


void NurbsEntity::extrudecurveDeleted()
{
    qDebug() << " extrudecurveDeleted ";
    m_xSectionParam = -1;
};

void NurbsEntity::addUserDrawnCurveandXsection(float param, std::shared_ptr<CurveModel> curve,  helperqt3d::IsectPlane& plane)
{
    if (param ==-1) {qDebug() << "Xsection does not have a position"; return;}
    bool alreadyexists = (m_xsections.find(param)!=m_xsections.end());
    if (alreadyexists)  {qDebug() << "Already curve at this position/xsection"; return;}

  //  m_extrudecurvelistener->mute(); // make it impossible to edit the extrude curve any more

    nurbscurves()->mute();  // unmute when curve is finished drawn
    nurbscurves()->addCurve(curve);
    m_xsections.emplace(param, XSectionInfo{param, curve.get(), plane});
}

// unmute curve and update graphics
void NurbsEntity::addCurveandXsectionEnd()
{
    nurbscurves()->unmute();
    CurveModel* curve=nurbscurves()->getCurves().back().get(); // get the curve that was added
    // find it's xsection object
    auto found=find_if(m_xsections.begin(), m_xsections.end(),    // find the curvecontroller that is associated with cm
                         [&] (const auto&  c) {return c.second.curve == curve; } );

    XSectionInfo& xsect = found->second;
    xsect.splinemath.setOpen(m_isOpenNurbs);
    redrawandUpdateNurbsSpline(xsect);
    recalculateAndUpdateGeometry();

}

void NurbsEntity::redrawandUpdateNurbsSpline(XSectionInfo& xsect)
{
    //  qDebug() << "redrawandUpdateNurbsSpline";
    CurveModel* curve = xsect.curve;
    if (curve->getSize()<4) return;
    // render the spline curve
    CurveModel samplednurbs;
    xsect.splinemath.setPoints(curve);
    xsect.splinemath.sampletheSpline(&samplednurbs);
    m_splineGeometry->updateData(samplednurbs.data());
}

QVector<QVector3D> NurbsEntity::interpolator(float coef,helperqt3d::IsectPlane& planeInterpol)
{
	int indexCurrent = m_managerCurves->getIndexCurve(coef);
	int indexNext = indexCurrent+1;

	if( indexNext >= m_managerCurves->getNbCurves())
	{
		planeInterpol  =m_managerCurves->getCurves(indexCurrent).m_plane;
		return m_managerCurves->getCurvePosition(indexCurrent);
	}


	CurveBezier bezierCurrent = m_managerCurves->getCurves(indexCurrent);
	CurveBezier bezierNext = m_managerCurves->getCurves(indexNext);

	helperqt3d::IsectPlane planeCurrent = bezierCurrent.m_plane;
	helperqt3d::IsectPlane planeNext = bezierNext.m_plane;

	float coefCurrent = (coef - bezierCurrent.m_coef) / ( bezierNext.m_coef -bezierCurrent.m_coef );



	planeInterpol.pointinplane = planeCurrent.pointinplane + coefCurrent *( planeNext.pointinplane - planeCurrent.pointinplane);
	planeInterpol.xaxis = planeCurrent.xaxis + coefCurrent *( planeNext.xaxis - planeCurrent.xaxis);
	planeInterpol.yaxis = planeCurrent.yaxis + coefCurrent *( planeNext.yaxis - planeCurrent.yaxis);

	planeInterpol.xaxis = planeInterpol.xaxis.normalized();
	planeInterpol.yaxis = planeInterpol.yaxis.normalized();

	int nbpts = bezierCurrent.m_positions.count();


	QVector<QVector3D> listePtsRes;
	for(int i = 0;i<nbpts;i++)
	{
		QVector3D P1 = bezierCurrent.m_positions[i];
		QVector3D P2 = bezierNext.m_positions[i];


		QVector3D res = P1 + ( P2-P1)*coefCurrent;
		listePtsRes.push_back( res);
	}


	return listePtsRes;
}



QVector<QVector3D> NurbsEntity::interpolatorOpt(float coef,helperqt3d::IsectPlane& planeInterpol,float precision)
{


	int indexCurrent = m_managerCurvesOpt->getIndexCurve(coef);
	int indexNext = indexCurrent+1;

	if( indexNext >= m_managerCurvesOpt->getNbCurves())
	{
		planeInterpol  =m_managerCurvesOpt->getCurves(indexCurrent).m_plane;
		return m_managerCurvesOpt->getGeneratriceInterpolated(indexCurrent,precision);
	}


	CurveBezierOpt bezierCurrent = m_managerCurvesOpt->getCurves(indexCurrent);
	CurveBezierOpt bezierNext = m_managerCurvesOpt->getCurves(indexNext);

	helperqt3d::IsectPlane planeCurrent = bezierCurrent.m_plane;
	helperqt3d::IsectPlane planeNext = bezierNext.m_plane;

	float coefCurrent = (coef - bezierCurrent.m_coef) / ( bezierNext.m_coef -bezierCurrent.m_coef );



	planeInterpol.pointinplane = planeCurrent.pointinplane + coefCurrent *( planeNext.pointinplane - planeCurrent.pointinplane);
	planeInterpol.xaxis = planeCurrent.xaxis + coefCurrent *( planeNext.xaxis - planeCurrent.xaxis);
	planeInterpol.yaxis = planeCurrent.yaxis + coefCurrent *( planeNext.yaxis - planeCurrent.yaxis);

	planeInterpol.xaxis = planeInterpol.xaxis.normalized();
	planeInterpol.yaxis = planeInterpol.yaxis.normalized();

	int nbpts =precision ; // bezierCurrent.m_positions.count();


	QVector<QVector3D> listePtsRes;

	QVector<QVector3D> listePts1 = m_managerCurvesOpt->getGeneratriceInterpolated(indexCurrent,precision);
	QVector<QVector3D> listePts2 = m_managerCurvesOpt->getGeneratriceInterpolated(indexNext,precision);
	for(int i = 0;i<nbpts;i++)
	{
		QVector3D P1 = listePts1[i];
		QVector3D P2 = listePts2[i];


		QVector3D res = P1 + ( P2-P1)*coefCurrent;
		listePtsRes.push_back( res);
	}


	return listePtsRes;
}

void NurbsEntity::UpdateGeometryWithTangentOpt(bool isopen)
{

	QVector<QVector3D> listeDir;
	QVector<QVector2D> listeLocalPosition;


	m_vertices.clear();
	m_normals.clear();
	m_indices.clear();

	int indexCurve = 0;
	int lastIndexCurve=0;
//	QVector<QVector3D> listepoints = m_managerCurves->getCurves(indexCurve).m_positions;

	float precision  = (float)(m_precision) ;// m_managerCurvesOpt->getPrecision();
//	qDebug()<< " ==>precision :"<<precision;

	int ajout = 0;
	int nbpts = precision;//20
	int nb =precision-1;//19
	if(m_isOpenNurbs==false)
	{
		ajout = 1;
		nb =precision+1;//21

	}

	QVector<QVector3D> directrice = m_managerCurvesOpt->getDirectriceInterpolated(precision);


	int steps = directrice.size() *(nbpts+ajout+1);// listepoints.size();

	int nbVertex = steps *steps;
	int nbindex = (steps-1)*(steps-1 )* 6;

	int nbd = directrice.size();


	m_vertices.reserve(nbVertex);
	m_normals.reserve(nbVertex);
	m_indices.reserve(nbindex);

	QVector3D posDepart =directrice[0];// m_planeCurrent.pointinplane;


	for (int i=0;i<nbd;i++)	//directrice
	{
			QVector3D pos = directrice[i];
			QVector3D vector1InPlane(0.0f,-1.0f,0.0f);
			QVector3D dir1;
			if( i == nbd-1)
			{
				 dir1 =  (-directrice[i-1] + directrice[i]).normalized();
			}
			else
			{
				 dir1 =  (directrice[i+1] - directrice[i]).normalized();
			}

			dir1.setY(0.0f);
			dir1 = dir1.normalized();

			QVector3D vector2InPlane = QVector3D::crossProduct(vector1InPlane,dir1);

			helperqt3d::IsectPlane planeSteps;//{pos,vector1InPlane,vector2InPlane};

			planeSteps.pointinplane = pos;
			planeSteps.xaxis = vector1InPlane;
			planeSteps.yaxis = -vector2InPlane;


			float coefDir = (float)(i/(float)nbd);

			helperqt3d::IsectPlane planeNext;


			QVector<QVector3D> listepoints = interpolatorOpt(coefDir,planeNext,precision);



			if(m_isOpenNurbs==false)
			{
				listepoints.push_back( listepoints[0]);listepoints.push_back( listepoints[0]);

			}

			listeDir.clear();
			listeLocalPosition.clear();

			QVector3D posDepart = directrice[i] ;//planeNext.pointinplane;

		for(int i=0;i<listepoints.count();i++)
			{

				listeDir.push_back(listepoints[i]  -posDepart);

			}

			for(int i=0;i<listepoints.count();i++)
			{
				QVector2D localpos = planeNext.getLocalPosition(posDepart+listeDir[i]);
				listeLocalPosition.push_back(localpos);
			}


			for(int j=0;j<listepoints.count()-1;j++) //generatrice -1
			{
				QVector2D localpos = listeLocalPosition[j];
				QVector3D pos =  planeSteps.getWorldPosition(localpos.x(), localpos.y());// directrice[i]+listeDir[j];



				m_vertices.push_back(pos);



				if ((i<nbd-1) && (j<nb-1))//-1
				{
					int ndx00 = (j+0)+(i+0)*nb;
					int ndx01 = (j+0)+(i+1)*nb;
					int ndx10 = (j+1)+(i+0)*nb;
					int ndx11 = (j+1)+(i+1)*nb;

					m_indices.push_back(ndx00);
					m_indices.push_back(ndx10);
					m_indices.push_back(ndx01);

					m_indices.push_back(ndx10);
					m_indices.push_back(ndx11);
					m_indices.push_back(ndx01);
				}
			}
		}


	 //recompute normals
	    std::vector<QVector3D> normals2;
	    normals2.resize(m_vertices.size(),QVector3D(0,0,0));

		if(!m_wireframeRendering)
		{
			for(int i =0;i<m_indices.size();i+=3)
			{
				QVector3D p1(m_vertices[m_indices[i]].x(),m_vertices[m_indices[i]].y(),m_vertices[m_indices[i]].z());
				QVector3D p2(m_vertices[m_indices[i+1]].x(),m_vertices[m_indices[i+1]].y(),m_vertices[m_indices[i+1]].z());
				QVector3D p3(m_vertices[m_indices[i+2]].x(),m_vertices[m_indices[i+2]].y(),m_vertices[m_indices[i+2]].z());

				 QVector3D v1 = p2 - p1;
				 QVector3D v2 = p2 - p3;

				 QVector3D n1 = QVector3D::crossProduct(v1,v2);
				 n1 = n1.normalized();

				 normals2[m_indices[i]]+= n1;
				 normals2[m_indices[i+1]]+= n1;
				 normals2[m_indices[i+2]]+= n1;

			}

			for(int i =0;i<normals2.size();i++)
			{
				normals2[i] = normals2[i].normalized();
				//m_normals[i] = normals2[i];
				m_normals.push_back(normals2[i]);
			}
		}

	   m_currentMesh->setRenderLines(m_wireframeRendering);
	    m_currentMesh->uploadMeshData(m_vertices, m_indices, m_normals);

}



void NurbsEntity::UpdateGeometryWithTangent2(std::vector<QVector3D> directrice,bool isopen)
{
	QVector<QVector3D> listeDir;
	QVector<QVector2D> listeLocalPosition;


	m_vertices.clear();
	m_normals.clear();
	m_indices.clear();

	int indexCurve = 0;
	int lastIndexCurve=0;
//	QVector<QVector3D> listepoints = m_managerCurves->getCurves(indexCurve).m_positions;

	int ajout = 0;
	int nbpts = 20;
	int nb =19;
	if(m_isOpenNurbs==false)
	{
		ajout = 1;
		nb =21;

	}


	int steps = directrice.size() *(nbpts+ajout+1);// listepoints.size();

	int nbVertex = steps *steps;
	int nbindex = (steps-1)*(steps-1 )* 6;

	int nbd = directrice.size();


	m_vertices.reserve(nbVertex);
	m_normals.reserve(nbVertex);
	m_indices.reserve(nbindex);

	QVector3D posDepart =directrice[0];// m_planeCurrent.pointinplane;


	for (int i=0;i<nbd;i++)	//directrice
	{
			QVector3D pos = directrice[i];
			QVector3D vector1InPlane(0.0f,-1.0f,0.0f);
			QVector3D dir1;
			if( i == nbd-1)
			{
				 dir1 =  (-directrice[i-1] + directrice[i]).normalized();
			}
			else
			{
				 dir1 =  (directrice[i+1] - directrice[i]).normalized();
			}

			dir1.setY(0.0f);
			dir1 = dir1.normalized();

			QVector3D vector2InPlane = QVector3D::crossProduct(vector1InPlane,dir1);

			helperqt3d::IsectPlane planeSteps;//{pos,vector1InPlane,vector2InPlane};

			planeSteps.pointinplane = pos;
			planeSteps.xaxis = vector1InPlane;
			planeSteps.yaxis = -vector2InPlane;


			float coefDir = (float)(i/(float)nbd);

			helperqt3d::IsectPlane planeNext;


			QVector<QVector3D> listepoints = interpolator(coefDir,planeNext);


			if(m_isOpenNurbs==false)
			{
				listepoints.push_back( listepoints[0]);listepoints.push_back( listepoints[0]);

			}

			listeDir.clear();
			listeLocalPosition.clear();

			QVector3D posDepart = directrice[i] ;//planeNext.pointinplane;

		for(int i=0;i<listepoints.count();i++)
			{
				listeDir.push_back(listepoints[i]  -posDepart);

			}

			for(int i=0;i<listepoints.count();i++)
			{
				QVector2D localpos = planeNext.getLocalPosition(posDepart+listeDir[i]);
				listeLocalPosition.push_back(localpos);
			}


			for(int j=0;j<listepoints.count()-1;j++) //generatrice -1
			{
				QVector2D localpos = listeLocalPosition[j];
				QVector3D pos =  planeSteps.getWorldPosition(localpos.x(), localpos.y());// directrice[i]+listeDir[j];



				m_vertices.push_back(pos);



				if ((i<nbd-1) && (j<nb-1))//-1
				{
					int ndx00 = (j+0)+(i+0)*nb;
					int ndx01 = (j+0)+(i+1)*nb;
					int ndx10 = (j+1)+(i+0)*nb;
					int ndx11 = (j+1)+(i+1)*nb;

					m_indices.push_back(ndx00);
					m_indices.push_back(ndx10);
					m_indices.push_back(ndx01);

					m_indices.push_back(ndx10);
					m_indices.push_back(ndx11);
					m_indices.push_back(ndx01);
				}
			}
		}


	 //recompute normals
	    std::vector<QVector3D> normals2;
	    normals2.resize(m_vertices.size(),QVector3D(0,0,0));

		if(!m_wireframeRendering)
		{
			for(int i =0;i<m_indices.size();i+=3)
			{
				QVector3D p1(m_vertices[m_indices[i]].x(),m_vertices[m_indices[i]].y(),m_vertices[m_indices[i]].z());
				QVector3D p2(m_vertices[m_indices[i+1]].x(),m_vertices[m_indices[i+1]].y(),m_vertices[m_indices[i+1]].z());
				QVector3D p3(m_vertices[m_indices[i+2]].x(),m_vertices[m_indices[i+2]].y(),m_vertices[m_indices[i+2]].z());

				 QVector3D v1 = p2 - p1;
				 QVector3D v2 = p2 - p3;

				 QVector3D n1 = QVector3D::crossProduct(v1,v2);
				 n1 = n1.normalized();

				 normals2[m_indices[i]]+= n1;
				 normals2[m_indices[i+1]]+= n1;
				 normals2[m_indices[i+2]]+= n1;

			}

			for(int i =0;i<normals2.size();i++)
			{
				normals2[i] = normals2[i].normalized();
				//m_normals[i] = normals2[i];
				m_normals.push_back(normals2[i]);
			}
		}

	   m_currentMesh->setRenderLines(m_wireframeRendering);
	    m_currentMesh->uploadMeshData(m_vertices, m_indices, m_normals);

}




 void NurbsEntity::UpdateGeometryWithTangent3(std::vector<QVector3D> directrice,bool isopen)
{
	QVector<QVector3D> listeDir;
	QVector<QVector2D> listeLocalPosition;


	m_vertices.clear();
	m_normals.clear();
	m_indices.clear();

	int indexCurve = 0;
	int lastIndexCurve=0;
	QVector<QVector3D> listepoints = m_managerCurves->getCurves(indexCurve).m_positions;


	if(m_isOpenNurbs==false)
	{
		listepoints.push_back( listepoints[0]);
		listepoints.push_back( listepoints[0]);// TODO provisoire, necessaire pour fermer la directrice probleme ailleurs!!
	}


	int steps = directrice.size() * listepoints.size();

	int nbVertex = steps *steps;
	int nbindex = (steps-1)*(steps-1 )* 6;

	int nb =listepoints.size()-1;//
	int nbd = directrice.size()-1;




	m_vertices.reserve(nbVertex);
	m_normals.reserve(nbVertex);
	m_indices.reserve(nbindex);

	QVector3D posDepart =directrice[0];// m_planeCurrent.pointinplane;

	for(int i=0;i<listepoints.size();i++)
	{
		listeDir.push_back(listepoints[i]  - posDepart);
	}


	for(int i=0;i<listepoints.size();i++)
	{
		QVector2D localpos = m_planeCurrent.getLocalPosition(posDepart+listeDir[i]);
		listeLocalPosition.push_back(localpos);
	}


	for (int i=0;i<nbd;i++)	//directrice
	{
			QVector3D pos = directrice[i];
			QVector3D vector1InPlane(0.0f,-1.0f,0.0f);
			QVector3D dir1 =  (directrice[i+1] - directrice[i]).normalized();
			QVector3D vector2InPlane = QVector3D::crossProduct(dir1 ,vector1InPlane);

			helperqt3d::IsectPlane planeNext{pos,vector1InPlane,vector2InPlane};


			float coefDir = (float)(i/(float)nbd);



			//QVector<QVector3D> listepoints = interpolator(coefDir);

			indexCurve = m_managerCurves->getIndexCurve(coefDir);

			if(indexCurve != lastIndexCurve)
			{

				QVector<QVector3D> listepoints = m_managerCurves->getCurves(indexCurve).m_positions;
				if(m_isOpenNurbs==false)
				{
					listepoints.push_back( listepoints[0]);listepoints.push_back( listepoints[0]);
				}

				listeDir.clear();
				listeLocalPosition.clear();

				 posDepart = directrice[i] ;//planeNext.pointinplane;

				for(int i=0;i<listepoints.count();i++)
				{
					listeDir.push_back(listepoints[i]  -posDepart);

				}

				for(int i=0;i<listepoints.count();i++)
				{
					QVector2D localpos = m_planeCurrent.getLocalPosition(posDepart+listeDir[i]);
					listeLocalPosition.push_back(localpos);
				}


				lastIndexCurve = indexCurve;
			}

			for(int j=0;j<listepoints.count()-1;j++) //generatrice -1
			{
				QVector2D localpos = listeLocalPosition[j];
				QVector3D pos =  planeNext.getWorldPosition(localpos.x(), localpos.y());// directrice[i]+listeDir[j];


				m_vertices.push_back(pos);



				if ((i<nbd-1) && (j<nb-1))//-1
				{
					int ndx00 = (j+0)+(i+0)*nb;
					int ndx01 = (j+0)+(i+1)*nb;
					int ndx10 = (j+1)+(i+0)*nb;
					int ndx11 = (j+1)+(i+1)*nb;

					m_indices.push_back(ndx00);
					m_indices.push_back(ndx10);
					m_indices.push_back(ndx01);

					m_indices.push_back(ndx10);
					m_indices.push_back(ndx11);
					m_indices.push_back(ndx01);
				}
			}
		}


	 //recompute normals
	    std::vector<QVector3D> normals2;
	    normals2.resize(m_vertices.size(),QVector3D(0,0,0));

		if(!m_wireframeRendering)
		{
			for(int i =0;i<m_indices.size();i+=3)
			{
				QVector3D p1(m_vertices[m_indices[i]].x(),m_vertices[m_indices[i]].y(),m_vertices[m_indices[i]].z());
				QVector3D p2(m_vertices[m_indices[i+1]].x(),m_vertices[m_indices[i+1]].y(),m_vertices[m_indices[i+1]].z());
				QVector3D p3(m_vertices[m_indices[i+2]].x(),m_vertices[m_indices[i+2]].y(),m_vertices[m_indices[i+2]].z());

				 QVector3D v1 = p2 - p1;
				 QVector3D v2 = p2 - p3;

				 QVector3D n1 = QVector3D::crossProduct(v1,v2);
				 n1 = n1.normalized();

				 normals2[m_indices[i]]+= n1;
				 normals2[m_indices[i+1]]+= n1;
				 normals2[m_indices[i+2]]+= n1;

			}

			for(int i =0;i<normals2.size();i++)
			{
				normals2[i] = normals2[i].normalized();
				//m_normals[i] = normals2[i];
				m_normals.push_back(normals2[i]);
			}
		}

	   m_currentMesh->setRenderLines(m_wireframeRendering);
	    m_currentMesh->uploadMeshData(m_vertices, m_indices, m_normals);

}



void NurbsEntity::UpdateGeometryWithTangent(std::vector<QVector3D> directrice, QVector<QVector3D> listepoints, int index,bool isopen)
{
	QVector<QVector3D> listeDir;
	QVector<QVector2D> listeLocalPosition;


	m_vertices.clear();
	m_normals.clear();
	m_indices.clear();


	if(m_isOpenNurbs==false)listepoints.push_back( listepoints[0]);


	int steps = directrice.size() * listepoints.size();

	int nbVertex = steps *steps;
	int nbindex = (steps-1)*(steps-1 )* 6;

	int nb = listepoints.size()-1;//
	int nbd = directrice.size()-1;

	m_vertices.reserve(nbVertex);
	m_normals.reserve(nbVertex);
	m_indices.reserve(nbindex);

	QVector3D posDepart = m_planeCurrent.pointinplane;


	for(int i=0;i<listepoints.size();i++)
	{
		listeDir.push_back(listepoints[i]  - posDepart);
	}

	QVector3D pos = directrice[0];
	QVector3D vector1InPlane(0.0f,-1.0f,0.0f);
	QVector3D vector2InPlane = listeDir[0].normalized();



	for(int i=0;i<listepoints.size();i++)
	{
		QVector2D localpos = m_planeCurrent.getLocalPosition(posDepart+listeDir[i]);
		listeLocalPosition.push_back(localpos);
	}


	for (int i=0;i<nbd;i++)	//directrice
	{
			QVector3D pos = directrice[i];
			QVector3D vector1InPlane(0.0f,-1.0f,0.0f);
			QVector3D dir1 =  (directrice[i+1] - directrice[i]).normalized();
			QVector3D vector2InPlane = QVector3D::crossProduct(dir1 ,vector1InPlane);

			helperqt3d::IsectPlane planeNext{pos,vector1InPlane,vector2InPlane};


			for(int j=0;j<nb;j++) //generatrice
			{
				QVector2D localpos = listeLocalPosition[j];
				QVector3D pos =  planeNext.getWorldPosition(localpos.x(), localpos.y());// directrice[i]+listeDir[j];


				m_vertices.push_back(pos);



				if ((i<nbd-1) && (j<nb-1))
				{
					int ndx00 = (j+0)+(i+0)*nb;
					int ndx01 = (j+0)+(i+1)*nb;
					int ndx10 = (j+1)+(i+0)*nb;
					int ndx11 = (j+1)+(i+1)*nb;

					m_indices.push_back(ndx00);
					m_indices.push_back(ndx10);
					m_indices.push_back(ndx01);

					m_indices.push_back(ndx10);
					m_indices.push_back(ndx11);
					m_indices.push_back(ndx01);
				}
			}
		}


	 //recompute normals
	    std::vector<QVector3D> normals2;
	    normals2.resize(m_vertices.size(),QVector3D(0,0,0));

		if(!m_wireframeRendering)
		{
			for(int i =0;i<m_indices.size();i+=3)
			{
				QVector3D p1(m_vertices[m_indices[i]].x(),m_vertices[m_indices[i]].y(),m_vertices[m_indices[i]].z());
				QVector3D p2(m_vertices[m_indices[i+1]].x(),m_vertices[m_indices[i+1]].y(),m_vertices[m_indices[i+1]].z());
				QVector3D p3(m_vertices[m_indices[i+2]].x(),m_vertices[m_indices[i+2]].y(),m_vertices[m_indices[i+2]].z());

				 QVector3D v1 = p2 - p1;
				 QVector3D v2 = p2 - p3;

				 QVector3D n1 = QVector3D::crossProduct(v1,v2);
				 n1 = n1.normalized();

				 normals2[m_indices[i]]+= n1;
				 normals2[m_indices[i+1]]+= n1;
				 normals2[m_indices[i+2]]+= n1;

			}

			for(int i =0;i<normals2.size();i++)
			{
				normals2[i] = normals2[i].normalized();
				//m_normals[i] = normals2[i];
				m_normals.push_back(normals2[i]);
			}
		}

	   m_currentMesh->setRenderLines(m_wireframeRendering);
	    m_currentMesh->uploadMeshData(m_vertices, m_indices, m_normals);


}

// Based on the user defined cross sections, this function creates interpolated inbetween cross sections
// and then uses all the cross section control points for generating a nurbs. Finally, the nurbs is sampled and rendered
void NurbsEntity::recalculateAndUpdateGeometry()
{
    if (m_xsections.size()==0)
    {
        m_currentMesh->setEmptyData();
        m_pointsGeometry->clearData();
        return;  // nothing to calculate
    }

    std::vector< CurveModel > curves;
    completeTheCurves(curves,m_numinbetweens);


    if (curves.size()<4) return;
    bool wireframe = m_wireframeRendering;

    tinynurbs::Surface3f m_nurbssurf;
    const int degree=3;
    m_nurbssurf.degree_u = degree;  // only degree 3 is supported
    m_nurbssurf.degree_v = degree;  // only degree 3 is supported


    int szU = int(curves.size());

    const int numcurvepoints =  (int)(curves[0].getSize());

    NurbsHelper uVals(szU, NurbsHelper::Type::Clamped);

    const int ptsV=numcurvepoints;


    // adding knots to nurbs //////////////////////////
    if (curves.size()<szU) return;

    for (int i=1; i<=uVals.getNumKnots(); i++)
    	m_nurbssurf.knots_u.push_back(uVals.getKnot(i));

    // all curves have the same knots (but different control points), so just get the knots from the first user defined xsection
    const std::vector<float>& knots = m_xsections.begin()->second.splinemath.getTinycurve().knots;
    for (int i = 0;i<knots.size();i++)
    	m_nurbssurf.knots_v.push_back(knots[i]);

    CurveModel renderControlpoints;


    // adding control points to nurbs //////////////////////////
    std::vector<glm::vec3> ptsArray;
    for (int i=0;i<szU;i++)
    {
        for (int j=0;j<ptsV;j++)
        {
            CurveModel& cm = curves[i];
            QVector3D p = cm.getPosition(j);
            ptsArray.push_back(glm::vec3(p[0],p[1],p[2]));
            renderControlpoints.insertBack(p);
        }
        if (!m_isOpenNurbs)
        {
            CurveModel& cm = curves[i];

            if( &cm == nullptr)
            {
            	qDebug()<<" CurveModel est null";

            }
            else
            {
            	QVector3D p = cm.getPosition(0);
            	ptsArray.push_back(glm::vec3(p[0],p[1],p[2]));
            }
        }
    }

    int szV = ptsV;
    if (!m_isOpenNurbs) szV += 1;

    m_nurbssurf.control_points = {(size_t)szU, (size_t)szV, ptsArray};

    bool isvalid = tinynurbs::surfaceIsValid(m_nurbssurf);
    if (!isvalid)
    {qDebug()<< "nurbs surface not valid!"; return;}

    //        evaluating nurbs to create geometry    //////////////////////////
    NurbsHelper   openuVals(ptsV,   NurbsHelper::Type::Clamped);
    NurbsHelper closeduVals(ptsV+1, NurbsHelper::Type::Clamped);
    NurbsHelper vVals=m_isOpenNurbs?openuVals:closeduVals;

  //  QVector<QVector3D> normals3D;
    std::vector<QVector3D> vertices,normals;
    std::vector<int> indices;

    m_vertices.clear();
    m_normals.clear();
    m_indices.clear();

    int steps = m_triangulateResolution;

    int nbVertex = steps *steps;
    int nbindex = (steps-1)*(steps-1 )* 6;

    m_vertices.reserve(nbVertex);
    m_normals.reserve(nbVertex);
    m_indices.reserve(nbindex);

    for (int i=0;i<steps;i++)
        for (int j=0;j<steps;j++)
        {
            float u = uVals.getParameter(i,steps);
            float v = vVals.getParameter(j,steps);

            glm::vec3 pt   = tinynurbs::surfacePoint (m_nurbssurf, u, v);
            glm::vec3 norm = tinynurbs::surfaceNormal(m_nurbssurf, u, v);

            vertices.push_back(QVector3D(pt.x,pt.y,pt.z));
            m_vertices.push_back(QVector3D(pt.x,pt.y,pt.z));

          //  if(norm.y> 0.0f)
           /// 	norm = -norm;

            normals.push_back(QVector3D(norm.x,norm.y,norm.z));
            //normals.push_back(QVector3D(0.0f,-1.0,0.0));
           // normals3D.push_back(QVector3D(norm.x,norm.y,norm.z));


         //   colors.push_back(QVector3D());
            //       indices.push_back(vertices.size()-1);
            // for each center between 4 corners (therefore exclude last i and j index)
            //   - make two triangles
            if ((i<steps-1) && (j<steps-1)) // exclude last i and j index
            {
                int ndx00 = (i+0)+(j+0)*steps;
                int ndx01 = (i+0)+(j+1)*steps;
                int ndx10 = (i+1)+(j+0)*steps;
                int ndx11 = (i+1)+(j+1)*steps;

                //qDebug()<<ndx00<<"|"<<ndx01<<"|"<<ndx10<<"|"<<ndx11;

                indices.push_back(ndx00);
                indices.push_back(ndx10);
                indices.push_back(ndx01);

                indices.push_back(ndx10);
                indices.push_back(ndx11);
                indices.push_back(ndx01);

                m_indices.push_back(ndx00);
				m_indices.push_back(ndx10);
				m_indices.push_back(ndx01);

				m_indices.push_back(ndx10);
				m_indices.push_back(ndx11);
				m_indices.push_back(ndx01);

                if (wireframe)
                {
                    indices.push_back(ndx00);
                    indices.push_back(ndx10);

                    indices.push_back(ndx10);
                    indices.push_back(ndx11);

                    indices.push_back(ndx11);
                    indices.push_back(ndx01);

                    indices.push_back(ndx01);
                    indices.push_back(ndx00);
                }
            }
        }


    //recompute normals
    std::vector<QVector3D> normals2;
    normals2.resize(vertices.size(),QVector3D(0,0,0));

	if(!wireframe)
	{
		for(int i =0;i<indices.size();i+=3)
		{
			QVector3D p1(vertices[indices[i]].x(),vertices[indices[i]].y(),vertices[indices[i]].z());
			QVector3D p2(vertices[indices[i+1]].x(),vertices[indices[i+1]].y(),vertices[indices[i+1]].z());
			QVector3D p3(vertices[indices[i+2]].x(),vertices[indices[i+2]].y(),vertices[indices[i+2]].z());

			 QVector3D v1 = p2 - p1;
			 QVector3D v2 = p2 - p3;

			 QVector3D n1 = QVector3D::crossProduct(v1,v2);
			 n1 = n1.normalized();

			 normals2[indices[i]]+= n1;
			 normals2[indices[i+1]]+= n1;
			 normals2[indices[i+2]]+= n1;

		}

		for(int i =0;i<normals2.size();i++)
		{
			normals2[i] = normals2[i].normalized();
			//m_normals[i] = normals2[i];
			m_normals.push_back(normals2[i]);
		}
	}
    m_currentMesh->setRenderLines(wireframe);
    m_currentMesh->uploadMeshData(vertices, indices, normals2);

    if (m_showNURBSPoints)
        m_pointsGeometry->updateData(renderControlpoints.data()) ;
    else
        m_pointsGeometry->clearData();
}


// Based on the m_analyticspline, this function takes a curve defined on xsect and copies it into xsectTo based on xsectTo.param
void NurbsEntity::copyXsectionToOtherDestionationOnSpline(const XSectionInfo& xsectFrom, XSectionInfo& xsectTo )
{
    CurveModel* xsectcurve = xsectFrom.curve;

    xsectTo.curve->clear();
    xsectTo.plane = getXsectionPlane(xsectTo.param);
    for (int j=0; j<int(xsectcurve->getSize()); j++)
    {
        QVector3D p  = xsectcurve->getPosition(j);                    // position in world space
        QVector2D pp = xsectFrom.plane.getLocalPosition(p);             // position moved to local space of xsection at xsect.param

        QVector3D newFrame = xsectTo.plane.getWorldPosition( pp.x(), pp.y());  // position moved to local space of xsection at toparam
        xsectTo.curve->insertBack(newFrame);                       // position in world space
    }
}


// creates <numsections> xsections inbetween each userdefined xsection along the spline curve by interpolating the user defined ones.
void NurbsEntity::completeTheCurves(std::vector< CurveModel >& outcurves, int numsections)
{
    if (m_xsections.size()==0) return;  // nothing to calculate

    std::vector<XSectionInfo*> vecXsections;

    CurveModel cm1,cm2;
    XSectionInfo xstart0{0,  &cm1, helperqt3d::IsectPlane()};
    XSectionInfo   xend1{1,  &cm2, helperqt3d::IsectPlane()};

    XSectionInfo firstUserdrawnxsect = m_xsections.begin()->second;
    if (firstUserdrawnxsect.param!=0)  // If an xsection curve does not exist at the start of the spline at param=0 then make one
    {
        copyXsectionToOtherDestionationOnSpline(firstUserdrawnxsect, xstart0);
        vecXsections.push_back(&xstart0);
    }

    for (auto xsect = m_xsections.begin() ;xsect!=m_xsections.end(); xsect++)
        vecXsections.push_back(&xsect->second);

    XSectionInfo lastUserdrawnxsect  = m_xsections.rbegin()->second;
    if (lastUserdrawnxsect.param!=1)  // If an xsection curve does not exist at the end of the spline at param=1 then make one
    {
        copyXsectionToOtherDestionationOnSpline(lastUserdrawnxsect, xend1);
        vecXsections.push_back(&xend1);
    }

    // now vecXsections contains a curve at the start and end of the spline (either userdefined or automatically made) as well as the userdefined curves in the middle

    int numxsections = vecXsections.size();
    for (int i = 0; i<numxsections-1; i++)
    {
        XSectionInfo* beforestart = vecXsections[std::max(i-1,0)];
        XSectionInfo* start       = vecXsections[i];
        XSectionInfo* end         = vecXsections[i+1];
        XSectionInfo* afterend    = vecXsections[std::min(i+2,numxsections-1)];

        makeinbetweenXsections(outcurves, numsections, *beforestart,  *start, *end, *afterend);
    }
}


void NurbsEntity::makeinbetweenXsections(std::vector< CurveModel >& outcurves, int numnew, const XSectionInfo& beforestart, const XSectionInfo& start, const XSectionInfo& end, const XSectionInfo& afterend)
{
    //   qDebug() << "make inbetween " << start.param << end.param;

    float step = 1.0f/float(numnew-1);
    for (int index = 0; index < numnew; index++)
    {
        float norm = step*index; // goes from 0 to 1 in <numnew> steps
        float inbetweenparam =  (1-norm)*start.param + norm*end.param;

        CurveModel curve;
        XSectionInfo result{inbetweenparam, &curve,helperqt3d::IsectPlane()};
        getSmoothInterpolatedCurve(result, beforestart, start, end, afterend);
        outcurves.push_back(*result.curve);
    }
}

// based on the curve at xsection start and xsection end, an interpolated cross section silhuette (curve) in result.curve
// is created based at the curve parameter result.param
void NurbsEntity::getSmoothInterpolatedCurve(XSectionInfo& result, const XSectionInfo& beforestart, const XSectionInfo& start, const XSectionInfo& end, const XSectionInfo& afterend)
{

    if (start.curve->getSize()!=end.curve->getSize())
    {
    	//qDebug() <<start.curve->getSize()<< " -Curves must be of equal size - "<<end.curve->getSize();
    	return;
    }

    result.curve->clear();
    float param = result.param;

    float len = end.param-start.param;
    float weight = (param-start.param)/len;

    // qDebug() <<  "from : " << start.param << "to : " << end.param << "  weight: " << weight << "parameter :" << param ;

    // for each curvepoint p, interpolate it in local space(s)

    for (unsigned int index = 0; index < start.curve->getSize(); index++)
    {
        QVector3D p0 = beforestart.curve->getPosition(index);
        QVector3D p1 =       start.curve->getPosition(index);
        QVector3D p2 =         end.curve->getPosition(index);
        QVector3D p3 =    afterend.curve->getPosition(index);

        QVector2D p0local = beforestart.plane.getLocalPosition(p0);
        QVector2D p1local =       start.plane.getLocalPosition(p1);
        QVector2D p2local =         end.plane.getLocalPosition(p2);
        QVector2D p3local =    afterend.plane.getLocalPosition(p3);

        QVector2D pinterplocal;

        if (m_linearInterpolate)
        {
            pinterplocal = p1local*(1-weight) +p2local*weight;
            result.plane = getXsectionPlane(param);
        }
        else
        {
            // For this interpolation, the z value is along the extrude spline and goes from 0 to 1
            // To get a correct interpolation, the distances along z must be correct relative to the distances for x and y.
            // Therefore, we multiply with the length of the extrude spline.
            float length = m_extrudeSpline.getLength();  // also multiply with user defined value to play with result

            glm::vec3 interp = CentripetalCatmullRom::catmull2(
                glm::vec3(p0local.x(),p0local.y(),beforestart.param*length),
                glm::vec3(p1local.x(),p1local.y(),start.param*length),
                glm::vec3(p2local.x(),p2local.y(),end.param*length),
                glm::vec3(p3local.x(),p3local.y(),afterend.param*length),
                weight,m_inbetweenxsectionalpha);

            pinterplocal = QVector2D(interp.x,interp.y);
            result.param = interp.z/length;
            result.plane = getXsectionPlane(interp.z/length);
        }

        // put pinterp into world frame
        QVector3D pinterpworld = result.plane.getWorldPosition(pinterplocal.x(), pinterplocal.y());
        result.curve->insertBack(QVector3D(pinterpworld.x(), pinterpworld.y(), pinterpworld.z()));
    }
}


// based on the curve at xsection start and xsection end, a linearly interpolated curve in result.curve is created based on result.param
void NurbsEntity::getInterpolatedCurve(XSectionInfo& result, const XSectionInfo& start, const XSectionInfo& end)
{
    const XSectionInfo& interpfrom = start;
    const XSectionInfo& interpto   = end;

    if (interpfrom.curve->getSize()!=interpto.curve->getSize())
    {qDebug() << "2-Curves must be of equal size"; return;}

    result.curve->clear();
    result.plane = getXsectionPlane(result.param);

    float len = end.param-start.param;
    if (len==0) len=1; // handle the case where interpfrom==interpto, i.e. when there is only one xsection defined.
    float weight = (result.param-start.param)/len;
    // qDebug() << "parameter :" << result.param <<  "from : " << interpfrom.param << "to : " << interpto.param << "  weight: " << weight;

    // for each curvepoint p, interpolate it in local space(s)

    for (unsigned int index = 0; index < interpfrom.curve->getSize(); index++)
    {
        QVector3D pfrom = interpfrom.curve->getPosition(index);
        QVector3D pto   =   interpto.curve->getPosition(index);

        QVector2D  pfromlocal = interpfrom.plane.getLocalPosition(pfrom);
        QVector2D  ptolocal   =   interpto.plane.getLocalPosition(pto);

        QVector2D pinterplocal = pfromlocal*(1-weight) +ptolocal*weight;

        // put pinterp into world frame
        QVector3D pinterpworld = result.plane.getWorldPosition(pinterplocal.x(), pinterplocal.y());

        result.curve->insertBack(QVector3D(pinterpworld.x(), pinterpworld.y(), pinterpworld.z()));
    }
}


void NurbsEntity::exportObj(QString path)
{

	Qt3DHelpers::writeObj(path.toStdString().c_str(),m_vertices,m_normals, m_indices);

}


void NurbsEntity::setPrecision(int value)
{
	if(m_precision != value)
	{
		m_precision  = value;

		if(m_managerCurvesOpt->getDirectriceOk())
		{

			QVector<QVector3D> dir= m_managerCurvesOpt->getDirectriceInterpolated(getPrecision());

			std::vector<QVector3D> points2;
			for(int i=0;i<getPrecision();i++)
			{
				points2.push_back(dir[i]);
			}

			m_extrudeSplineGeometry->updateData(points2);
		}


		if(m_managerCurvesOpt->getNbCurves()> 0)
		{
			UpdateGeometryWithTangentOpt(m_isOpenNurbs);
		}
	}
}

int NurbsEntity::getPrecision()
{
	return m_precision;
}



/*
void NurbsEntity::saveNurbs(QString path, std::vector<QVector3D> directrice)
{
	QFile file(path);
	if(!file.open(QIODevice::WriteOnly | QIODevice::Text))
	{
		qDebug()<<" ouverture du fichier impossible "<<path;
		return;
	}
	QTextStream out(&file);

	out<<getColorNurbs().red()<<"-"<<getColorNurbs().green()<<"-"<<getColorNurbs().blue()
				<<"|"<<getColorDirectrice().red()<<"-"<<getColorDirectrice().green()<<"-"<<getColorDirectrice().blue()<<"\n";


	out<<"directrice\n";
	for(int i=0;i<directrice.size();i++)
	{
		out<<directrice[i].x()<<"|"<<directrice[i].y()<<"|"<<directrice[i].z();
		out<<"\n";
	}

	out<<"\n";

	for(int i=0;i<m_managerCurves->getNbCurves();i++)
	{
		CurveBezier curve = m_managerCurves->getCurves(i);

		out<<i<<"  positions\n";
		for(int j =0;j<curve.m_positions3D.size();j++)
		{
			out<<curve.m_positions3D[j].x()<<"|"<<curve.m_positions3D[j].y()<<"|"<<curve.m_positions3D[j].z();
			out<<"\n";
		}
		out<<"\n";
		out<<"tangentes\n";
		for(int j =0;j<curve.m_tangente3D.size();j++)
		{
			out<<curve.m_tangente3D[j].x()<<"|"<<curve.m_tangente3D[j].y()<<"|"<<curve.m_tangente3D[j].z();
			out<<"\n";
		}
		out<<"\n";
	}

	file.close();


}
*/


