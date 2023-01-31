#pragma once

#include <QObject>
#include <QEntity>
#include <QVector3D>
#include <QCamera>
#include <QColor>
#include <tinynurbs/tinynurbs.h>
#include "curvemodel.h"
#include "helperqt3d.h"

#include "splinemath.h"

#include "PointCtrl.h"
#include "GraphEditor_ListBezierPath.h"
#include "randomlineview.h"
#include "randomTransformation.h"


class PointsLinesGeometry;
class MeshGeometry;
class MeshGeometry2;

class CurveListener;
namespace Qt3DRender
{
    class QMaterial;
    class QGeometry;
}


class CurveBezierOpt
{
public:
	float m_coef;
	GraphEditor_ListBezierPath* m_path;
	helperqt3d::IsectPlane m_plane;
	//QVector<PointCtrl> m_listCtrls;
	QVector<QVector3D > m_positions3D;  //Positions des points dans le repere3D
	QVector<QVector3D > m_tangente3D;
	RandomTransformation* m_randomTransfo;

	QColor m_color;



	CurveBezierOpt()
	{
		m_coef = -1.0f;
		m_path = nullptr;
	}
	CurveBezierOpt(GraphEditor_ListBezierPath* path )
	{
		m_coef = -1.0f;
		m_path = path;
	}

	CurveBezierOpt(float coef,GraphEditor_ListBezierPath* path, helperqt3d::IsectPlane plane,QVector<QVector3D > positions3D,QVector<QVector3D > tangentes3D,RandomTransformation* transfo ,bool cloner)
	{
		m_coef = coef;

		if(path !=nullptr)
		{
			m_path =  path->clone();
			m_path->setEnabled(cloner);
		}
		else m_path = nullptr;
		m_plane = plane;
		m_positions3D = positions3D;
		m_tangente3D = tangentes3D;
		m_randomTransfo = transfo;
	}

	void setColor(QColor col)
	{
		if(m_path != nullptr)
			m_path->setColor(col);
		else
			qDebug()<<"CurveBezierOpt path est NULLLLL";
	}
};

class CurveBezier
{
public :
		float m_coef;
		QVector<QVector3D > m_positions;  //points de controles
		QVector<PointCtrl> m_listCtrls;		// points de controles en 2D
		helperqt3d::IsectPlane m_plane;

		QVector<QVector3D > m_positions3D;  //Positions des points dans le repere3D
		QVector<QVector3D > m_tangente3D;	//Positions de tangentes dans le repere3D

		QPointF m_cross2D;
		QVector3D m_cross3D;




	/*	CurveBezier(float coef , QVector<QVector3D> positions)
		{
			m_coef = coef;
			m_positions = positions;
		}*/

		CurveBezier()
		{
			m_coef=-1.0f;


		}
		CurveBezier(float coef , QVector<PointCtrl> positions)
		{
			m_coef = coef;
			m_listCtrls = positions;

		}
		CurveBezier(float coef , QVector<PointCtrl> positions,QVector<QVector3D> pts)
		{
			m_coef = coef;
			m_listCtrls = positions;
			m_positions = pts;
		}

		CurveBezier(float coef , QVector<PointCtrl> positions,QVector<QVector3D> pts, QPointF crosspos)
		{
			m_coef = coef;
			m_listCtrls = positions;
			m_positions = pts;
			m_cross2D  =crosspos;
		}

		CurveBezier(float coef , QVector<PointCtrl> positions,QVector<QVector3D> pts, QPointF crosspos,helperqt3d::IsectPlane plane,QVector<QVector3D > pos3D,QVector<QVector3D > tan3D,QVector3D cross3d)
		{
			m_coef = coef;
			m_listCtrls = positions;
			m_positions = pts;
			m_cross2D  =crosspos;
			m_plane = plane;
			m_positions3D = pos3D;
			m_tangente3D = tan3D;
			m_cross3D = cross3d;
		}
};



class CurveBezierManager
{
private:

	QVector<CurveBezier> m_listeCurves;
public:
	CurveBezierManager()
	{

	}

	int  exist(float coef)
	{
		for(int i =0;i<m_listeCurves.count();i++)
		{
			float eps = 0.01f;

			if(m_listeCurves[i].m_coef > coef- eps && m_listeCurves[i].m_coef < coef+ eps)
			{
				return i;
			}

		}
		return -1;
	}

	int getIndexCurve(float coef)
	{

		for(int i =m_listeCurves.count()-1;i>=0;i--)
		{
			if( coef > m_listeCurves[i].m_coef )
			{

				return i;

			}
		}
		//if( index >= 0)
		//	return index;

		return 0;
	}

	const CurveBezier& getCurves(float coef)
	{
		for(int i =m_listeCurves.count()-1;i>=0;i--)
		{
			if(coef > m_listeCurves[i].m_coef )
			{
				return m_listeCurves[i];
			}
		}
		return m_listeCurves[0];
	}

	QVector<QVector3D > getCurvePosition(int index)
	{
		//if(index >=0 && index <m_listeCurves.count())
		return m_listeCurves[index].m_positions;


	}

	/*QVector<QVector3D > getCurvePosition3D(int index)
		{
			//if(index >=0 && index <m_listeCurves.count())
			return m_listeCurves[index].m_positions3D;


		}*/

	const CurveBezier& getCurves(int index)
	{
		static CurveBezier errorRet;

		if(index >=0 && index <m_listeCurves.count())
			return m_listeCurves[index];

		qDebug()<<"CurveBezier ERRREUR ";
		return errorRet;
	}

	void add(const CurveBezier& curve)
	{
		if(m_listeCurves.count()==0)
		{
			m_listeCurves.push_back(curve);
			return;
		}
		else
		{
			for(int i =0;i<m_listeCurves.count();i++)
			{
				if(m_listeCurves[i].m_coef > curve.m_coef)
				{

					m_listeCurves.insert(i,curve);
					return;
				}
			}
		}
		m_listeCurves.push_back(curve);


	}

	int getNbCurves()
	{
		return m_listeCurves.count();
	}

	void supprimer(int index)
	{
		m_listeCurves.removeAt(index);
	}

	void replace(int index ,CurveBezier curve )
	{
		m_listeCurves.replace(index, curve);
	}

	void show()
	{
		for(int i =0;i<m_listeCurves.count();i++)
		{
			qDebug()<<i<<" , coef :"<<m_listeCurves[i].m_coef;
			for(int j=0;j< m_listeCurves[i].m_listCtrls.count();j++)
				qDebug()<<"   ==>"<<j <<" ,position :"<<m_listeCurves[i].m_listCtrls[j].m_position;
		}
	}

};



class CurveBezierOptManager
{
private:

	QVector<CurveBezierOpt> m_listeCurves;
	CurveBezierOpt m_directriceCurve;
	QMatrix4x4 m_transformScene;
	IsoSurfaceBuffer m_bufferIso;
	//RandomLineView* m_ortho = nullptr;
	bool m_directriceOk= false;

public:
	CurveBezierOptManager()
	{
		m_directriceOk= false;
		//m_ortho = nullptr;
	}

	void setDirectrice(CurveBezierOpt directrice,QMatrix4x4 transformScene,IsoSurfaceBuffer bufferIso)
	{
		m_directriceCurve = directrice;
		m_transformScene  = transformScene;
		m_bufferIso = bufferIso;
		m_directriceOk = true;

	}

	bool getDirectriceOk(){ return m_directriceOk;}

/*	void setOrtho(RandomLineView* view)
	{
		m_ortho = view;

	}*/

	int  exist(float coef)
	{
		for(int i =0;i<m_listeCurves.count();i++)
		{
			float eps = 0.01f;

			if(m_listeCurves[i].m_coef > coef- eps && m_listeCurves[i].m_coef < coef+ eps)
			{
				return i;
			}

		}
		return -1;
	}

	int getIndexCurve(float coef)
	{

		for(int i =m_listeCurves.count()-1;i>=0;i--)
		{
			if( coef > m_listeCurves[i].m_coef )
			{

				return i;

			}
		}
		//if( index >= 0)
		//	return index;

		return 0;
	}

	const CurveBezierOpt& getCurves(float coef)
	{
		for(int i =m_listeCurves.count()-1;i>=0;i--)
		{
			if(coef > m_listeCurves[i].m_coef )
			{
				return m_listeCurves[i];
			}
		}
		return m_listeCurves[0];
	}

	/*QVector<QVector3D > getCurvePosition(int index)
	{
		//if(index >=0 && index <m_listeCurves.count())
		return m_listeCurves[index].m_positions;


	}*/



	const CurveBezierOpt& getCurves(int index)
	{
		static CurveBezierOpt errorRet;

		if(index >=0 && index <m_listeCurves.count())
			return m_listeCurves[index];

		qDebug()<<"CurveBezier ERRREUR ";
		return errorRet;
	}

	void add(const CurveBezierOpt& curve)
	{
		if(m_listeCurves.count()==0)
		{
			m_listeCurves.push_back(curve);
			return;
		}
		else
		{
			for(int i =0;i<m_listeCurves.count();i++)
			{
				if(m_listeCurves[i].m_coef > curve.m_coef)
				{

					m_listeCurves.insert(i,curve);
					return;
				}
			}
		}
		m_listeCurves.push_back(curve);


	}

	int getNbCurves()
	{
		return m_listeCurves.count();
	}

	void supprimer(int index)
	{
		m_listeCurves.removeAt(index);
	}

	void replace(int index ,CurveBezierOpt curve )
	{
		m_listeCurves.replace(index, curve);
	}

	void show()
	{
		/*for(int i =0;i<m_listeCurves.count();i++)
		{
			qDebug()<<i<<" , coef :"<<m_listeCurves[i].m_coef;
			for(int j=0;j< m_listeCurves[i].m_listCtrls.count();j++)
				qDebug()<<"   ==>"<<j <<" ,position :"<<m_listeCurves[i].m_listCtrls[j].m_position;
		}*/
	}

	float getPrecision()
	{
		return m_directriceCurve.m_path->getPrecision();
	}

	QVector<QVector3D> getDirectriceInterpolated(float precision)
	{
		QVector<QVector3D> interpolated;

		interpolated.reserve(precision);
		float coef = 0.0f;
		for(int i=0;i<precision;i++)
		{
			coef = i/(precision-1.0f);
			QPointF pos2D = m_directriceCurve.m_path->getPosition(coef);
			QVector3D pos3D(pos2D.x(),m_bufferIso.getAltitude(pos2D),pos2D.y());
			QVector3D pos = m_transformScene * pos3D;
			interpolated.push_back(pos);

		}

		return interpolated;
	}

	QVector<QVector3D> getGeneratriceInterpolated(int index,float precision)
	{
		QVector<QVector3D> interpolated;
	/*	if(m_ortho ==nullptr)
		{
			qDebug()<<"Error nurbs n'a pas de random ortho ";
			return interpolated;

		}*/
		GraphEditor_ListBezierPath* current = m_listeCurves[index].m_path;


	//	qDebug()<<index<<" get world  transfo width: "<<m_listeCurves[index].m_randomTransfo->width();
		interpolated.reserve(precision);
		float coef = 0.0f;
		for(int i=0;i<precision;i++)
		{
			coef = i/(precision-1.0f);
			QPointF pos2D = m_listeCurves[index].m_path->getPosition(coef);
			//QVector3D posTr = m_ortho->viewWorldTo3dWordExtended(pos2D);
			QVector3D posTr = m_listeCurves[index].m_randomTransfo->imageToWorld(pos2D);
			QVector3D pos3D(posTr.x(),posTr.z(),posTr.y());
			QVector3D pos = m_transformScene * pos3D;
			interpolated.push_back(pos);

		}

		return interpolated;
	}


	QVector<QVector3D> get3DWorld(QVector<QPointF> liste,RandomTransformation* transfo)
	{
		QVector<QVector3D> res;
		//qDebug()<<" get world  transfo width: "<<transfo->width();
		for(int i=0;i<liste.count();i++)
		{
			//QVector3D posTr = m_ortho->viewWorldTo3dWordExtended(liste[i]);
			QVector3D posTr = transfo->imageToWorld(liste[i]);
			QVector3D pos3D(posTr.x(),posTr.z(),posTr.y());
			QVector3D pos = m_transformScene * pos3D;

			 //qDebug()<<liste[i]<<" ===> pos3D   : "<<posTr;
			res.push_back(pos);
		}

		return res;
	}
};


//using namespace std;

class NurbsEntity : public Qt3DCore::QEntity
{
    Q_OBJECT


public:
    explicit NurbsEntity(int precision, QVector3D posCam,Qt3DCore::QNode *parent = nullptr);
    ~NurbsEntity();

      QVector3D getPlanePosition()
    {
    	return m_positionPlane;
    }

    bool isOpen(){
    	return m_isOpenNurbs;
    }

    CurveListener* extrudeCurve(){return m_extrudecurvelistener;}
    void extrudecurveUpdated(bool finished);
    void extrudecurveDeleted();

    QVector3D  setXsectionPosWithTangent(float param,QVector3D position, QVector3D normal);
    QVector3D  setXsectionPos(float param);
    float getXSectionPos(){return m_xSectionParam;}
    helperqt3d::IsectPlane getXsectionPlane(float param);
    void  addUserDrawnCurveandXsection(float param, std::shared_ptr<CurveModel> curve, helperqt3d::IsectPlane& plane);
    void  addCurveandXsectionEnd();

    void  addinbetweenXsection(float pos, std::shared_ptr<CurveModel> curve);
    void  deleteXSection();
    int   numXsections(){return int(m_xsections.size());}
    bool  hasActiveXsection(){return m_extrudeSpline.isValid() && (m_xSectionParam!=-1);};
    bool  existsXsectionatParam(float param){ return (m_xsections.find(param)!=m_xsections.end());};

    CurveModel* getCurveModel(float param)
        {
        	auto position = m_xsections.find(param);
        	 XSectionInfo& xsect =position->second;
        	//XSectionInfo sectionInfo = m_xsections.find(param).second;
        	return xsect.curve;
        }


  /*  void setOrtho(RandomLineView* view)
    	{
    	m_managerCurvesOpt->setOrtho(view);
    	}*/

    void setInsertPointPos(float param);
    void insertCurveNurbspoint();

    void setShowNurbsPoints(bool show);
    void setOpenorClosed(bool isopen)            {m_isOpenNurbs            = isopen;        recalculateAndUpdateGeometry();}    // NB must be done before making nurbs!
    void setTriangulateResolution(int resolution){
    	m_triangulateResolution  = resolution;
    	recalculateAndUpdateGeometry();
    }
    void setWireframRendering(bool wireframe)    {m_wireframeRendering     = wireframe;     recalculateAndUpdateGeometry();}
    void setNuminbetweens(int numinbetweens)     {m_numinbetweens          = numinbetweens; recalculateAndUpdateGeometry();}
    void setLinearInterpolate(bool val)          {m_linearInterpolate      = val;           recalculateAndUpdateGeometry();}
    void setinbetweenxsectionalpha(float a)      {m_inbetweenxsectionalpha = a;             recalculateAndUpdateGeometry();}


    void setColorNurbs(QColor col);
    void setColorDirectrice(QColor col);

    QColor getColorNurbs();
    QColor getColorDirectrice();

    void setPositionLight(QVector3D);


    void createDirectriceFromTangent(std::vector<QVector3D> points,GraphEditor_ListBezierPath* path,QMatrix4x4 transformScene,IsoSurfaceBuffer m_bufferIso,QColor col);
    void createGeneratriceFromTangent(QVector<PointCtrl> listeCtrls,std::vector<QVector3D> directrice,QVector<QVector3D> listepoints,QVector<QVector3D> listeCtrl3D,QVector<QVector3D>  listeTangent3D,QVector3D cross3d, int index,bool isopen,QPointF cross);
    void createGeneratriceFromTangent(GraphEditor_ListBezierPath* path,RandomTransformation* transfo,std::vector<QVector3D> directrice, bool compute =true, float coef =-1.0f);
    void createGeneratriceFromTangent(QVector<PointCtrl> pts,RandomTransformation* transfo,std::vector<QVector3D> directrice,bool open,bool compute);
    void exportObj(QString path);
  //  void saveNurbs(QString path, std::vector<QVector3D>);


    helperqt3d::IsectPlane m_planeCurrent;
    helperqt3d::IsectPlane plane;

    GraphEditor_ListBezierPath* getCurrentPath(float pos);


    const QVector<PointCtrl>&  getCurrentCurve(float pos);
    QVector<QVector3D>  getCurrentPosition3D(float pos);

    QVector<QVector3D>  getCurrentTangente3D(float pos);

    helperqt3d::IsectPlane getCurrentPLane(float pos);

    int getNbCurves()
    	{
    		return m_managerCurves->getNbCurves();
    	}

    int getNbCurvesOpt()
       	{
       		return m_managerCurvesOpt->getNbCurves();
       	}

    QPointF getCrossPts(float pos)
    {
    	return m_managerCurves->getCurves(pos).m_cross2D;
    }

    CurveBezierManager* getManagerCurve()
    {
    	 return m_managerCurves;
    }
    CurveBezierOptManager* getManagerCurveOpt()
      {
      	 return m_managerCurvesOpt;
      }

    QVector<QVector3D> get3DWorld(QVector<QPointF> liste,RandomTransformation* transfo)
    {
    	return m_managerCurvesOpt->get3DWorld(liste,transfo);
    }

    void setPrecision(int value);

    int getPrecision();

    void deleteGeneratrice();

private:

    QVector<QVector3D> interpolator(float coef,helperqt3d::IsectPlane& planeInterpol);
    QVector<QVector3D> interpolatorOpt(float coef,helperqt3d::IsectPlane& planeInterpol, float precision);
    void UpdateGeometryWithTangent2(std::vector<QVector3D> directrice,bool isopen);
    void UpdateGeometryWithTangent3(std::vector<QVector3D> directrice,bool isopen);
    void UpdateGeometryWithTangentOpt(bool isopen);
    void UpdateGeometryWithTangent(std::vector<QVector3D> directrice, QVector<QVector3D> listepoints, int index,bool isopen);
    void recalculateAndUpdateGeometry();
    SplineMath m_extrudeSpline;

    //BezierMath m_extrudeBezier;
    //std::vector<QVector3D> m_ptsCtrlBezier;

    struct XSectionInfo
    {
        float param;                  // parameter along spline that this xsection is positioned at
        CurveModel* curve;            // the silhuette curve that is drawn for this cross section
        helperqt3d::IsectPlane plane; // the plane/x/y axis that the curve is drawn in
        SplineMath splinemath;        // a spline based on the points in <curve>
    };

    std::map<float,XSectionInfo> m_xsections;  // a map from parameter along extrusion spline to the cross sections

    XSectionInfo* m_selectedXSection = nullptr;  // If a user clicks on a xsection curve, it becomes selected
    void redrawandUpdateNurbsSpline(XSectionInfo& xsect);
    void makeinbetweenXsections(std::vector< CurveModel >& outcurves, int numnew, const XSectionInfo& beforestart, const XSectionInfo& start, const XSectionInfo& end, const XSectionInfo& afterend);
    void getInterpolatedCurve(XSectionInfo& result,  const XSectionInfo& start, const XSectionInfo& end);
    void getSmoothInterpolatedCurve(XSectionInfo& result, const XSectionInfo& beforestart, const XSectionInfo& start, const XSectionInfo& end, const XSectionInfo& afterend);
    void completeTheCurves(std::vector<CurveModel>& outcurves, int numsections);
    void copyXsectionToOtherDestionationOnSpline(const XSectionInfo& xsectFrom, XSectionInfo& xsectTo );

    void nurbscurveUpdated(bool finished,CurveModel* cm);
    void nurbscurveDeleted();
    void nurbscurveSetSelected(bool selected,CurveModel* cm);

    CurveListener*  nurbscurves(){return m_nurbscurvelistener;}

    PointsLinesGeometry* m_xSectionframeGeometry = nullptr; // the square frame around xsection
    PointsLinesGeometry* m_pointsGeometry        = nullptr; // for showing all nurbs control points and for showing the insert new point position
    PointsLinesGeometry* m_splineGeometry        = nullptr; // the spline on the selected xsection
    PointsLinesGeometry* m_extrudeSplineGeometry = nullptr; // the extrusion spline
    MeshGeometry2*        m_currentMesh           = nullptr; // the meshed nurbs function

    CurveListener* m_extrudecurvelistener;  // contains the extrude curve that the geometry is extruded along
    CurveListener* m_nurbscurvelistener;    // contains all the silhuette/outline curves on the xsections

    float m_xSectionParam = -1;
    float m_insertPointParam;
    bool  m_isOpenNurbs = true;
    int   m_triangulateResolution = 40;//40
    bool  m_wireframeRendering = false;
    bool  m_showNURBSPoints = false;
    int   m_numinbetweens;
    bool  m_linearInterpolate;
    float m_inbetweenxsectionalpha;
    QVector3D m_positionPlane;
   // static void getPlanevectorsFromNormal(const QVector3D& normal, QVector3D& vector1InPlane, QVector3D& vector2InPlane);


    QColor m_colorNurbs= Qt::blue;
    QColor m_colorDirectrice = Qt::yellow;
    Qt3DExtras::QDiffuseSpecularMaterial* m_matNurbs;
    Qt3DExtras::QDiffuseSpecularMaterial* m_matdirectrice;

    Qt3DRender::QMaterial * m_material;
    Qt3DRender::QParameter*   m_parameterColor;
    Qt3DRender::QParameter*   m_parameterLightPosition;

    Qt3DCore::QNode *m_parent;

    std::vector<QVector3D> m_vertices;
	std::vector<QVector3D> m_normals;
    std::vector<int> m_indices;


    CurveBezierManager* m_managerCurves= nullptr;

    CurveBezierOptManager* m_managerCurvesOpt= nullptr;


    int m_precision = 20;



    //tinynurbs::Surface3f m_nurbssurf;
};

