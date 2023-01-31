#pragma once
#include <memory>
#include <QObject>
#include <QEntity>
#include <QPickingSettings>
#include <QPickEvent>
#include <QPickTriangleEvent>
#include <QPointer>

#include "curvemodel.h"
//#include "help.h"
#include "nurbsdataset.h"
#include "curvecontroller.h"
#include "nurbsclone.h"
#include "randomview3d.h"
#include "isochronprovider.h"
#include "qt3dhelpers.h"

//#include "GraphEditor_ListBezierPath.h"
//#include "PointCtrl.h"
//using namespace std;

class PointsLinesGeometry;
class MeshGeometry;
class NurbsEntity;
class CurveListener;
class NurbsDataset;
class randomView3D;
class PointCtrl;
class GraphEditor_ListBezierPath;
class RandomTransformation;

namespace Qt3DRender
{
    class QMaterial;
    class QGeometry;
    class QPickTriangleEvent;
}

class Manager : public QObject
{
    Q_OBJECT

public:
	struct Generatrix {
		QVector<PointCtrl> generatrix;
		bool generatrixClosed = false;
		double pos = 0;
		int widthAffine = 0;
		int heightAffine = 0;
		std::array<double, 6> directAffine = {0, 0, 0, 0, 0, 0};
		int widthRandom = 0;
		int heightRandom = 0;
		QPolygonF polygonRandom;
		QVector3D positionPlane;
		QVector3D axeXPlane;
		QVector3D axeYPlane;

	};

	struct NurbsParams
	{
		QColor color = Qt::white;
		int precision = 40;
		int timerLayer = -1;
		QVector<PointCtrl> directrix;
		bool directrixClosed = false;
		QVector<Generatrix> generatrices;
	};

    explicit Manager(WorkingSetManager* workingset,QString name, int precision, QVector3D posCam,Qt3DCore::QEntity* root,IsoSurfaceBuffer surface,int timeLayer, QObject* parent);
    explicit Manager(WorkingSetManager* workingset,QString name, QVector3D posCam,Qt3DCore::QEntity* root, QObject* parent);
    virtual ~Manager();

    //singleton
  //  static Manager* singleton(){return Manager::thesingleton;}

//private:   static Manager* thesingleton;


 /*   public: enum ToolbarState {EnumSelect, EnumSelectShift, EnumRotate, EnumResizebbox, EnumDraw, EnumErase};
    Q_ENUM(ToolbarState)   // https://doc.qt.io/qt-5/qtqml-cppintegration-data.html#enumeration-types
    private: ToolbarState m_toolbarState;
    public: void  setToolbarState(ToolbarState state){if (state==m_toolbarState) return; m_toolbarState = state; emit toolbarStateChanged();} ToolbarState toolbarState(){return m_toolbarState;}
    Q_PROPERTY(ToolbarState toolbarState READ toolbarState WRITE setToolbarState NOTIFY toolbarStateChanged) signals: void toolbarStateChanged();
*/



public:
  //   Q_PROPERTY(CurveInstantiator* curveinstantiator MEMBER m_curveInstantiator)  // set from qml (Scene.qml)

    signals:
	void sendNurbsY(QVector3D, QVector3D);
	void sendCurveData(std::vector<QVector3D>,bool);

//	void sendCurveData(QVector<PointCtrl*>,bool);
	void sendCurveDataTangent(QVector<PointCtrl>,bool ,QPointF);
	void sendCurveDataTangent2(QVector<QVector3D> ,QVector<QVector3D> ,bool ,QPointF);
	void sendCurveDataTangentOpt(GraphEditor_ListBezierPath*);

	void sendAnimationCam(int button,QVector3D pos);


	void generateDirectrice(QVector<PointCtrl>,QColor,bool);




public slots:

	void updateDirectriceCurve(QVector<QVector3D>);
    void curveDrawMouseBtnDownGeomIsect(QVector3D point);          // called when user is drawing spline on box geometry
    void curveDrawMouseBtnDownXsectIsect(int mouseX, int mouseY);  // called when user is drawing outline curve on x section

    void curveDrawSection();
    QVector3D ptsInPlane(helperqt3d::IsectPlane plane, QVector3D points);

    void curveDrawSection(QVector<QVector3D>, int index=-1,bool isopen =true);
    void endCurve();

    void addSelectedCurvepoint();
    void removeSelectedCurvepoint();
    void setOpenorClosed(bool isopen);

    void setSliderXsectPos(float param);
    QVector3D setSliderXsectPos(float param, QVector3D position,QVector3D normal);
    void addinbetweenXsection(float pos);
    void setNuminbetweens(int numinbetweens );
    void deleteXSection();
    void setLinearInterpolateInbetweens(bool val);
    void setinbetweenxsectionalpha(float a);

    void setSliderInsertPointPos(float param);
    void addSelectedCurveNurbspoint();

    void setWireframeRendering(bool wireframe);
    void setShowNurbsPoints(bool show);
    void setTriangulateResolution(int resolution);


    void resetNurbs();

    void destroyRandom3D(RandomView3D*);

    void addinbetweenXsectionClone(float pos, bool emitted = true);


public:
    std::shared_ptr<CurveModel> newCurve(helperqt3d::IsectPlane plane = helperqt3d::IsectPlane());
    void enablePicking(bool enabled){m_curveInstantiator->enablePicking(enabled);}
    void setNurbs(NurbsEntity* nurbs){m_nurbs=nurbs;}

    void setColorNurbs(QColor col);
    void setColorDirectrice(QColor col);

    void setPositionLight(QVector3D pos);

    QColor getColorNurbs();
     QColor getColorDirectrice();

     void createDirectriceFromTangent(GraphEditor_ListBezierPath*,QMatrix4x4,QColor);
     void createGeneratriceFromTangent(QVector<PointCtrl> listeCtrls,QVector<QVector3D> points,QVector<QVector3D> listeCtrl3D,QVector<QVector3D>  listeTangent3D,QVector3D cross3d, int index=-1,bool isopen =true, QPointF cross = QPointF(0,0));
     void createDirectriceFromTangent(std::vector<QVector3D> points);

   void   createGeneratriceFromTangent(GraphEditor_ListBezierPath* path,RandomTransformation* transfo,bool compute =true, float coef = 0);
   void createGeneratriceFromTangent(QVector<PointCtrl> listeCtrls,RandomTransformation* transfo,bool open,bool compute);
   /* QVector3D getPosCurrentPlane()
    {
    	return m_nurbs->getPlanePosition();
    }*/

   QVector3D  getPositionDirectrice(float pos);

   QPointF  getNormalDirectrice(float pos);

   void createNurbsGeneric(GraphEditor_ListBezierPath* path,RandomTransformation* transfo);


    void setDirectriceOk(int count)
    {
    	m_nbPtsDirectrice = count;
    	m_directriceOk= true;

    }

    QString getNameId(){ return m_nameId;}
    void setNameId(QString name ){ m_nameId = name;}

    std::vector< std::shared_ptr<CurveModel> > getOtherCurve()
    {
    	qDebug()<<" getOtherCurve size "<<m_curves.size();
    	return m_curves;
    }

    int getNbCurves()
    {
    	return m_curves.size();
    }

    CurveModel* getCurveDirectrice()
    {
    	return m_curveDirectrice;
    }

    NurbsDataset* getNurbsData()
    {
    	return m_nurbsData;
    }

    void setPrecision(int value);

    int getPrecision();

    void show();
    void hide();

    void setRandom3D(RandomView3D* rand);

    RandomView3D* getRandom3d()
    {
    	return m_random3d;
    }
    void deleteRandom3d()
    {
    	if(m_random3d != nullptr)
    	{
    		m_random3d->destroyRandom();
    		delete m_random3d;
    		m_random3d = nullptr;
    	}
    }

    float getXSection();

    void objToEditable(IsoSurfaceBuffer surface,QVector3D posCam, Qt3DCore::QEntity* root);
    void editerNurbs();

    void importNurbsObj(QString, Qt3DCore::QEntity*,QVector3D posCam, QColor col);
    void exportNurbsObj(QString);
    void saveNurbs(QString);
    void loadNurbs(QString ,QMatrix4x4);

    bool m_isTangent = false;


    float getAltitude(QPointF);

    bool getModeEditable()const
    {
    	return m_modeEditable;
    }


    int getTimerLayer()const
    {
    	return m_timeLayer;
    }

    void setTimeLayer(int time)
    {
    	m_timeLayer = time;
    }

	void deleteGeneratrice();

	static NurbsParams read(QString path, bool* ok=nullptr);
	static bool write(QString path, NurbsParams params);

private:
    void deleteCurve(CurveModel* curve);
    CurveModel*  getSelectedCurve();


    bool m_directriceOk=false;
    int m_nbPtsDirectrice =-1;
    QString m_nameId="";
    bool m_actifRayon= false;

    CurveModel* m_curveDirectrice = nullptr;

    NurbsEntity*         m_nurbs               = nullptr;
    CurveInstantiator*   m_curveInstantiator   = nullptr;
    CurveModel*          m_currentlyDrawnCurve = nullptr;
    std::vector< std::shared_ptr<CurveModel> >  m_curves; //  Curvecontroller also stores curvemodels, but only pointers to them. This is the owner of all curvemodels


    NurbsDataset* m_nurbsData = nullptr;
    NurbsClone* m_cloneNurbs = nullptr;

    QPointer<RandomView3D> m_random3d = nullptr;


    std::vector<QVector3D> m_listePosBezier;

    IsoSurfaceBuffer m_bufferIso;

    QPointer<GraphEditor_ListBezierPath> m_directriceBezier;
    QMatrix4x4 m_transformScene;

    Qt3DRender::QParameter* m_parameterColor=nullptr;


  //  std::vector<QVector3D> m_generatrice;

    Qt3DCore::QEntity* m_entityObj = nullptr;
    bool m_modeEditable = false;
    QColor m_colorObj;
    int m_timeLayer=-1;
    Qt3DRender::QParameter* m_parameterLightPosition= nullptr;


    NurbsParams m_nurbsParam;

   // bool m_directriceOk = false;



};
