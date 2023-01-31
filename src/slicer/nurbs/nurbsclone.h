/*
 * NurbsClone.h
 *
 *  Created on: 09mars 2022
 *      Author: l1049100 (sylvain)
 */

#ifndef NURBSCLONE_H
#define NURBSCLONE_H

#include "curvemodel.h"
#include "curvecontroller.h"
#include "curvelistener.h"
#include "splinemath.h"


class CurveListener;

class Nurbs: public QObject
{

	Q_OBJECT
public:
    explicit Nurbs(QObject *parent = nullptr);
    ~Nurbs();

    QVector3D getPlanePosition()
    {
    	return m_positionPlane;
    }

    CurveListener* extrudeCurve(){return m_extrudecurvelistener;}
    void extrudecurveUpdated(bool finished);
    void extrudecurveDeleted();

    QVector3D  setXsectionPos(float param);
    float getXSectionPos(){return m_xSectionParam;};
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

    void setInsertPointPos(float param);
    void insertCurveNurbspoint();


    void setOpenorClosed(bool isopen)            {m_isOpenNurbs            = isopen;        recalculateAndUpdateGeometry();}    // NB must be done before making nurbs!
      void setNuminbetweens(int numinbetweens)     {m_numinbetweens          = numinbetweens; recalculateAndUpdateGeometry();}
    void setLinearInterpolate(bool val)          {m_linearInterpolate      = val;           recalculateAndUpdateGeometry();}
    void setinbetweenxsectionalpha(float a)      {m_inbetweenxsectionalpha = a;             recalculateAndUpdateGeometry();}


private:
    void recalculateAndUpdateGeometry();

    SplineMath m_extrudeSpline;

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


    CurveListener* m_extrudecurvelistener;  // contains the extrude curve that the geometry is extruded along
    CurveListener* m_nurbscurvelistener;    // contains all the silhuette/outline curves on the xsections

    float m_xSectionParam = -1;
    float m_insertPointParam;
    bool  m_isOpenNurbs = true;

    int   m_numinbetweens;
    bool  m_linearInterpolate;
    float m_inbetweenxsectionalpha;
    QVector3D m_positionPlane;

};



class NurbsClone
{
public:
	NurbsClone();

	void createDirectrice(QVector3D point);
	void updateDirectrice(QVector<QVector3D> listepoints);

	void addGeneratrice(QVector<QVector3D> listepoints,int index);

	void setOpenorClosed(bool isopen)            {m_isOpenNurbs            = isopen; }
	Nurbs* nurbs()
	{
		return m_nurbs;
	}


private:


	//CurveModel* m_curveDirectrice = nullptr;

	 CurveModel*          m_currentlyDrawnCurve = nullptr;
	 std::vector< std::shared_ptr<CurveModel> >  m_curves;

	 bool  m_isOpenNurbs = true;
	 Nurbs* m_nurbs = nullptr;





};


#endif
