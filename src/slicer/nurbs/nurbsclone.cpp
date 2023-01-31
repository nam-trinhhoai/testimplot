#include "nurbsclone.h"

#include <functional>

NurbsClone::NurbsClone()
{
	m_nurbs = new Nurbs();


}

void NurbsClone::createDirectrice(QVector3D point)
{
	if (m_currentlyDrawnCurve==nullptr)
	    {
	        std::shared_ptr<CurveModel> curve = std::make_shared<CurveModel>();
	           m_curves.push_back(curve);
	        m_nurbs->extrudeCurve()->addCurve(curve);
	        m_currentlyDrawnCurve = curve.get();
	     //   m_curveDirectrice = curve.get();
	      //  m_curveInstantiator->selectCurve(m_currentlyDrawnCurve);
	      //  m_curveInstantiator->enablePicking(false);
	    }
	    m_currentlyDrawnCurve->insertBack(point);
}

void NurbsClone::updateDirectrice(QVector<QVector3D> listepoints)
{
	if (m_currentlyDrawnCurve!=nullptr)
	{
		m_currentlyDrawnCurve->clear();

		for(int i=0;i<listepoints.count();i++)
		{
			m_currentlyDrawnCurve->insertBack(listepoints[i]);

		}


	}
	/*if (m_currentlyDrawnCurve!=nullptr)
	  {

		  for(int i=0;i<listepoints.count();i++)
		  {
			  int sizeCurrent = m_currentlyDrawnCurve->getSize();
			  if(i<sizeCurrent)
			  {
				  if(!qFuzzyCompare(listepoints[i],m_currentlyDrawnCurve->getPosition(i)))
				  {
					  m_currentlyDrawnCurve->setPoint(i,listepoints[i]);
					  }
			  }
			  else
			  {
				  m_currentlyDrawnCurve->insertBack(listepoints[i]);
			  }
		  }

			  //m_curveDirectrice->emitModelUpdated(true);
	  }*/
}


void NurbsClone::addGeneratrice(QVector<QVector3D> listepoints,int index)
{
	float pos  =m_nurbs->getXSectionPos();
	bool exist = m_nurbs->existsXsectionatParam(pos);
	if(exist)
	{
		//qDebug()<<" existe nurbs , index:"<<index;
		CurveModel* curve = m_nurbs->getCurveModel(pos);
		int indice =index;
		if(indice>=0)
		{

			QVector3D newPos = listepoints[indice];
			//qDebug()<<indice<<"curveDrawSection newPos "<<newPos;
			curve->setPoint((uint)indice,newPos);
		}
		else
		{
			curve->setAllPoints(listepoints);
		}
		return ;
		/*m_curveInstantiator->selectCurve(m_nurbs->getCurveModel(pos));

		//get index current
		int indice =index;
		if(indice>=0)
		{

			QVector3D newPos = listepoints[indice];
			//qDebug()<<indice<<"curveDrawSection newPos "<<newPos;
			m_curveInstantiator->getSelectedCurve()->setPoint((uint)indice,newPos);
		}
		return ;*/
	}

	/*if (m_directriceOk ==false)
	{
		qDebug()<<"not found directrice nurbs";
		return;
	}*/


	helperqt3d::IsectPlane plane = m_nurbs->getXsectionPlane(m_nurbs->getXSectionPos());

	  std::shared_ptr<CurveModel> curve =std::make_shared<CurveModel>();
      m_curves.push_back(curve);

	 m_nurbs->addUserDrawnCurveandXsection(m_nurbs->getXSectionPos(),  curve, plane);

	// m_currentlyDrawnCurve = curve.get();
	// m_curveInstantiator->selectCurve(m_currentlyDrawnCurve);


	 for(int i=0;i<listepoints.count();i++)
	 {

		 curve->insertBack(listepoints[i]);

	 }

}


//========================================================================================================
//========================================================================================================




Nurbs::Nurbs(QObject *parent) : QObject(parent)
{
	 m_extrudecurvelistener = new CurveListener(this);
		 m_nurbscurvelistener = new CurveListener(this);
	 using namespace std::placeholders;
	    m_extrudecurvelistener->setCallbackCurveUpdated( std::bind(&Nurbs::extrudecurveUpdated,     this , _1));
	    m_extrudecurvelistener->setCallbackCurveDeleted( std::bind(&Nurbs::extrudecurveDeleted,     this));
	    //  m_extrudecurvelistener->setCallbackCurveSelected(std::bind(&NurbsEntity::extrudecurveSetSelected, this, _1, _2));

	    m_nurbscurvelistener->setCallbackCurveUpdated( std::bind(&Nurbs::nurbscurveUpdated,     this , _1, _2));
	 //   m_nurbscurvelistener->setCallbackCurveDeleted( std::bind(&NurbsEntity::nurbscurveDeleted,     this));
	    m_nurbscurvelistener->setCallbackCurveSelected(std::bind(&Nurbs::nurbscurveSetSelected, this, _1, _2));
}


Nurbs::~Nurbs()
{

}



helperqt3d::IsectPlane Nurbs::getXsectionPlane(float param)
{
    float param2=    m_extrudeSpline.todistance(param);

    glm::vec3 start =  m_extrudeSpline.getPosition(param2);
    glm::vec3 tang =   m_extrudeSpline.getTangent(param2);

    QVector3D pos = QVector3D(start.x,start.y,start.z);
    QVector3D vector1InPlane, vector2InPlane;
    helperqt3d::getPlanevectorsFromNormal(QVector3D(tang.x, tang.y, tang.z),vector1InPlane,vector2InPlane);

    return helperqt3d::IsectPlane{pos,vector1InPlane,vector2InPlane};
}

void Nurbs::nurbscurveDeleted()
{
   // qDebug("NurbsEntity curveDeleted");
  //  recalculateAndUpdateGeometry();
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

QVector3D Nurbs::setXsectionPos(float pos)
{
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

   // m_xSectionframeGeometry->updateData(lines.data());

    return plane.getNormal();
   /* if (m_xsections.size()>0)
    {
    glm::vec3 pt   = tinynurbs::surfacePoint(m_nurbssurf, pos, 0.0f);
       qDebug()<<" point nurbs"<<pt[0]<<"  "<<pt[1]<<"  "<<pt[2];
    }*/

}

void Nurbs::setInsertPointPos(float param)
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
        //m_pointsGeometry->updateData(cm.data());
    }



}

void Nurbs::extrudecurveUpdated(bool finished)
{
    //if (!finished) return;  // use this line if you want refresh only when control point is released after move and not continuosly

    CurveModel& controlpts = *extrudeCurve()->getCurves()[0].get();
    int numctrlpts=controlpts.getSize();
    // qDebug() << "extrudecurveUpdated" << numctrlpts;
    if (numctrlpts<4) return;

    m_extrudeSpline.setOpen(true);
    m_extrudeSpline.setPoints(&controlpts);

    CurveModel samplednurbs;
    m_extrudeSpline.sampletheSpline(&samplednurbs);
   // m_extrudeSplineGeometry->updateData(samplednurbs.data());

    //recalculateAndUpdateGeometry();
};


void Nurbs::addinbetweenXsection(float param, std::shared_ptr<CurveModel> curve)
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

void Nurbs::deleteXSection()
{
    auto it =  m_xsections.find(m_xSectionParam);
    if (it!=m_xsections.end())
    {
        it->second.curve->emitModeltobeDeleted();
        m_xsections.erase(it);

        //recalculateAndUpdateGeometry();
       // m_splineGeometry->clearData();
    }
}

void Nurbs::insertCurveNurbspoint()
{
  /*  m_nurbscurvelistener->mute();
    for (auto& xsect : m_xsections) // add the knot point and update the control points for all curves
        xsect.second.splinemath.addShapepreservingPoint(m_insertPointParam);

    m_nurbscurvelistener->unmute();*/
}

void Nurbs::nurbscurveUpdated(bool finished, CurveModel* cm)
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
void Nurbs::nurbscurveSetSelected(bool selected, CurveModel* cm)
{
    if (!selected) return;
    auto position=find_if(m_xsections.begin(), m_xsections.end(),    // find the curvecontroller that is associated with cm
                            [&] (const auto&  c) {return c.second.curve == cm; } );

    setXsectionPos(position->first);
    redrawandUpdateNurbsSpline(position->second);
}


void Nurbs::extrudecurveDeleted()
{
    qDebug() << " extrudecurveDeleted ";
    m_xSectionParam = -1;
};

void Nurbs::addUserDrawnCurveandXsection(float param, std::shared_ptr<CurveModel> curve,  helperqt3d::IsectPlane& plane)
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
void Nurbs::addCurveandXsectionEnd()
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

void Nurbs::redrawandUpdateNurbsSpline(XSectionInfo& xsect)
{
    //  qDebug() << "redrawandUpdateNurbsSpline";
    CurveModel* curve = xsect.curve;
    if (curve->getSize()<4) return;
    // render the spline curve
    CurveModel samplednurbs;
    xsect.splinemath.setPoints(curve);
    xsect.splinemath.sampletheSpline(&samplednurbs);
   // m_splineGeometry->updateData(samplednurbs.data());
}


// Based on the user defined cross sections, this function creates interpolated inbetween cross sections
// and then uses all the cross section control points for generating a nurbs. Finally, the nurbs is sampled and rendered
void Nurbs::recalculateAndUpdateGeometry()
{
    if (m_xsections.size()==0)
    {
        //m_currentMesh->setEmptyData();
       // m_pointsGeometry->clearData();
        return;  // nothing to calculate
    }

    return;
/*    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::vector< CurveModel > curves;
    completeTheCurves(curves,m_numinbetweens);


  //  qDebug()<<" curves.size() :"<<curves.size();
    if (curves.size()<4) return;
   // bool wireframe = m_wireframeRendering;

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
            QVector3D p = cm.getPosition(0);
            ptsArray.push_back(glm::vec3(p[0],p[1],p[2]));
        }
    }

    int szV = ptsV;
    if (!m_isOpenNurbs) szV += 1;

    m_nurbssurf.control_points = {(size_t)szU, (size_t)szV, ptsArray};

    bool isvalid = tinynurbs::surfaceIsValid(m_nurbssurf);
    if (!isvalid)
    {qDebug()<< "nurbs surface not valid!"; return;}

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      qDebug() <<" Nurbs , recalculateAndUpdateGeometry Time : " << std::chrono::duration<double, std::milli>(end-start).count();
*/
    //        evaluating nurbs to create geometry    //////////////////////////
/*    NurbsHelper   openuVals(ptsV,   NurbsHelper::Type::Clamped);
    NurbsHelper closeduVals(ptsV+1, NurbsHelper::Type::Clamped);
    NurbsHelper vVals=m_isOpenNurbs?openuVals:closeduVals;

    QVector<QVector3D> normals3D;
    std::vector<QVector3D> vertices,colors,normals;
    std::vector<int> indices;

    int steps = m_triangulateResolution;
    for (int i=0;i<steps;i++)
        for (int j=0;j<steps;j++)
        {
            float u = uVals.getParameter(i,steps);
            float v = vVals.getParameter(j,steps);

            glm::vec3 pt   = tinynurbs::surfacePoint (m_nurbssurf, u, v);
            glm::vec3 norm = tinynurbs::surfaceNormal(m_nurbssurf, u, v);

            vertices.push_back(QVector3D(pt.x,pt.y,pt.z));

            normals.push_back(QVector3D(norm.x,norm.y,norm.z));
            normals3D.push_back(QVector3D(norm.x,norm.y,norm.z));


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

                indices.push_back(ndx00);
                indices.push_back(ndx10);
                indices.push_back(ndx01);

                indices.push_back(ndx10);
                indices.push_back(ndx11);
                indices.push_back(ndx01);

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

    m_currentMesh->setRenderLines(wireframe);
    m_currentMesh->uploadMeshData(vertices, indices, normals);// ,colors);
*/
    //render the control points


   /* if (m_showNURBSPoints)
        m_pointsGeometry->updateData(renderControlpoints.data());
    else
        m_pointsGeometry->clearData();*/
}


// Based on the m_analyticspline, this function takes a curve defined on xsect and copies it into xsectTo based on xsectTo.param
void Nurbs::copyXsectionToOtherDestionationOnSpline(const XSectionInfo& xsectFrom, XSectionInfo& xsectTo )
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
void Nurbs::completeTheCurves(std::vector< CurveModel >& outcurves, int numsections)
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


void Nurbs::makeinbetweenXsections(std::vector< CurveModel >& outcurves, int numnew, const XSectionInfo& beforestart, const XSectionInfo& start, const XSectionInfo& end, const XSectionInfo& afterend)
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
void Nurbs::getSmoothInterpolatedCurve(XSectionInfo& result, const XSectionInfo& beforestart, const XSectionInfo& start, const XSectionInfo& end, const XSectionInfo& afterend)
{
    if (start.curve->getSize()!=end.curve->getSize())
    {qDebug() << "Curves must be of equal size"; return;}

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
void Nurbs::getInterpolatedCurve(XSectionInfo& result, const XSectionInfo& start, const XSectionInfo& end)
{
    const XSectionInfo& interpfrom = start;
    const XSectionInfo& interpto   = end;

    if (interpfrom.curve->getSize()!=interpto.curve->getSize())
    {qDebug() << "Curves must be of equal size"; return;}

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



