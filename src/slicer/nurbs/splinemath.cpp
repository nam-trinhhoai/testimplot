#include "splinemath.h"


SplineMath::SplineMath()
{
	 m_tinycurve.degree = 3;
}

void SplineMath::update()
{
    if (!m_points) return;

    m_tinycurve.degree = 3;

    int numctrlpts = m_points->getSize();

//  qDebug() << "cptrlpts  " << m_tinycurve.control_points.size() << "knots " <<m_tinycurve.knots.size() << " curvemodelpts " << numctrlpts;

    if (m_tinycurve.control_points.size()>numctrlpts+(m_isOpen?0:1)) // A point was added
        m_updateKnots = true;

    if (m_tinycurve.control_points.size()<numctrlpts+(m_isOpen?0:1)) // A point was removed
        m_updateKnots = true;

    m_tinycurve.control_points.clear();

    NurbsHelper   open(numctrlpts,    NurbsHelper::Type::Clamped);
    NurbsHelper closed(numctrlpts+1,  NurbsHelper::Type::Clamped);
    m_generator=m_isOpen?open:closed;

    // Don't generate new knots if addShapepreservingPoint() has been called. We need to use these knots so that the curve does not change its shape
    if (m_updateKnots)
    {
        m_tinycurve.knots.clear();
        for (int i=1;i<=m_generator.getNumKnots(); i++)
            m_tinycurve.knots.push_back(m_generator.getKnot(i));
    }

    for (int i=0;i<numctrlpts;i++)
    {
        QVector3D p = m_points->getPosition(i);
        m_tinycurve.control_points.push_back(glm::vec3(p[0],p[1],p[2]));
    }
    if (!m_isOpen)
    {
        QVector3D p = m_points->getPosition(0);
        m_tinycurve.control_points.push_back(glm::vec3(p[0],p[1],p[2]));
    }

    bool isvalid = tinynurbs::curveIsValid(m_tinycurve);
    if (!isvalid)  {qDebug()<< "nurbs curve INVALID!"; return;}

    makecurvelengthtable();
}

bool SplineMath::isValid()
{
    return tinynurbs::curveIsValid(m_tinycurve);
}

void SplineMath::sampletheSpline(CurveModel* result, int numsamples)
{

   // qDebug() << "min max param " << m_generator.getMinParameter() << m_generator.getMaxParameter();
    for (int i=0;i<numsamples;i++)
    {
        float parameter =  m_generator.getParameter(i,numsamples);
        glm::vec3 pt = tinynurbs::curvePoint(m_tinycurve, parameter);
        result->insertBackSilent(QVector3D(pt.x,pt.y,pt.z));
    }
    result->emitModelUpdated(false);
}

void SplineMath::addShapepreservingPoint(float normalizedparameter)
{
    qDebug() << "addShapepreservingPoint";
  //  normalizedparameter = todistance(normalizedparameter);
    float u =  m_generator.getParameter(normalizedparameter);
    m_tinycurve = tinynurbs::curveKnotInsert(m_tinycurve, u);
    m_points->clear();
    for (int i=0; i < m_tinycurve.control_points.size()-(m_isOpen?0:1); i++)
    {
        glm::vec3 val = m_tinycurve.control_points[i];
        m_points->insertBack(QVector3D(val[0], val[1], val[2]));
    }
    m_updateKnots = false;
}

glm::vec3 SplineMath::getPosition(float normalizedparameter)
{
    float u =  m_generator.getParameter(normalizedparameter);
    return tinynurbs::curvePoint(m_tinycurve, u);
}

glm::vec3 SplineMath::getTangent(float normalizedparameter)
{
     float u =  m_generator.getParameter(normalizedparameter);
     return  tinynurbs::curveTangent(m_tinycurve, u);
}

// maps from a spline parameter in [0,1] to a new parameter in [0,1] which is distance preserving.
// i.e. this is a reparameterization by arc length
float SplineMath::todistance(float normalizedparameter)
{
    // m_curvelength is discretized, so to make it more continuous, we linearly interpolate between the samples
    int sz = m_curvelengthtable.size();
    float index = float(sz-1)*normalizedparameter;
    int ilow  = std::floor(index);
    int ihigh = std::min(ilow+1, sz-1);
    float fract = index-ilow;
    float dist = m_curvelengthtable[ilow]*(1-fract) +  m_curvelengthtable[ihigh]*fract;
    return dist;
}

// Higher <steps> gives more accurate result
// Note: This is a coarse approximation and better algorithms exist
void SplineMath::makecurvelengthtable(int steps)
{
    // first calculate length of the curve
    glm::vec3 pos;
    float length=0;
    glm::vec3 oldpos = getPosition(0);

    for (int i=1; i<steps; i++)
    {
        float normalized = float(i)/float(steps-1);
        pos = getPosition(normalized);
        length += glm::length(pos-oldpos);
        oldpos=pos;
    }

    m_curvelength = length;

    // Then fill in table m_curvelength of approximately equal length distances
    // Then m_curvelength[x] where x=0 represents 0 and x=m_curvelength.size() represents 1
    // is a function mapping from a spline parameter in [0,1] to the actual spline parameter that must be used
    // to get to a point on the curve according to the length of the curve. I.e. looking up in the middle of m_curvelength gives
    // the parameter that is the center of the curve according to length.
    float steplength = 5*length/float(steps);

   // qDebug() << "Length of curve :" << length << "  step length : " << steplength;

    oldpos = getPosition(0);
    float distance   = 0;
    float eqstepdist = steplength;

    m_curvelengthtable.clear();

    m_curvelengthtable.push_back(0);

    for (int i=1; i<steps-1; i++)
    {
        float normalized = float(i)/float(steps-1);
        pos = getPosition(normalized);
        distance += glm::length(pos-oldpos);

        if (eqstepdist<=distance)
        {
            m_curvelengthtable.push_back(normalized);
            eqstepdist += steplength;
        }
        oldpos=pos;
    }

    m_curvelengthtable.push_back(1);
    /*
    for (int i=0;i<m_curvelength.size(); i++)
        qDebug() << i << m_curvelength[i];// << m_curvelength[i]-m_curvelength[std::max(i-1,0)];
    */

}


//=========================================================
BezierMath::BezierMath()
{

}


void BezierMath::fillKnotVector()
{

	if (m_polygon.size()>2)
	{
		int middleKnotNumber = m_polygon.size() - 4;
		m_knotVector.clear();
		for (int counter = 0; counter < 4; ++counter)
			m_knotVector.push_back(0.0);
		for (int counter = 1; counter <= middleKnotNumber; ++counter)
			m_knotVector.push_back(1.0 / (middleKnotNumber + 1) * counter);
		for (int counter = 0; counter < 4; ++counter)
			m_knotVector.push_back(1.0);
	}

}

void  BezierMath::interpolateCurve()
{

	if (m_polygon.size()>2)
	{
		m_interpolatedPoints.clear();

		if (m_IsClosedCurved)
		{
			QVector<QPointF> ctrlPoints = m_polygon;
			int  currentK =  3 ;

			for  (qreal u = currentK; u < ctrlPoints.size(); u +=  0.01 )
			{
				QPointF pt( 0.0 ,  0.0 );
				for  ( int  i = 0 ; i < ctrlPoints.size(); ++i){
					QPointF pts = ctrlPoints[i];
					pts *= NN(currentK, i, u);
					pt += pts;
				}
				m_interpolatedPoints.push_back(pt);
			}

		}
		else
		{
			m_controlPoints.clear();
			foreach(QPointF p, m_polygon)
			{
				m_controlPoints.push_back(new QPointF(p));
			}
			m_boorNetPoints.clear();
			m_bezierInterpolator.CalculateBoorNet(m_controlPoints, m_knotVector, m_boorNetPoints);
			m_interpolatedPoints.push_back(*(m_controlPoints.first()));
			for (int counter = 0; counter < m_boorNetPoints.size() - 3; counter += 3)
				m_bezierInterpolator.InterpolateBezier(m_boorNetPoints[counter],
						m_boorNetPoints[counter + 1],
						m_boorNetPoints[counter + 2],
						m_boorNetPoints[counter + 3],
						m_interpolatedPoints);
			m_interpolatedPoints.push_back(*(m_controlPoints.last()));

		}
	}
}

QPointF BezierMath::bezierCurveByCasteljau(float u)
{
/*	QVector<QPointF> ctrlPts;
		for (int i=0;i< m_controlPoints.size();i++)
			ctrlPts.push_back(*m_controlPoints[i]);*/
	QPointF  pos= bezierCurveByCasteljauRec(m_polygon, u);

	return pos;
}

QPointF BezierMath::bezierCurveByCasteljauRec(QVector<QPointF> in_pts, float i){
    if(in_pts.size() == 1) return in_pts[0];
    QVector<QPointF> pts ;
    for(unsigned int it = 0 ; it < in_pts.size() - 1; it++){
    	QPointF vecteur = in_pts[it + 1] - in_pts[it];
        vecteur = vecteur * i;
        pts.push_back(in_pts[it] + vecteur);
    }

    return bezierCurveByCasteljauRec(pts, i);
}

/*

 std::vector<glm::vec3> & Curve::bezierCurveByCasteljau(const std::vector<glm::vec3>& controlPoints){
    _controlPoints = controlPoints;
    _vertices.clear();

    auto etape = (float)_iterations;
    if(etape < 0) etape = 0;
    if(etape > (float) NB_POINTS*_controlPoints.size()) etape = (float) NB_POINTS;
    long int nbU = NB_POINTS;

    float i = 0.0f;
    while(i <=  etape/(float)nbU){
        glm::vec3 v = bezierCurveByCasteljauRec(controlPoints, (float) i);
        //std::cout << v.x << " " << v.y << " " << v.z << std::endl;
        _vertices.push_back(v);
        i+=1.0f/(float)nbU;
    }
    if( etape/(float)nbU == 1) _vertices.push_back(controlPoints.at(controlPoints.size()-1));
    return _vertices;
}

glm::vec3 Curve::bezierCurveByCasteljauRec(std::vector<glm::vec3> in_pts, float i){
    if(in_pts.size() == 1) return in_pts.at(0);
    std::vector<glm::vec3> pts ;
    for(unsigned int it = 0 ; it < in_pts.size() - 1; it++){
        glm::vec3 vecteur = in_pts.at(it + 1 ) - in_pts.at(it);
        vecteur = vecteur * i;
        pts.push_back(in_pts.at(it) + vecteur);
    }

    return bezierCurveByCasteljauRec(pts, i);
}
 */

void BezierMath::update()
{
	m_polygon.clear();

	for(int i=0;i<m_points->data().size();i++)
	{
		m_polygon.push_back(QPointF(m_points->data()[i].x(),m_points->data()[i].z()));
	}

 	fillKnotVector();
	interpolateCurve();
}

bool BezierMath::isValid()
{

}

void BezierMath::sampletheSpline(CurveModel* result, int numsamples)
{


}

void BezierMath::addShapepreservingPoint(float normalizedparameter)
{

}
glm::vec3 BezierMath::getPosition(float mu)
{
	QPointF pts = bezierCurveByCasteljau(mu);
	return glm::vec3(pts.x(),0.0f,pts.y());


}

glm::vec3 BezierMath::getTangent(float normalizedparameter)
{

}

// maps from a spline parameter in [0,1] to a new parameter in [0,1] which is distance preserving.
// i.e. this is a reparameterization by arc length
float BezierMath::todistance(float normalizedparameter)
{

}

// Higher <steps> gives more accurate result
// Note: This is a coarse approximation and better algorithms exist
void BezierMath::makecurvelengthtable(int steps)
{

}




qreal BezierMath::NN1( int  i, qreal u)
{
	qreal t = u - i;
	if  ( 0  <= t && t <  1 ){
		return  t;
	}
	if  ( 1  <= t && t <  2 ){
		return   2  - t;
	}
	return   0 ;
}

qreal BezierMath::NN2( int  i, qreal u)
{
	qreal t = u - i;
	if  ( 0  <= t && t <  1 ){
		return   0.5  * t * t;
	}
	if  ( 1  <= t && t <  2 ){
		return   3  * t - t * t - 1.5 ;
	}
	if  ( 2  <= t && t <  3 ){
		return   0.5  * pow( 3  - t,  2 );
	}
	return   0 ;
}

qreal BezierMath::NN3( int  i, qreal u)
{
	qreal t = u - i;
	qreal a =  1.0  /  6.0 ;
	if  ( 0  <= t && t <  1 ){
		return  a * t * t * t;
	}
	if  ( 1  <= t && t <  2 ){
		return  a * (- 3  * pow(t -  1 ,  3 ) +  3  * pow(t -  1 ,  2 ) +  3  * (t -  1 ) +  1 );
	}
	if  ( 2  <= t && t <  3 ){
		return  a * ( 3  * pow(t -  2 ,  3 ) -  6  * pow(t -  2 ,  2 ) +  4 );
	}
	if  ( 3  <= t && t <  4 ){
		return  a * pow( 4  - t,  3 );
	}
	return   0 ;
}

qreal BezierMath::NN( int  k,  int  i, qreal u)
{
	switch  (k) {
	case   1 :
		return  NN1(i, u);
	case   2 :
		return   NN2(i, u);
	case   3 :
		return   NN3(i, u);
	default :
		break ;
	}
}


//========================================================================================================



//==========================================================================================================
//==========================================================================================================

/*
 * bezierinterpolator.cpp
 *
 *  Created on: Oct 27, 2021
 *      Author: l1046262
 */


#include <QtCore/qmath.h>
#include <QHash>

const unsigned BezierInterpolator2D::curveRecursionLimit = 32;
const double BezierInterpolator2D::curveCollinearityEpsilon = 1e-30;
const double BezierInterpolator2D::curveAngleToleranceEpsilon = 0.01;
const double BezierInterpolator2D::AngleTolerance = 0.0;
const double BezierInterpolator2D::CuspLimit = 0.0;

BezierInterpolator2D::BezierInterpolator2D() : DistanceTolerance(0.5) {}

// InterpolateBezier - interpolates points with bezier curve.
// Algorithm is based on article "Adaptive Subdivision of Bezier Curves" by
// Maxim Shemanarev.
// http://www.antigrain.com/research/adaptive_bezier/index.html
void BezierInterpolator2D::InterpolateBezier(double x1, double y1,
                                           double x2, double y2,
                                           double x3, double y3,
                                           double x4, double y4,
                                           QVector<QPointF> &interpolatedPoints,
                                           unsigned level) const {
  if(level > curveRecursionLimit) {
    return;
  }

  // Calculate all the mid-points of the line segments
  double x12   = (x1 + x2) / 2;
  double y12   = (y1 + y2) / 2;
  double x23   = (x2 + x3) / 2;
  double y23   = (y2 + y3) / 2;
  double x34   = (x3 + x4) / 2;
  double y34   = (y3 + y4) / 2;
  double x123  = (x12 + x23) / 2;
  double y123  = (y12 + y23) / 2;
  double x234  = (x23 + x34) / 2;
  double y234  = (y23 + y34) / 2;
  double x1234 = (x123 + x234) / 2;
  double y1234 = (y123 + y234) / 2;

  if(level > 0) {
    // Enforce subdivision first time

    // Try to approximate the full cubic curve by a single straight line
    double dx = x4-x1;
    double dy = y4-y1;

    double d2 = qAbs(((x2 - x4) * dy - (y2 - y4) * dx));
    double d3 = qAbs(((x3 - x4) * dy - (y3 - y4) * dx));

    double da1, da2;

    if(d2 > curveCollinearityEpsilon && d3 > curveCollinearityEpsilon) {
      // Regular care
      if((d2 + d3)*(d2 + d3) <= DistanceTolerance * (dx*dx + dy*dy)) {
        // If the curvature doesn't exceed the distance_tolerance value
        // we tend to finish subdivisions.
        if(AngleTolerance < curveAngleToleranceEpsilon) {
          interpolatedPoints.push_back(QPointF(x1234, y1234));
          return;
        }

        // Angle & Cusp Condition
        double a23 = qAtan2(y3 - y2, x3 - x2);
        da1 = fabs(a23 - qAtan2(y2 - y1, x2 - x1));
        da2 = fabs(qAtan2(y4 - y3, x4 - x3) - a23);
        if(da1 >= M_PI) da1 = 2*M_PI - da1;
        if(da2 >= M_PI) da2 = 2*M_PI - da2;

        if(da1 + da2 < AngleTolerance) {
          // Finally we can stop the recursion
          interpolatedPoints.push_back(QPointF(x1234, y1234));
          return;
        }

        if(CuspLimit != 0.0) {
          if(da1 > CuspLimit) {
            interpolatedPoints.push_back(QPointF(x2, y2));
            return;
          }

          if(da2 > CuspLimit) {
            interpolatedPoints.push_back(QPointF(x3, y3));
            return;
          }
        }
      }
    } else {
      if(d2 > curveCollinearityEpsilon) {
        // p1,p3,p4 are collinear, p2 is considerable
        if(d2 * d2 <= DistanceTolerance * (dx*dx + dy*dy)) {
          if(AngleTolerance < curveAngleToleranceEpsilon) {
            interpolatedPoints.push_back(QPointF(x1234, y1234));
            return;
          }

          // Angle Condition
          da1 = fabs(qAtan2(y3 - y2, x3 - x2) - qAtan2(y2 - y1, x2 - x1));
          if(da1 >= M_PI)
            da1 = 2*M_PI - da1;

          if(da1 < AngleTolerance) {
            interpolatedPoints.push_back(QPointF(x2, y2));
            interpolatedPoints.push_back(QPointF(x3, y3));
            return;
          }

          if(CuspLimit != 0.0) {
            if(da1 > CuspLimit) {
              interpolatedPoints.push_back(QPointF(x2, y2));
              return;
            }
          }
        }
      } else if(d3 > curveCollinearityEpsilon) {
        // p1,p2,p4 are collinear, p3 is considerable
        if(d3 * d3 <= DistanceTolerance * (dx*dx + dy*dy)) {
          if(AngleTolerance < curveAngleToleranceEpsilon) {
            interpolatedPoints.push_back(QPointF(x1234, y1234));
            return;
          }

          // Angle Condition
          da1 = fabs(qAtan2(y4 - y3, x4 - x3) - qAtan2(y3 - y2, x3 - x2));
          if(da1 >= M_PI) da1 = 2*M_PI - da1;

          if(da1 < AngleTolerance) {
            interpolatedPoints.push_back(QPointF(x2, y2));
            interpolatedPoints.push_back(QPointF(x3, y3));
            return;
          }

          if(CuspLimit != 0.0) {
            if(da1 > CuspLimit) {
              interpolatedPoints.push_back(QPointF(x3, y3));
              return;
            }
          }
        }
      } else {
        // Collinear case
        dx = x1234 - (x1 + x4) / 2;
        dy = y1234 - (y1 + y4) / 2;
        if(dx*dx + dy*dy <= DistanceTolerance) {
          interpolatedPoints.push_back(QPointF(x1234, y1234));
          return;
        }
      }
    }
  }

  // Continue subdivision
  InterpolateBezier(x1, y1, x12, y12, x123, y123, x1234, y1234,
                    interpolatedPoints, level + 1);
  InterpolateBezier(x1234, y1234, x234, y234, x34, y34, x4, y4,
                    interpolatedPoints, level + 1);
}

void BezierInterpolator2D::InterpolateBezier(const QPointF &p1, const QPointF &p2,
                                           const QPointF &p3, const QPointF &p4,
										   QVector<QPointF> &interpolatedPoints,
                                           unsigned level) const {
  InterpolateBezier(p1.x(), p1.y(), p2.x(), p2.y(), p3.x(), p3.y(), p4.x(),
                    p4.y(), interpolatedPoints, level);
}

// CalculateBoorNet - inserts new control points with de Boor algorithm for
// transformation of B-spline into composite Bezier curve.
void BezierInterpolator2D::CalculateBoorNet(const QVector<QPointF *> &controlPoints,
    const QVector<qreal> &knotVector,
	QVector<QPointF> &boorNetPoints) const {
  Q_ASSERT(controlPoints.size() > 2);
  Q_ASSERT(knotVector.size() > 4);
  // We draw uniform cubic B-spline that passes through endpoints, so we assume
  // that multiplicity of first and last knot is 4 and 1 for knots between.

  QVector<qreal> newKnotVector = knotVector;
  boorNetPoints.clear();
  for (int counter = 0; counter < controlPoints.size(); ++counter)
    boorNetPoints.push_back(*controlPoints[counter]);

  // Insert every middle knot 2 times to increase its multiplicity from 1 to 3.
  const int curveDegree = 3;
  const int increaseMultiplicity = 2;

  for (int knotCounter = 4; knotCounter < newKnotVector.size() - 4;
       knotCounter += 3) {
    QHash< int, QHash<int, QPointF> > tempPoints;
    for (int counter = knotCounter - curveDegree; counter <= knotCounter;
         ++counter)
      tempPoints[counter][0] = boorNetPoints[counter];

    for (int insertCounter = 1; insertCounter <= increaseMultiplicity;
         ++insertCounter)
      for (int i = knotCounter - curveDegree + insertCounter; i < knotCounter;
           ++i) {
        double coeff = (newKnotVector[knotCounter] - newKnotVector[i]) /
            (newKnotVector[i + curveDegree - insertCounter + 1] - newKnotVector[i]);
        QPointF newPoint = (1.0 - coeff) * tempPoints[i - 1][insertCounter - 1] +
                           coeff * tempPoints[i][insertCounter - 1];
        tempPoints[i][insertCounter] = newPoint;
      }

    for (int counter = 0; counter < increaseMultiplicity; ++counter)
      newKnotVector.insert(knotCounter, newKnotVector[knotCounter]);

    // Fill new control points.
    QVector<QPointF> newBoorNetPoints;
    for (int counter = 0; counter <= knotCounter - curveDegree; ++counter)
      newBoorNetPoints.push_back(boorNetPoints[counter]);

    for (int counter = 1; counter <= increaseMultiplicity; ++counter) {
      QPointF &newP = tempPoints[knotCounter - curveDegree + counter][counter];
      newBoorNetPoints.push_back(newP);
    }

    for (int counter = -curveDegree + increaseMultiplicity + 1; counter <= -1;
         ++counter) {
      QPointF &newP = tempPoints[knotCounter + counter][increaseMultiplicity];
      newBoorNetPoints.push_back(newP);
    }

    for (int counter = increaseMultiplicity - 1; counter >= 1; --counter)
      newBoorNetPoints.push_back(tempPoints[knotCounter - 1][counter]);

    for (int counter = knotCounter - 1; counter < boorNetPoints.size(); ++counter)
      newBoorNetPoints.push_back(boorNetPoints[counter]);

    boorNetPoints = newBoorNetPoints;
  }
}

void BezierInterpolator2D::SetDistanceTolerance(double value) {
  DistanceTolerance = value;
}


