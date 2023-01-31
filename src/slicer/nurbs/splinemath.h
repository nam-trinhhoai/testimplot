#pragma once

#include <QDebug>
#include <tinynurbs/tinynurbs.h>
#include "curvemodel.h"

#include <iostream>
#include "glm/ext.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/io.hpp>

//http://www.antigrain.com/research/adaptive_bezier/index.html

class NurbsHelper
{
public:
    enum Type {Closed, Clamped};
    // Type Closed is based on https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-curve-closed.html
    // For four points, the three first must be repeated at end: points: [p1 p2 p3 p4 p1 p2 p3]
    // The knots (11) will be [0 .1 .2 .3 .4 .5 .6 .7 .8 .9  1]
    // Evaluation interval will be [.3 to .7]. E.g.  [getParameter(0,100) , getParameter(99,100)]
    // NB: The user must take care of adding (3) repeated points and report the resulting number of points to the constructor
    // NBNB: Type Closed is no longer used in this code since adding new control points while preserving shape is not straightforward.

    // Type Clamped
    // Example Clamped knots for 7 points: [0,0,0,  0, .25, .5, .75, 1,  1,1,1]
    // I.e. going from 0 to 1 with knot multiplicity 3 on endpoints and even deltas in between
    // Evaluation interval is also [0 to 1] E.g.  [getParameter(0,100) , getParameter(99,100)]

    NurbsHelper(){};

    NurbsHelper(int numPointsAtLeast4, Type type) :
        _numPoints(numPointsAtLeast4),
        _type(type),
        _minParam(getMinParameter()),
        _maxParam(getMaxParameter())
    {}

    int getNumKnots() const
    {
        return _numPoints+_dg3+1;
    }

    int getMinAllowedPoints() const
    {
        return _dg3+1; // i.e. 4
    }

    // valid index is from 1 to getNumKnots()
    float getKnot(const int index) const
    {
        const int numKnots = getNumKnots();
        if (_type==Clamped)
        {
            const int start = _dg3+1;
            const int end   =  numKnots-_dg3;
            if (index< start)   return 0;
            if (index>=end+1)   return 1;
            return normalize(index,start,end);
        }
        //   if (_type==Closed)
        return normalize(index, 1, numKnots);
    }

    float getParameter(int sampleNum, int numSamples)
    {
        float norm = float(sampleNum)/float(numSamples-1);
        return _minParam*(1-norm) + _maxParam*norm;
    }

    float getParameter(float normalized)
    {
        return _minParam*(1-normalized) + _maxParam*normalized;
    }


    float getMinParameter() const
    {
        if (_type==Clamped) return 0;
        const int numKnots = getNumKnots();
        const int start    = _dg3+1;
        return normalize(start, 1, numKnots);
    }

    float getMaxParameter() const
    {
        if (_type==Clamped) return 1;
        const int numKnots = getNumKnots();
        const int end      = numKnots-_dg3;
        return normalize(end, 1, numKnots);
    }

private:
    int   _dg3 = 3;  // only degree 3 is supported
    int   _numPoints;
    Type  _type;
    float _minParam;
    float _maxParam;

    // maps val=min to 0 and val=max to 1 and linearly in between
    static float normalize(float val, float min0, float max1)
    {
        return (val-min0)/(max1-min0);
    }
};


class BezierInterpolator2D {
public:
	BezierInterpolator2D();

  // InterpolateBezier - interpolates points with bezier curve.
  // Algorithm is based on article "Adaptive Subdivision of Bezier Curves" by
  // Maxim Shemanarev.
  // http://www.antigrain.com/research/adaptive_bezier/index.html
  void InterpolateBezier(double x1, double y1, double x2, double y2, double x3,
                         double y3, double x4, double y4,
						 QVector<QPointF> &interpolatedPoints,
                         unsigned level = 0) const;

  void InterpolateBezier(const QPointF &p1, const QPointF &p2,
                         const QPointF &p3, const QPointF &p4,
						 QVector<QPointF> &interpolatedPoints,
                         unsigned level = 0) const;


  // CalculateBoorNet - inserts new control points with de Boor algorithm for
  // transformation of B-spline into composite Bezier curve.
  void CalculateBoorNet(const QVector<QPointF*> &controlPoints,
                        const QVector<qreal> &knotVector,
						QVector<QPointF> &boorNetPoints) const;

  void SetDistanceTolerance(double value);

private:
  // Casteljau algorithm (interpolating Bezier curve) parameters.
  static const unsigned curveRecursionLimit;
  static const double curveCollinearityEpsilon;
  static const double curveAngleToleranceEpsilon;
  static const double AngleTolerance;
  static const double CuspLimit;

  double DistanceTolerance;
};

class BezierMath
{
public:
	BezierMath();
	void setOpen(bool isopen){m_isOpen=isopen;}
	void setPoints(CurveModel* points){m_points=points; update();}

	void fillKnotVector();
	void interpolateCurve();

	int getNbCtrlPts(){
		return m_interpolatedPoints.size();
	}

	QPointF getPosition(int index)
	{
		return m_interpolatedPoints[index];
	}

	  void sampletheSpline(CurveModel* result, int numsamples=100);
	    void addShapepreservingPoint(float normalizedparameter); // normalizedparameter is in range [0,1] covering the whole curve
	    glm::vec3 getPosition(float normalizedparameter);        // normalizedparameter is in range [0,1] covering the whole curve
	    glm::vec3 getTangent(float normalizedparameter);         // normalizedparameter is in range [0,1] covering the whole curve
	    bool isValid(); // true if spline is defined so that e.g. sampletheSpline/getPosition/getTangent can be called
	    const tinynurbs::Curve3f& getTinycurve(){return m_tinycurve;}

	    float todistance(float normalizedparameter);
	    float getLength(){return m_curvelength;}


	    QPointF bezierCurveByCasteljau(float u);
	    QPointF bezierCurveByCasteljauRec(QVector<QPointF> in_pts, float i);

private:
	 qreal NN1( int  i, qreal u);
	 qreal NN2( int  i, qreal u);
	 qreal NN3( int  i, qreal u);
	 qreal NN(  int  k,  int  i, qreal u);

	void update();
	bool m_isOpen = true;


	 CurveModel* m_points = nullptr;
	 tinynurbs::Curve3f m_tinycurve;
	     NurbsHelper m_generator;
	     bool m_updateKnots=true;


	     void makecurvelengthtable(int steps = 100);
	     std::vector<float> m_curvelengthtable; // maps from curvelength to parametervalue
	     float m_curvelength = 0;

	     QVector<qreal> m_knotVector;
	     QVector<QPointF> m_boorNetPoints;
	     QVector<QPointF> m_polygon;
	     QVector<QPointF> m_interpolatedPoints;
	     QVector<QPointF*> m_controlPoints;

	     BezierInterpolator2D m_bezierInterpolator;
	     bool m_IsClosedCurved= false;


};

class SplineMath
{
public:
    SplineMath();

    void setOpen(bool isopen){m_isOpen=isopen;};
    void setPoints(CurveModel* points){m_points=points; update();}
    void sampletheSpline(CurveModel* result, int numsamples=100);
    void addShapepreservingPoint(float normalizedparameter); // normalizedparameter is in range [0,1] covering the whole curve
    glm::vec3 getPosition(float normalizedparameter);        // normalizedparameter is in range [0,1] covering the whole curve
    glm::vec3 getTangent(float normalizedparameter);         // normalizedparameter is in range [0,1] covering the whole curve
    bool isValid(); // true if spline is defined so that e.g. sampletheSpline/getPosition/getTangent can be called
    const tinynurbs::Curve3f& getTinycurve(){return m_tinycurve;}

    float todistance(float normalizedparameter);
    float getLength(){return m_curvelength;}

private:
    void update();
    bool m_isOpen = true;   // true if spline is an open curve, false if it is a closed curve
    CurveModel* m_points = nullptr;
    tinynurbs::Curve3f m_tinycurve;
    NurbsHelper m_generator;
    bool m_updateKnots=true;


    void makecurvelengthtable(int steps = 100);
    std::vector<float> m_curvelengthtable; // maps from curvelength to parametervalue
    float m_curvelength = 0;
};


class CentripetalCatmullRom
{
public:
    glm::vec3 getPosition(float normalizedparameter){return glm::vec3();};
    glm::vec3 getTangent(float normalizedparameter) {return glm::vec3();};


    bool isValid(){return true;};
    void setOpen(bool isopen){};

    void setPoints(CurveModel* points)
    {
        m_points=points;
        update();
    };

    void update()
    {

    }

    glm::vec3 toglm(QVector3D& v){return glm::vec3(v.x(),v.y(),v.z());}


    void sampletheSpline(CurveModel* result, int samplespersegment=10)
    {
        int numpts = m_points->getSize();
        if (numpts<2) return;

        for (int i=0;i<numpts-1;i++)
        {
            int lastone = (i==numpts-2)?1:0; // The for loop does not include sampling the endpoint, so add that for the absolute final sample
            for (int j=0; j<samplespersegment+lastone; j++)
            {
                float step  = 1.0f/samplespersegment;
                float param = step*float(j);

               // glm::vec3 pt = glm::catmullRom(
                glm::vec3 pt = catmull2(
                    toglm(m_points->getPosition(std::max(i-1,0))),
                    toglm(m_points->getPosition(i-0)),
                    toglm(m_points->getPosition(i+1)),
                    toglm(m_points->getPosition(std::min(i+2,numpts-1))),
                    param);
                result->insertBackSilent(QVector3D(pt.x,pt.y,pt.z));
            }
        }

        result->emitModelUpdated(false);
    }

    // based on https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline
    static glm::vec3 catmull2(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, float t, float alpha=1)
    {
        if (p0==p1) p0=p1-(p2-p1); // if two first are equal then extrapolate first based on p1 and p2, i.e. mirror p2 around p1 to create p0
        if (p2==p3) p3=p2-(p2-p1); // same as above but for two last

        float t0 = 0.0f;
        float t1 = getT( t0, alpha, p0, p1 );
        float t2 = getT( t1, alpha, p1, p2 );
        float t3 = getT( t2, alpha, p2, p3 );

        t = t1*(1-t)+ t2*t;
        glm::vec3 A1 = ( t1-t )/( t1-t0 )*p0 + ( t-t0 )/( t1-t0 )*p1;
        glm::vec3 A2 = ( t2-t )/( t2-t1 )*p1 + ( t-t1 )/( t2-t1 )*p2;
        glm::vec3 A3 = ( t3-t )/( t3-t2 )*p2 + ( t-t2 )/( t3-t2 )*p3;
        glm::vec3 B1 = ( t2-t )/( t2-t0 )*A1 + ( t-t0 )/( t2-t0 )*A2;
        glm::vec3 B2 = ( t3-t )/( t3-t1 )*A2 + ( t-t1 )/( t3-t1 )*A3;
        glm::vec3 C  = ( t2-t )/( t2-t1 )*B1 + ( t-t1 )/( t2-t1 )*B2;
        return C;
    }

    static float getT(float t, float alpha, const glm::vec3& p0, const glm::vec3& p1)
    {
        glm::vec3 d  = p1 - p0;
        float a = glm::dot(d, d);
        float b = std::pow(a, alpha*.5f);

        return (b + t);
    }


    CurveModel* m_points = nullptr;
};
