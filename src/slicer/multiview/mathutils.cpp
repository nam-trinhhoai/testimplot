#include "mathutils.h"
#include <QMatrix4x4>

MathUtils::MathUtils(QObject *parent)
    : QObject(parent)
{
}

QQuaternion MathUtils::multQuaternions(const QQuaternion &a, const QQuaternion &b) const
{
    return a * b;
}

QQuaternion MathUtils::rotationTo(const QVector3D &from, const QVector3D &to)
{
    return QQuaternion::rotationTo(from, to);
}

QVector3D MathUtils::rotatedVector(const QQuaternion &a, const QVector3D &v) const
{
    return a.rotatedVector(v);
}

QVector3D MathUtils::extractTranslationFromMatrix(const QMatrix4x4 &m)
{
    return m.column(3).toVector3D();
}

LineAABBIntersectionResult MathUtils::lineAABBIntersection(const QVector3D &lineOrig, const QVector3D &lineAxis,
                                                           const QVector3D &aabbMinCorner, const QVector3D &aabbMaxCorner)
{
    double t1 = (aabbMinCorner.x() - lineOrig.x()) / lineAxis.x();
    double t2 = (aabbMaxCorner.x() - lineOrig.x()) / lineAxis.x();
    double t3 = (aabbMinCorner.y() - lineOrig.y()) / lineAxis.y();
    double t4 = (aabbMaxCorner.y() - lineOrig.y()) / lineAxis.y();
    double t5 = (aabbMinCorner.z() - lineOrig.z()) / lineAxis.z();
    double t6 = (aabbMaxCorner.z() - lineOrig.z()) / lineAxis.z();

    // tmin and tmax are the two values that define the two points that intersect the volume's bounding box
    double tmin = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
    double tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));

    LineAABBIntersectionResult result;
    result.m_minPoint = lineOrig + lineAxis * tmin;
    result.m_maxPoint = lineOrig + lineAxis * tmax;
    return result;
}

