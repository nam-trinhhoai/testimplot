#ifndef MATHUTILS_H
#define MATHUTILS_H

#include <QObject>
#include <QQuaternion>

class LineAABBIntersectionResult
{
    Q_GADGET
    Q_PROPERTY(QVector3D minPoint READ minPoint CONSTANT)
    Q_PROPERTY(QVector3D maxPoint READ maxPoint CONSTANT)
public:
    QVector3D minPoint() const { return m_minPoint; }
    QVector3D maxPoint() const { return m_maxPoint; }

    QVector3D m_minPoint;
    QVector3D m_maxPoint;
};

class MathUtils : public QObject
{
    Q_OBJECT
public:
    explicit MathUtils(QObject *parent = nullptr);

    Q_INVOKABLE QQuaternion multQuaternions(const QQuaternion &a, const QQuaternion &b) const;
    Q_INVOKABLE QQuaternion rotationTo(const QVector3D &from, const QVector3D &to);
    Q_INVOKABLE QVector3D rotatedVector(const QQuaternion &a, const QVector3D &v) const;
    Q_INVOKABLE QVector3D extractTranslationFromMatrix(const QMatrix4x4 &m);
    Q_INVOKABLE LineAABBIntersectionResult lineAABBIntersection(const QVector3D &lineOrig, const QVector3D &lineAxis,
                                                                const QVector3D &aabbMinCorner, const QVector3D &aabbMaxCorner);
};

Q_DECLARE_METATYPE(LineAABBIntersectionResult);

#endif // MATHUTILS_H
