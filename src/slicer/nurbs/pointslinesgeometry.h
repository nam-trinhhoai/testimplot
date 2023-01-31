#pragma once

//#include <Qt3DRender/QGeometryRenderer>

#include <Qt3DCore/QGeometry>
//#include <QVector3D>
#include <vector>

namespace Qt3DRender {
    class QBuffer;
    class QAttribute;
    class QGeometry;
}

class CurveModel;

class PointsLinesGeometry : public Qt3DCore::QGeometry
{
    Q_OBJECT

public:
    explicit PointsLinesGeometry(Qt3DCore::QNode* parent = nullptr);
    ~PointsLinesGeometry();

    void updateData(const std::vector<QVector3D>& pointsequence);
    void clearData();

private:
    void setData();
   
    //  The graphics
    Qt3DCore::QBuffer*     m_vertexBuffer;
    Qt3DCore::QAttribute*  m_vertexAttribute;
    Qt3DCore::QBuffer*     m_indexBuffer;
    Qt3DCore::QAttribute*  m_indexAttribute;
    Qt3DCore::QGeometry*   m_geometry;
};
