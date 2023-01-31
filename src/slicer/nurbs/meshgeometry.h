#pragma once

#include <Qt3DCore/QBuffer>
#include <Qt3DRender/QGeometryRenderer>
#include <vector>


/*
namespace Qt3DRender {
    class QAttribute;
    class QGeometry;
}
*/

//using namespace std;


class MeshGeometry : public Qt3DRender::QGeometryRenderer
{
    Q_OBJECT

public:
    explicit MeshGeometry(Qt3DCore::QNode *parent = nullptr);
    ~MeshGeometry();

    void setEmptyData();
    void uploadMeshData(std::vector<QVector3D>& vertices, std::vector<int>& indices, std::vector<QVector3D>& normals);//, std::vector<QVector3D>& colors);
    void setRenderPoints(bool pointsNotTriangles);
    void setRenderLines(bool linesNotTriangles);

private:
    Qt3DCore::QGeometry  *m_geometry;
    Qt3DCore::QBuffer    *m_vertexBuffer;
    Qt3DCore::QBuffer    *m_indexBuffer;
    Qt3DCore::QAttribute *m_positionAttribute;
    Qt3DCore::QAttribute *m_normalAttribute;
   // Qt3DRender::QAttribute *m_colorAttribute;
    Qt3DCore::QAttribute *m_indexAttribute;

    uint m_valuesPerVertex;
};


class MeshGeometry2 : public Qt3DRender::QGeometryRenderer
{
    Q_OBJECT

public:
    explicit MeshGeometry2(Qt3DCore::QNode *parent = nullptr);
    ~MeshGeometry2();

    void setEmptyData();
    void uploadMeshData(std::vector<QVector3D>& vertices, std::vector<int>& indices, std::vector<QVector3D>& normals);//, std::vector<QVector3D>& colors);

    void setRenderLines(bool linesNotTriangles);
    void computeNormals();

private:
    Qt3DCore::QGeometry  *m_geometry;

    Qt3DCore::QBuffer    *m_vertexBuffer;
    Qt3DCore::QBuffer    *m_normalBuffer;
    Qt3DCore::QBuffer    *m_indexBuffer;

    Qt3DCore::QAttribute *m_positionAttribute;
    Qt3DCore::QAttribute *m_normalAttribute;
    Qt3DCore::QAttribute *m_indexAttribute;

    uint m_valuesPerVertex;
};
