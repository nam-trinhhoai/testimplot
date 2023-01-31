#include "volumeboundingmesh.h"
#include <Qt3DCore/QBuffer>
#include <Qt3DCore/QAttribute>
#include <Qt3DCore/QGeometry>
#include <cstring>

namespace {
// 8 sets of 3 lines of 2 vertices each
const int VertexCount = 8 * 3 * 2;
}

VolumeBoundingMesh::VolumeBoundingMesh(Qt3DCore::QNode *parent)
    : Qt3DRender::QGeometryRenderer(parent)
    , m_geometry(new Qt3DCore::QGeometry())
    , m_positionAttribute(new Qt3DCore::QAttribute())
    , m_vertexBuffer(new Qt3DCore::QBuffer())
{
   // m_vertexBuffer->setType(Qt3DCore::QBuffer::VertexBuffer);
    m_positionAttribute->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
    m_positionAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
   // m_positionAttribute->setDataType(Qt3DCore::QAttribute::Float);
   // m_positionAttribute->setDataSize(3);
    m_positionAttribute->setVertexSize(3);
    m_positionAttribute->setCount(VertexCount);
    m_positionAttribute->setBuffer(m_vertexBuffer);
    m_geometry->addAttribute(m_positionAttribute);

    updateGeometry();

    setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);
    setGeometry(m_geometry);

    QObject::connect(this, &VolumeBoundingMesh::dimensionsChanged,
                     this, &VolumeBoundingMesh::updateGeometry);
}

VolumeBoundingMesh::~VolumeBoundingMesh()
{
}

void VolumeBoundingMesh::setDimensions(const QVector3D dimensions)
{
    if (m_dimensions == dimensions)
        return;
    m_dimensions = dimensions;
    emit dimensionsChanged();
}

QVector3D VolumeBoundingMesh::dimensions() const
{
    return m_dimensions;
}

void VolumeBoundingMesh::updateGeometry()
{
    QByteArray rawData;
    rawData.resize(VertexCount * sizeof(QVector3D));
    QVector3D *vertices = reinterpret_cast<QVector3D *>(rawData.data());



    const QVector3D extents = m_dimensions;// * 0.5f;
    const QVector3D corners[] {
        QVector3D(0, extents.y(), extents.z()), // Top NW
        QVector3D(extents.x(), extents.y(), extents.z()), // Top NE
        QVector3D(0, extents.y(), 0), // Top SW
        QVector3D(extents.x(), extents.y(), 0), // Top SE
        QVector3D(0, 0, extents.z()), // Bottom NW
        QVector3D(extents.x(), 0, extents.z()), // Bottom NE
        QVector3D(0, 0, 0), // Bottom SW
        QVector3D(extents.x(), 0, 0), // Bottom SE
    };


    const float len = 0.2f;
    const int indices[] = {
        1, 2, 4,
        0, 3, 5,
        0, 3, 6,
        1, 2, 7,
        5, 6, 0,
        4, 7, 1,
        4, 7, 2,
        5, 6, 3
    };

    for (int i = 0; i < 8; ++i) {
        const QVector3D baseCorner = corners[i];
        vertices[i * 6] = baseCorner;
        vertices[i * 6 + 2] = baseCorner;
        vertices[i * 6 + 4] = baseCorner;
        vertices[i * 6 + 1] = baseCorner + (corners[indices[i * 3]] - baseCorner) * len;
        vertices[i * 6 + 3] = baseCorner + (corners[indices[i * 3 + 1]] - baseCorner) * len;
        vertices[i * 6 + 5] = baseCorner + (corners[indices[i * 3 + 2]] - baseCorner) * len;
    }

    m_vertexBuffer->setData(rawData);
}
