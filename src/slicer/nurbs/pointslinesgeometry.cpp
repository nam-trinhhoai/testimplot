#include "pointslinesgeometry.h"
#include <Qt3DCore/QBuffer>
#include <Qt3DCore/QAttribute>
#include <Qt3DCore/QGeometry>
#include <cstring>
#include <QDebug>
#include "QQuaternion"
#include "qmath.h"

#include "curvemodel.h"

PointsLinesGeometry::PointsLinesGeometry(Qt3DCore::QNode *parent)
    : Qt3DCore::QGeometry(parent)
{
     m_vertexBuffer = new Qt3DCore::QBuffer();
    m_indexBuffer  = new Qt3DCore::QBuffer();

    m_indexAttribute  = new Qt3DCore::QAttribute(parent);
    m_vertexAttribute = new Qt3DCore::QAttribute(parent);


    m_vertexAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
    m_vertexAttribute->setBuffer(m_vertexBuffer);
   // m_vertexAttribute->setVertexBaseType(Qt3DRender::QAttribute::VertexBaseType::Float);
    m_vertexAttribute->setVertexSize(3);
    m_vertexAttribute->setByteOffset(0);
    m_vertexAttribute->setByteStride(3 * sizeof(float));
    m_vertexAttribute->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());

    m_indexAttribute->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
    m_indexAttribute->setBuffer(m_indexBuffer);
    m_indexAttribute->setVertexBaseType(Qt3DCore::QAttribute::UnsignedShort);
    m_indexAttribute->setVertexSize(1);
    m_indexAttribute->setByteOffset(0);
    m_indexAttribute->setByteStride(0);

    addAttribute(m_vertexAttribute);
    addAttribute(m_indexAttribute);

   // m_vertexBuffer->setAccessType(Qt3DRender::QBuffer::ReadWrite);
  //  m_vertexBuffer->setUsage(Qt3DRender::QBuffer::DynamicDraw);

}

PointsLinesGeometry::~PointsLinesGeometry()
{

}

void PointsLinesGeometry::clearData()
{
    m_vertexAttribute->setCount(0);
    m_indexAttribute->setCount(0);
}

void PointsLinesGeometry::updateData(const std::vector<QVector3D>& pointsequence)
{
    if (pointsequence.size()==0)
    {
        clearData();
        return;
    }


   // qDebug()<<"PointsLinesGeometry::updateData() :"<<pointsequence.size();

    unsigned int numvertices =  int(pointsequence.size());
    unsigned int numindices = numvertices;
    QByteArray rawData;
    QByteArray rawDataIndices;

    rawData.resize(static_cast<int>(numvertices * 3 *sizeof(float)));
    float *vertices = reinterpret_cast<float *>(rawData.data());

    for (unsigned int i=0; i<numvertices; i++)
    {
        const QVector3D &p = pointsequence[i];
        vertices[i*3+0]= p.x();
        vertices[i*3+1]= p.y();
        vertices[i*3+2]= p.z();
    }

    int indsize = static_cast<int>(numindices * sizeof(ushort));
    rawDataIndices.resize(indsize);
    ushort *indices = reinterpret_cast<ushort *>(rawDataIndices.data());

    for (ushort i=0; i<numvertices; i++)
        indices[i]=i;

    m_vertexBuffer->setData(rawData);
    m_indexBuffer->setData(rawDataIndices);

    m_vertexAttribute->setCount(numvertices);
    m_indexAttribute->setCount(numindices);
  // setVertexCount(static_cast<int>(numindices));

}


