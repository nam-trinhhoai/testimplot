#ifndef VOLUMEBOUNDINGMESH_H
#define VOLUMEBOUNDINGMESH_H

#include <Qt3DRender/QGeometryRenderer>
#include <QVector3D>

namespace Qt3DCore {
class QBuffer;
class QAttribute;
class QGeometry;
}

// Draws wireframe of (the corners of) a cube
class VolumeBoundingMesh : public Qt3DRender::QGeometryRenderer
{
    Q_OBJECT
    Q_PROPERTY(QVector3D dimensions READ dimensions WRITE setDimensions NOTIFY dimensionsChanged)
public:
    explicit VolumeBoundingMesh(Qt3DCore::QNode *parent = nullptr);
    ~VolumeBoundingMesh();

    void setDimensions(const QVector3D dimensions);
    QVector3D dimensions() const;

signals:
    void dimensionsChanged();

private:
    void updateGeometry();

    QVector3D m_dimensions;
    Qt3DCore::QBuffer *m_vertexBuffer;
    Qt3DCore::QAttribute *m_positionAttribute;
    Qt3DCore::QGeometry *m_geometry;
};

#endif // VOLUMEBOUNDINGMESH_H
