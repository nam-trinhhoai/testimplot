
#include "Viewer3D.h"

Viewer3D::Viewer3D(QWidget *parent) 
{
//    setAttribute(Qt::WA_DeleteOnClose);
//    m_moveStartPoint.setX(-1);

//    m_view = new Qt3DExtras::Qt3DWindow();
/**
    m_view->installEventFilter(this);

    m_view->defaultFrameGraph()->setClearColor(QColor(QRgb(0x4d4d4f)));

    QWidget *container = QWidget::createWindowContainer(m_view);
    QSize screenSize = m_view->screen()->size();
    container->setMinimumSize(QSize(200, 100));
    container->setMaximumSize(screenSize);

    QHBoxLayout *hLayout = new QHBoxLayout(this);
    QVBoxLayout *vLayout = new QVBoxLayout();
    hLayout->addWidget(container, 1);

    setWindowTitle(QStringLiteral("Mesh Viewer"));

    // Root entity
    m_rootEntity = new Qt3DCore::QEntity();

    // Scene modifier
    m_sceneModifier = new SceneModifier(m_rootEntity);

    // Window geometry
    resize(parent->geometry().width() * 0.8, parent->geometry().height() * 0.8);
    move(parent->geometry().center() - QPoint(width() / 2, height() / 2));

    // Camera
    Qt3DRender::QCamera *cameraEntity = m_view->camera();

    //cameraEntity->lens()->setPerspectiveProjection(22.5f, m_view->width()/m_view->height(), 0.01f, 1000.0f);
    cameraEntity->setPosition(QVector3D(0, 0, 500.0f));
    cameraEntity->setUpVector(QVector3D(0, 1, 0));
    cameraEntity->setViewCenter(QVector3D(0, 0, 0));
    cameraEntity->transform()->setScale(1.f);

    // Set root object of the scene
    m_view->setRootEntity(m_rootEntity);
*/
}

Viewer3D::~Viewer3D()
{}

/**
bool Viewer3D::eventFilter(QObject *obj, QEvent *ev)
{
    if (ev->type() == QEvent::Wheel)
    {
        wheelEvent(dynamic_cast<QWheelEvent*>(ev));
        return true;
    }
    else if (ev->type() == QEvent::MouseButtonPress)
    {
        mousePressEvent(dynamic_cast<QMouseEvent*>(ev));
        return true;
    }
    else if (ev->type() == QEvent::MouseMove)
    {
        mouseMoveEvent(dynamic_cast<QMouseEvent*>(ev));
        return true;
    }
    else if (ev->type() == QEvent::MouseButtonRelease)
    {
        mouseReleaseEvent(dynamic_cast<QMouseEvent*>(ev));
        return true;
    }

    return QObject::eventFilter(obj, ev);
}

void Viewer3D::wheelEvent(QWheelEvent *we)
{
    Qt3DCore::QTransform* transform = m_view->camera()->transform();

    float scale = transform->scale();
    QPoint delta = we->angleDelta();
    float zoom_distance = scale * static_cast<float>(delta.y()) / 500.f;
    scale -= zoom_distance;
    scale = std::min(10.0000f, scale);
    scale = std::max(0.001f, scale);
    transform->setScale(scale);
}

void Viewer3D::mousePressEvent(QMouseEvent *ev)
{
    if (ev->button() == Qt::LeftButton)
    {
        m_moveStartPoint = ev->pos();
        m_cameraMatrix = m_view->camera()->transform()->matrix();
    }
}

void Viewer3D::mouseMoveEvent(QMouseEvent *ev)
{
    if (m_moveStartPoint.x() > -1)
    {
        QPoint delta = ev->pos() - m_moveStartPoint;
        float angle = static_cast<float>(QPoint::dotProduct(delta, delta)) / 100.f;
        QVector3D axis = QVector3D(delta.y(), delta.x(), 0);

        QMatrix4x4 rotationMatrix = Qt3DCore::QTransform::rotateAround(-m_view->camera()->position(), angle, axis);

        QMatrix4x4 matrix = rotationMatrix * m_cameraMatrix;
        m_view->camera()->transform()->setMatrix(matrix);
    }
}

void Viewer3D::mouseReleaseEvent(QMouseEvent *ev)
{
    if (m_moveStartPoint.x() > -1)
    {
        m_moveStartPoint.setX(-1);
        m_cameraMatrix = m_view->camera()->transform()->matrix();
    }
}


// #########  Scene modifier class h #########
class SceneModifier : public QObject
{
    Q_OBJECT

    public:
        SceneModifier(Qt3DCore::QEntity* rootEntity);
        void addTriangleMeshCustomMaterial(QString name, const std::vector<Import3d::Triangle>& meshVector);

    private:
        Qt3DCore::QEntity* m_rootEntity;
};

// #########  Scene modifier class cpp #########
#include "SceneModifier.h"
#include "TriangleMeshRenderer.h"
#include "MaterialWireFrame.h"

SceneModifier::SceneModifier(Qt3DCore::QEntity* rootEntity) :
    m_rootEntity(rootEntity),
    QObject(rootEntity)
{
}

void SceneModifier::addTriangleMeshCustomMaterial(QString name, const std::vector<Import3d::Triangle>& meshVector)
{
    if (!m_rootEntity)
    {
        return;
    }

    // Mesh entity
    Qt3DCore::QEntity *triangleMeshEntity = new Qt3DCore::QEntity(m_rootEntity);
    triangleMeshEntity->setObjectName(QStringLiteral("customMeshEntity"));

    TriangleMeshRenderer *triangleMeshRenderer = new TriangleMeshRenderer(meshVector);
    MaterialWireFrame* materialWireFrame = new MaterialWireFrame();
    Qt3DCore::QTransform *transform = new Qt3DCore::QTransform;
    transform->setScale(1.f);

    triangleMeshEntity->addComponent(triangleMeshRenderer);
    triangleMeshEntity->addComponent(transform);
    triangleMeshEntity->addComponent(materialWireFrame);

    //emit meshAdded(name, triangleMeshEntity);
}

// ######### Point and Triangle structs #########
struct Point
{
    QVector3D p; //point x, y, z
    QVector3D c; //color red, green, blue

    Point() {}

    Point(float xp, float yp, float zp)
    {
        p = QVector3D(xp, yp, zp);
        c = QVector3D(0, 0, 0);
    }
    Point(QVector3D pos, unsigned char r, unsigned char g, unsigned char b)
    {
        p = pos;
        c = QVector3D(static_cast<float>(r) / 255.f,
                      static_cast<float>(g) / 255.f,
                      static_cast<float>(b) / 255.f);
    }
};

struct Triangle 
{
    Point vertices[3];

    Triangle()
    {
    }

    Triangle(Point p1, Point p2, Point p3)
    {
        vertices[0] = p1;
        vertices[1] = p2;
        vertices[2] = p3;
    }

};


// ######### TriangleMeshRenderer class h #########
class TriangleMeshRenderer : public Qt3DRender::QGeometryRenderer
{
    Q_OBJECT
public:
    explicit TriangleMeshRenderer(const std::vector<Import3d::Triangle>& meshVector, Qt3DCore::QNode *parent = 0);
    ~TriangleMeshRenderer();
};


class TriangleMeshGeometry : public Qt3DRender::QGeometry
{
    Q_OBJECT
public:
    TriangleMeshGeometry(const std::vector<Import3d::Triangle>& meshVector, TriangleMeshRenderer *parent);
};


// ######### TriangleMeshRenderer class cpp #########
TriangleMeshRenderer::TriangleMeshRenderer(const std::vector<Import3d::Triangle>& meshVector, QNode *parent)
    : Qt3DRender::QGeometryRenderer(parent)
{
    setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);
    setGeometry(new TriangleMeshGeometry(meshVector, this));
}

TriangleMeshRenderer::~TriangleMeshRenderer()
{
}

TriangleMeshGeometry::TriangleMeshGeometry(const std::vector<Import3d::Triangle>& meshVector, TriangleMeshRenderer *parent)
    : Qt3DRender::QGeometry(parent)
{
    Qt3DRender::QBuffer *vertexDataBuffer = new Qt3DRender::QBuffer(Qt3DRender::QBuffer::VertexBuffer, this);
    Qt3DRender::QBuffer *indexDataBuffer = new Qt3DRender::QBuffer(Qt3DRender::QBuffer::IndexBuffer, this);

    // Vertexbuffer
    QByteArray vertexBufferData;
    // Buffer size = triangle count * 3 * (3 + 3 + 3), 3 vertices per trinalge, each 3 floats for vertex position x,y,z, 3 floats normal and 3 floats color
    int bytesPerVertex = 9 * sizeof(float);
    int bytesPerTriangle = 3 * bytesPerVertex;
    vertexBufferData.resize(static_cast<int>(meshVector.size()) * bytesPerTriangle);
    char* pByte = vertexBufferData.data();
    int i = 0;
    // Indexbuffer
    QByteArray indexBufferData;
    indexBufferData.resize(static_cast<int>(meshVector.size()) * 3 * sizeof(uint));
    uint* rawIndexArray = reinterpret_cast<uint*>(indexBufferData.data());
    int idx = 0;

    for (int n = 0; n < meshVector.size(); ++n)
    {
        QVector3D nt = QVector3D::normal(meshVector[n].vertices[0].p, meshVector[n].vertices[1].p, meshVector[n].vertices[2].p); 

        for (int v = 0; v < 3; ++v)
        {
            // Vertex
            *reinterpret_cast<float*>(pByte) = meshVector[n].vertices[v].p.x(); pByte += 4;
            *reinterpret_cast<float*>(pByte) = meshVector[n].vertices[v].p.y(); pByte += 4;
            *reinterpret_cast<float*>(pByte) = meshVector[n].vertices[v].p.z(); pByte += 4;
            // Normal
            *reinterpret_cast<float*>(pByte) = nt.x(); pByte += 4;
            *reinterpret_cast<float*>(pByte) = nt.y(); pByte += 4;
            *reinterpret_cast<float*>(pByte) = nt.z(); pByte += 4;
            // Color
            *reinterpret_cast<float*>(pByte) = meshVector[n].vertices[v].c.x(); pByte += 4;
            *reinterpret_cast<float*>(pByte) = meshVector[n].vertices[v].c.y(); pByte += 4;
            *reinterpret_cast<float*>(pByte) = meshVector[n].vertices[v].c.z(); pByte += 4;

            // Index
            rawIndexArray[idx] = static_cast<uint>(idx++);
        }
    }

    vertexDataBuffer->setData(vertexBufferData);
    indexDataBuffer->setData(indexBufferData);

    // Attributes
    Qt3DRender::QAttribute *positionAttribute = new Qt3DRender::QAttribute();
    positionAttribute->setAttributeType(Qt3DRender::QAttribute::VertexAttribute);
    positionAttribute->setBuffer(vertexDataBuffer);
    positionAttribute->setDataType(Qt3DRender::QAttribute::Float);
    positionAttribute->setDataSize(3);
    positionAttribute->setByteOffset(0);
    positionAttribute->setByteStride(bytesPerVertex);
    positionAttribute->setCount(3 * static_cast<int>(meshVector.size()));
    positionAttribute->setName(Qt3DRender::QAttribute::defaultPositionAttributeName());

    Qt3DRender::QAttribute *normalAttribute = new Qt3DRender::QAttribute();
    normalAttribute->setAttributeType(Qt3DRender::QAttribute::VertexAttribute);
    normalAttribute->setBuffer(vertexDataBuffer);
    normalAttribute->setDataType(Qt3DRender::QAttribute::Float);
    normalAttribute->setDataSize(3);
    normalAttribute->setByteOffset(3 * sizeof(float));
    normalAttribute->setByteStride(bytesPerVertex);
    normalAttribute->setCount(3 * static_cast<int>(meshVector.size()));
    normalAttribute->setName(Qt3DRender::QAttribute::defaultNormalAttributeName());

    Qt3DRender::QAttribute *colorAttribute = new Qt3DRender::QAttribute();
    colorAttribute->setAttributeType(Qt3DRender::QAttribute::VertexAttribute);
    colorAttribute->setBuffer(vertexDataBuffer);
    colorAttribute->setDataType(Qt3DRender::QAttribute::Float);
    colorAttribute->setDataSize(3);
    colorAttribute->setByteOffset(6 * sizeof(float));
    colorAttribute->setByteStride(bytesPerVertex);
    colorAttribute->setCount(3 * static_cast<int>(meshVector.size()));
    colorAttribute->setName(Qt3DRender::QAttribute::defaultColorAttributeName());

    Qt3DRender::QAttribute *indexAttribute = new Qt3DRender::QAttribute();
    indexAttribute->setAttributeType(Qt3DRender::QAttribute::IndexAttribute);
    indexAttribute->setBuffer(indexDataBuffer);
    indexAttribute->setDataType(Qt3DRender::QAttribute::UnsignedInt);
    indexAttribute->setDataSize(1);
    indexAttribute->setByteOffset(0);
    indexAttribute->setByteStride(0);
    indexAttribute->setCount(3 * static_cast<int>(meshVector.size()));

    addAttribute(positionAttribute);
    addAttribute(normalAttribute);
    addAttribute(colorAttribute);
    addAttribute(indexAttribute);

    parent->setGeometry(this);
}
*/
