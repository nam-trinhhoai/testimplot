#ifndef Qt3DHelpers_H
#define Qt3DHelpers_H
#include <QString>
#include <QVector3D>
#include <QVector2D>
#include <QColor>
#include <QObject>
#include <Qt3DRender/QObjectPicker>
#include <Qt3DRender/QPickingSettings>

class QWindow;

namespace Qt3DRender
{
	class QEffect;
	class QCamera;
}

namespace Qt3DCore {
	class QEntity;
}

class Qt3DHelpers : public QObject
{
    Q_OBJECT
public:

	//static void view_generated_mesh();
	static Qt3DRender::QEffect * generateImageEffect(const QString& fragPath, const QString & vertPath);

	static QVector3D computeCameraRay(QWindow *parent,Qt3DRender::QCamera *camera,const QVector2D &position) ;
	static bool intersectPlane(QVector3D & result,const QVector3D &planePoint,const QVector3D &planeNormal,const QVector3D &linePoint,const QVector3D &lineDirection);

    static QVector3D computePerpendicularVector(QVector3D vec);

    static qreal mix(qreal a0, qreal a1, qreal a2, qreal a3, qreal x);
    static QVector3D mix(QVector3D p0, QVector3D p1, QVector3D p2, QVector3D p3, qreal x);

    static Qt3DCore::QEntity* drawNormals(const QVector<QVector3D>& positionsVec, const QColor& color, Qt3DCore::QEntity *_rootEntity, int lineWidth);
	static Qt3DCore::QEntity* drawLine(const QVector3D& start, const QVector3D& end, const QColor& color, Qt3DCore::QEntity *_rootEntity);
	static Qt3DCore::QEntity* drawLines(const QVector<QVector3D>& positionsVec, const QColor& color, Qt3DCore::QEntity *_rootEntity, int lineWidth);
    static Qt3DCore::QEntity* drawExtruders(const QVector<QVector3D>& positionsVec, const QVector<QVector3D>& colorVec, const QVector<float>& widthVec, Qt3DCore::QEntity *_rootEntity, Qt3DRender::QCamera *camera,QString nameWell, bool modeFilaire, bool showNormals);

    static Qt3DCore::QEntity* drawLog(const QVector<QVector3D>& positionsVec,const QVector<float>& widthVec,const QColor& color, Qt3DCore::QEntity *_rootEntity, int lineWidth);


    //static Qt3DCore::QEntity* drawLog(const QVector<QVector3D>& positionsVec, Qt3DCore::QEntity *_rootEntity, Qt3DRender::QCamera *camera,QString nameWell, bool modeFilaire, bool showNormals);
    static QVector3D screenToWorld( QMatrix4x4 viewMatrix, QMatrix4x4 projecMatrix,QVector2D pos2D,int screenWidth,int screenHeight);
    static QVector2D worldToScreen(QVector3D pos, QMatrix4x4 viewMatrix, QMatrix4x4 projecMatrix,int screenWidth,int screenHeight);

    static Qt3DCore::QEntity* loadObj(const char* filename,Qt3DCore::QEntity* _rootEntity);
    static void writeObj(const char* filename, std::vector<QVector3D> vertices,std::vector<QVector3D> normals, std::vector<int> indices);
    void writeObjWidthUV(const char* filename, std::vector<QVector3D> vertices,std::vector<QVector3D> normals,std::vector<QVector2D> uvs, std::vector<int> indices);

	~Qt3DHelpers(){};
private:
	Qt3DHelpers(){}

  //  static Qt3DRender::QObjectPicker *sPicker;
  //  static Qt3DRender::QPickingSettings *sPickingSettings;
    static Qt3DRender::QCamera *sCamera;

};

#endif
