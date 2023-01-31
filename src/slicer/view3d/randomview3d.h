#ifndef RANDOMVIEW2D_H
#define RANDOMVIEW2D_H

#include <QObject>
#include <QEntity>
#include <QVector3D>
#include <QVector>
#include <QPointer>

#include <Qt3DExtras/QTextureMaterial>
#include <Qt3DCore/QBuffer>
#include <Qt3DRender/QGeometryRenderer>

#include <Qt3DRender/QLayer>
#include <Qt3DCore/QTransform>
#include "cudaimagetexture.h"
#include "randomdataset.h"

#include <QPickingSettings>
#include <QPickEvent>
#include <QPickTriangleEvent>

class RandomDataset;
class RandomTexDataset;
class RandomLineView;
class GraphEditor_LineShape;

class CustomGeometry : public Qt3DRender::QGeometryRenderer
{
    Q_OBJECT

public:
    explicit CustomGeometry(Qt3DCore::QNode *parent = nullptr);
    ~CustomGeometry();

    void setEmptyData();
    void uploadMeshData(std::vector<QVector3D>& vertices, std::vector<int>& indices,std::vector<QVector2D>& uvs);

  //  void uploadMeshData(std::vector<QVector3D>& vertices, std::vector<int>& indices, std::vector<QVector3D>& normals, std::vector<QVector3D>& colors,std::vector<QVector2D>& uvs);
    void setRenderPoints(bool pointsNotTriangles);
    void setRenderLines(bool linesNotTriangles);

private:
    Qt3DCore::QGeometry  *m_geometry;
    Qt3DCore::QBuffer    *m_vertexBuffer;
    Qt3DCore::QBuffer    *m_indexBuffer;
    Qt3DCore::QAttribute *m_positionAttribute;
   // Qt3DRender::QAttribute *m_normalAttribute;
   // Qt3DRender::QAttribute *m_colorAttribute;
    Qt3DCore::QAttribute *m_indexAttribute;

    Qt3DCore::QBuffer *m_texBuffer;
    Qt3DCore::QAttribute *m_textureAttribute;

    uint m_valuesPerVertex;
};


class RandomView3D: public Qt3DCore::QEntity
{
    Q_OBJECT

public:
	RandomView3D(WorkingSetManager* workingset, RandomLineView* random,GraphEditor_LineShape* line, QString nameView,/*Qt3DRender::QLayer* layer,*/Qt3DCore::QNode *parent = nullptr);

	RandomView3D(WorkingSetManager* workingset, RandomLineView* random,GraphEditor_LineShape* line, QString nameView,
			QVector<CudaImageTexture*> cudaTextures,QVector<QVector2D> ranges,/*Qt3DRender::QLayer* laye,*/Qt3DCore::QNode *parent = nullptr);
    ~RandomView3D();

    void init(QVector<QVector3D> listepoints, float width, float height);//,CudaImageTexture* cudaTexture,QVector2D range);
    void update(QVector3D center, QVector3D normal);
    void initMaterial(CudaImageTexture* cudaTexture,QVector2D range);
    void setParam(float width, float height);
    void refreshWidth(QVector<QVector3D> listepoints,float width);
    void updateMaterial(CudaImageTexture* cudaTexture,QVector2D range);

    bool isEquals(QString name);
    bool isEquals(QVector<QVector3D> listepoints);

    QVector<QVector3D> getPoints();
    QString getName();

    void setSelected(bool b);


    void setZScale(float);

    void destroyRandom();

    RandomLineView* getRandomLineView();

    void show();
    void hide();

    void setColorCross(QColor c);

    signals:
	void sendAnimationCam(int button,QVector3D pos);
	void destroy(RandomView3D*);


	public slots:
	void deleteRamdomLine();


private:
    Qt3DCore::QEntity *m_planeEntity = nullptr;
    Qt3DRender::QMaterial* m_material;
   // Qt3DExtras::QTextureMaterial* m_material;
    CustomGeometry* m_currentMesh = nullptr;

    QVector<QVector3D> m_listePts;

    QString m_nameView;
    bool m_selected;

    float m_ratioSize;
    bool m_actifRayon = false;
    float m_width;
    float m_height;

    Qt3DCore::QTransform*  m_transfo;

    Qt3DRender::QParameter * m_parameterTexture= nullptr;
    Qt3DRender::QParameter * m_parameterRange= nullptr;
    QPointer<Qt3DRender::QParameter> m_parameterHover;


    QPointer<RandomLineView> m_random ;
    GraphEditor_LineShape* m_lineShape = nullptr;

    RandomDataset* m_randomData = nullptr;
    WorkingSetManager* m_workingset = nullptr;
};


#endif
