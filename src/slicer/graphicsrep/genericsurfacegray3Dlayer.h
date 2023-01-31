#ifndef GenericSurfaceGray3DLayer_H
#define GenericSurfaceGray3DLayer_H

#include <QMatrix4x4>
#include <QVector2D>
#include <QEntity>
#include <QMaterial>
#include <Qt3DRender/QParameter>
#include <Qt3DCore/QTransform>
#include <QObject>
#include <QCamera>

#include "qt3dhelpers.h"
#include "surfacemesh.h"
#include "cudaimagepaletteholder.h"


class CudaImageTexture;
class ColorTableTexture;
class CUDAImagePaletteHolder;
class SurfaceMesh;

namespace Qt3DCore {
	class QEntity;
	class QTransform;
}

namespace Qt3DRender {
	class QMaterial;
	class QParameter;
	class QPickEvent;
	class QCamera;
}
namespace Qt3DInput
{
	class QMouseEvent;
}

class GenericSurfaceGray3DLayer: public QObject
{
	Q_OBJECT
public:
	GenericSurfaceGray3DLayer(QMatrix4x4 scene,QMatrix4x4 object,QString path);
	~GenericSurfaceGray3DLayer();

	void update(CUDAImagePaletteHolder *palette, QString path, int simplifySteps, int compression);
	void Show(Qt3DCore::QEntity *root,QMatrix4x4 transformMesh, int width, int depth,CudaImageTexture * cudaAttributeTexture,
			CudaImageTexture * cudaSurfaceTexture, QVector2D ratioAttribute, ColorTableTexture * colorTexture, CUDAImagePaletteHolder *palette,
			float heightThreshold, float cubescale, float cubeorigin,Qt3DRender::QCamera * camera,float opacite,ImageFormats::QSampleType sampleTypeImage,
			ImageFormats::QSampleType sampleTypeIso, int simplifySteps,int compression);

	void hide();

	void zScale(float val);
	void rangeChanged(const QVector2D & value);
	void setOpacity(float val);

	void reloadFromCache(SurfaceMeshCache& meshCache);

protected slots:
	void updateIsoSurface();

private:
	void updateTexture(CudaImageTexture * texture,CUDAImagePaletteHolder *img );

	Qt3DRender::QCamera * m_camera;
	Qt3DCore::QEntity *m_sliceEntity;
	SurfaceMesh *m_EntityMesh;
	Qt3DCore::QTransform *m_transform;
	Qt3DRender::QMaterial *m_material;

	CUDAImagePaletteHolder *m_palette;

	Qt3DRender::QParameter * m_paletteAttrRangeParameter;
	Qt3DRender::QParameter * m_opacityParameter;

	float m_cubeOrigin;
	float m_cubeScale;

	float m_heightThreshold;

	bool m_actifRayon;

	int m_width;
	int m_depth;
	QString m_path;

	QMatrix4x4 m_sceneTr;
	QMatrix4x4 m_objectTr;


};


#endif
