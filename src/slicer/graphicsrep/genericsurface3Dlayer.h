#ifndef GenericSurface3DLayer_H
#define GenericSurface3DLayer_H

#include <QMatrix4x4>
#include <QVector2D>
#include <QEntity>
#include <QMaterial>
#include <Qt3DRender/QParameter>
#include <Qt3DCore/QTransform>
#include <QObject>
#include <QCamera>
#include <Qt3DRender/QLayer>

#include "qt3dhelpers.h"
#include "surfacemesh.h"
#include "cudaimagepaletteholder.h"
#include "genericmaterialinitializer.h"
#include "rgbinterleavedmaterialinitializer.h"


class CudaImageTexture;
class IImagePaletteHolder;
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
	class QLayer;
}
namespace Qt3DInput
{
	class QMouseEvent;
}


class GenericSurface3DLayer: public QObject
{
	Q_OBJECT
public:
	GenericSurface3DLayer(QMatrix4x4 scene,QMatrix4x4 object,QString path="");

	void update(IImagePaletteHolder *palette, QString path, int simplifySteps, int compression);


	void Show(Qt3DCore::QEntity *root,QMatrix4x4 transformMesh, int width, int depth,CudaImageTexture * cudaSurfaceTexture, GenericMaterialInitializer* genericMaterial,
	IImagePaletteHolder *palette,float heightThreshold, float cubescale, float cubeorigin,Qt3DRender::QCamera * camera,float opacite,ImageFormats::QSampleType sampleTypeIso, int simplifySteps, int compression/*,Qt3DRender::QLayer* layer*/);

	void hide();


	void setVisible(bool b);

	void zScale(float val);
	void setOpacity(float val);

	//void updateTextureRGB(CudaImageTexture * texture);
	//void updateTextureIso(CudaImageTexture * texture);
	void reloadFromCache(SurfaceMeshCache& meshCache);


	float distanceSigned(QVector3D position, bool* ok);

	void activateNullValue(float nullValue);
	void deactivateNullValue();
	float nullValue() const;
	bool isNullValueActive() const;


signals:
	void sendPositionTarget(QVector3D pos,QVector3D target);
	void sendPositionCam(int, QVector3D pos);

private:
	void updateTexture(CudaImageTexture * texture,IImagePaletteHolder *img );

	Qt3DRender::QCamera * m_camera;
	Qt3DCore::QEntity *m_sliceEntity;
	SurfaceMesh *m_EntityMesh;
	Qt3DCore::QTransform *m_transform;
	Qt3DRender::QMaterial *m_material;

	IImagePaletteHolder *m_palette;

	Qt3DRender::QParameter * m_opacityParameter;
	Qt3DRender::QParameter * m_isoSurfaceParameter;
	Qt3DRender::QParameter * m_nullValueParameter;
	Qt3DRender::QParameter * m_nullValueActiveParameter;


	float m_cubeOrigin;
	float m_cubeScale;

	int m_width;
	int m_depth;
	float m_heightThreshold;
	float m_nullValue;
	bool m_nullValueActive;

	bool m_actifRayon;
	QString m_path;

	QMatrix4x4 m_sceneTr;
	QMatrix4x4 m_objectTr;

	Qt3DRender::QObjectPicker *spicker;

	GenericMaterialInitializer* m_genericMaterial;

	Qt3DCore::QEntity *m_root=nullptr;

};


#endif
