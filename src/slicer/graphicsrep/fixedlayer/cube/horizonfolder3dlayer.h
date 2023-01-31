#ifndef HorizonFolder3DLayer_H
#define HorizonFolder3DLayer_H

#include <QVector3D>
#include "graphic3Dlayer.h"
#include "lookuptable.h"
#include "surfacemesh.h"
#include "genericsurface3Dlayer.h"
#include "rgbinterleavedmaterialinitializer.h"
#include "surfacecollision.h"
#include "horizonfolderdata.h"
#include "cpuimagepaletteholder.h"
#include <QMutex>
#include <QPointer>

class FixedRGBLayersFromDatasetAndCubeRep;
class FixedRGBLayersFromDatasetAndCube;
class ColorTableTexture;
class CudaImageTexture;
class IImagePaletteHolder;
class CUDARGBInterleavedImage;
class HorizonDataRep;
class CPUImagePaletteHolder;

namespace Qt3DCore {
	class QEntity;
	class QTransform;
}

namespace Qt3DRender {
	class QMaterial;
	class QParameter;
	class QPickEvent;
}
namespace Qt3DInput
{
	class QMouseEvent;
}

class HorizonFolder3DLayer : public Graphic3DLayer, public SurfaceCollision {
Q_OBJECT
public:
HorizonFolder3DLayer(HorizonDataRep *rep, QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera);
	virtual ~HorizonFolder3DLayer();

	virtual void show() override;
	virtual void hide() override;

	virtual QRect3D boundingRect() const override;

	virtual void refresh() override;


	virtual void zScale(float val) override;

	float distanceSigned(QVector3D position, bool* ok)override;

	void setBuffer(CUDARGBInterleavedImage* image ,CPUImagePaletteHolder* isoSurfaceHolder);

	void internalShow();
	 void internalHide();


	 void generateCacheAnimation(CUDARGBInterleavedImage* image ,CPUImagePaletteHolder* isoSurfaceHolder);
	 void clearCacheAnimation();

	 void createEntityCache(CUDARGBInterleavedImage* image ,CPUImagePaletteHolder* isoSurfaceHolder);

	 void setVisible(int index);



private:
	HorizonFolderData * data() const;
	void updateTexture(CudaImageTexture * texture,CUDARGBInterleavedImage *img );
	void updateTexture(CudaImageTexture * texture,IImagePaletteHolder *img );

	//bool m_actifRayon;

protected slots:
	void minValueActivated(bool activated);
	void minValueChanged(float value);

void updateRgb();
void updateIsoSurface();


	void opacityChanged(float val);
	void rangeChanged(unsigned int i, const QVector2D & value);
protected:
	//Qt3DCore::QEntity *m_sliceEntity;
//	SurfaceMesh *m_EntityMesh;
	float m_cubeOrigin;
	float m_cubeScale;

	ColorTableTexture * m_colorTexture;
	QPointer<CudaImageTexture> m_cudaRgbTexture;
	CudaImageTexture * m_cudaSurfaceTexture;


	QPointer<CUDARGBInterleavedImage> m_lastimage=nullptr;
	QPointer<CPUImagePaletteHolder> m_lastiso = nullptr;
	//Qt3DRender::QMaterial *m_material;
	//Qt3DRender::QParameter * m_opacityParameter;

	//Qt3DRender::QParameter * m_paletteRedRangeParameter;
	//Qt3DRender::QParameter * m_paletteGreenRangeParameter;
	//Qt3DRender::QParameter * m_paletteBlueRangeParameter;

//	Qt3DCore::QTransform *m_transform;

	HorizonDataRep *m_rep;

	GenericSurface3DLayer* m_surface;
	RGBInterleavedMaterialInitializer* m_rgbMaterial;

	bool m_showOK = false;
	bool m_showInternal = false;


	QMutex m_mutex;

	//stockage en memoire pour animation
	QList<GenericSurface3DLayer*> m_listSurface;

};

#endif
