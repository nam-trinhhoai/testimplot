#ifndef RGBLayerRGT3DLayer_H
#define RGBLayerRGT3DLayer_H

#include <QVector3D>
#include <QMutex>
#include "graphic3Dlayer.h"
#include "lookuptable.h"
#include "genericsurface3Dlayer.h"
#include "rgbmaterialinitializer.h"
#include "surfacecollision.h"

class RGBLayerRGTRep;
class LayerSlice;
class RGBLayerSlice;
class ColorTableTexture;
class CudaImageTexture;
class CUDAImagePaletteHolder;
class SurfaceMesh;
class GenericSurface3DLayer;

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

class RGBLayerRGT3DLayer : public Graphic3DLayer, public SurfaceCollision {
Q_OBJECT
public:
	RGBLayerRGT3DLayer(RGBLayerRGTRep *rep, QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera);
	virtual ~RGBLayerRGT3DLayer();

	virtual void show() override;
	virtual void hide() override;

	virtual QRect3D boundingRect() const override;

	virtual void refresh() override;


	virtual void zScale(float val) override;

	float distanceSigned(QVector3D position, bool* ok)override;


private:
	RGBLayerSlice * rgbLayerSlice() const;
	void updateTexture(CudaImageTexture * texture,CUDAImagePaletteHolder *img );

	void updateSurfaceY();
protected slots:
	void updateRed();
	void updateGreen();
	void updateBlue();
	void updateIsoSurface();


	void opacityChanged(float val);
	void rangeChanged(unsigned int i, const QVector2D & value);
	void minValueActivated(bool activated);
	void minValueChanged(float value);

protected:



	ColorTableTexture * m_colorTexture;
	CudaImageTexture * m_cudaRedTexture;
	CudaImageTexture * m_cudaGreenTexture;
	CudaImageTexture * m_cudaBlueTexture;
	CudaImageTexture * m_cudaSurfaceTexture;


	RGBLayerRGTRep *m_rep;


    float m_cubeOrigin;
    float m_cubeScale;

    //surface generique
    GenericSurface3DLayer* m_surface;
    RGBMaterialInitializer* m_rgbMaterial;
};

#endif
