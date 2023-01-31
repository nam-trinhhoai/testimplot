#ifndef FixedLayersFromDatasetAndCube3DLayer_H
#define FixedLayersFromDatasetAndCube3DLayer_H

#include <QVector3D>
#include "graphic3Dlayer.h"
#include "lookuptable.h"
#include "surfacemesh.h"
#include "graymaterialinitializer.h"
#include "surfacecollision.h"

class FixedLayersFromDatasetAndCubeRep;
class FixedLayersFromDatasetAndCube;
class ColorTableTexture;
class CudaImageTexture;
class CUDAImagePaletteHolder;
class CPUImagePaletteHolder;
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

class FixedLayersFromDatasetAndCube3DLayer : public Graphic3DLayer, public SurfaceCollision{
Q_OBJECT
public:
	FixedLayersFromDatasetAndCube3DLayer(FixedLayersFromDatasetAndCubeRep *rep,
			QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera);
	virtual ~FixedLayersFromDatasetAndCube3DLayer();

	virtual void show() override;
	virtual void hide() override;

	virtual QRect3D boundingRect() const override;

	virtual void refresh() override;


	virtual void zScale(float val) override;


	float distanceSigned(QVector3D position, bool* ok)override;




private:
	FixedLayersFromDatasetAndCube * data() const;
	void updateTexture(CudaImageTexture * texture,CPUImagePaletteHolder *img );

	//bool m_actifRayon;

protected slots:
	void updateAttribute();
	void updateIsoSurface();
	void updateLookupTable(const LookupTable &table);

	void opacityChanged(float val);
	void rangeChanged(const QVector2D & value);

protected:
	float m_cubeOrigin;
	float m_cubeScale;

	ColorTableTexture * m_colorTexture;
	CudaImageTexture * m_cudaAttributeTexture;
	CudaImageTexture * m_cudaSurfaceTexture;

	FixedLayersFromDatasetAndCubeRep *m_rep;

	GenericSurface3DLayer* m_surface;
	GrayMaterialInitializer* m_grayMaterial;
};

#endif
