#ifndef LayerRGT3DLayer_H
#define LayerRGT3DLayer_H

#include <QVector3D>
#include "graphic3Dlayer.h"
#include "lookuptable.h"
#include "genericsurface3Dlayer.h"
#include "graymaterialinitializer.h"
#include "surfacecollision.h"

class LayerRGTRep;
class LayerSlice;
class ColorTableTexture;
class CudaImageTexture;
class CUDAImagePaletteHolder;

class LayerRGT3DLayer: public Graphic3DLayer, public SurfaceCollision {
Q_OBJECT
public:
	LayerRGT3DLayer(LayerRGTRep *rep, QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera);
	virtual ~LayerRGT3DLayer();

	virtual void show() override;
	virtual void hide() override;

	virtual QRect3D boundingRect() const override;

	virtual void refresh() override;
	virtual void zScale(float val) override;

	float distanceSigned(QVector3D position, bool* ok)override;

private:
	LayerSlice * layerSlice() const;
	void updateTexture(CudaImageTexture * texture,CUDAImagePaletteHolder *img );
protected slots:
	void update();
	void updateIsoSurface();

	void updateLookupTable(const LookupTable & table);
	void opacityChanged(float val);
	void rangeChanged();
protected:
	ColorTableTexture * m_colorTexture;
	CudaImageTexture * m_cudaTexture;
	CudaImageTexture * m_cudaSurfaceTexture;


    GrayMaterialInitializer* m_grayMaterial;

	LayerRGTRep *m_rep;

public:
	GenericSurface3DLayer* m_surface;
};

#endif
