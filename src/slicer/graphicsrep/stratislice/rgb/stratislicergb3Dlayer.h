#ifndef StratiSliceRGB3DLayer_H
#define StratiSliceRGB3DLayer_H

#include <QVector3D>
#include "graphic3Dlayer.h"
#include "lookuptable.h"

class StratiSliceRGBAttributeRep;
class StratiSlice;
class ColorTableTexture;
class CudaImageTexture;
class CUDAImagePaletteHolder;

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

class StratiSliceRGB3DLayer: public Graphic3DLayer {
Q_OBJECT
public:
	StratiSliceRGB3DLayer(StratiSliceRGBAttributeRep *rep, QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera);
	virtual ~StratiSliceRGB3DLayer();

	virtual void show() override;
	virtual void hide() override;

	virtual QRect3D boundingRect() const override;

	virtual void refresh() override;


	virtual void zScale(float val) override;


private:
	StratiSlice * stratiSlice() const;
	void updateTexture(CudaImageTexture * texture,CUDAImagePaletteHolder *img );
protected slots:
	void updateRed();
	void updateGreen();
	void updateBlue();
	void updateIsoSurface();

	void opacityChanged(float val);
	void rangeChanged(unsigned int i, const QVector2D & value);
protected:
	Qt3DCore::QEntity *m_sliceEntity;
	ColorTableTexture * m_colorTexture;
	CudaImageTexture * m_cudaRedTexture;
	CudaImageTexture * m_cudaGreenTexture;
	CudaImageTexture * m_cudaBlueTexture;
	CudaImageTexture * m_cudaSurfaceTexture;

	Qt3DRender::QMaterial *m_material;
	Qt3DRender::QParameter * m_opacityParameter;

	Qt3DRender::QParameter * m_paletteRedRangeParameter;
	Qt3DRender::QParameter * m_paletteGreenRangeParameter;
	Qt3DRender::QParameter * m_paletteBlueRangeParameter;

	Qt3DCore::QTransform *m_transform;

	StratiSliceRGBAttributeRep *m_rep;
};

#endif
