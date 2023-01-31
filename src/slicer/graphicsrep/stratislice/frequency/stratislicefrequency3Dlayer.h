#ifndef StratiSliceFrequency3DLayer_H
#define StratiSliceFrequency3DLayer_H

#include <QVector3D>
#include "graphic3Dlayer.h"
#include "lookuptable.h"

class StratiSliceFrequencyRep;
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

class StratiSliceFrequency3DLayer: public Graphic3DLayer {
Q_OBJECT
public:
	StratiSliceFrequency3DLayer(StratiSliceFrequencyRep *rep, QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera);
	virtual ~StratiSliceFrequency3DLayer();

	virtual void show() override;
	virtual void hide() override;

	virtual QRect3D boundingRect() const override;

	virtual void refresh() override;

	virtual void zScale(float val) override;


private:
	StratiSlice * stratiSlice() const;
	void updateTexture(CudaImageTexture * texture,CUDAImagePaletteHolder *img );
protected slots:
	void update();
	void updateIsoSurface();

	void updateLookupTable(const LookupTable & table);
	void opacityChanged(float val);
	void rangeChanged();
protected:
	Qt3DCore::QEntity *m_sliceEntity;
	ColorTableTexture * m_colorTexture;
	CudaImageTexture * m_cudaTexture;
	CudaImageTexture * m_cudaSurfaceTexture;

	Qt3DRender::QMaterial *m_material;
	Qt3DRender::QParameter * m_opacityParameter;
	Qt3DRender::QParameter * m_paletteRangeParameter;

	StratiSliceFrequencyRep *m_rep;

	Qt3DCore::QTransform *m_transform;

};

#endif
