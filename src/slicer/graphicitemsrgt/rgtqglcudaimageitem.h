#ifndef RGTQGLCUDAImageItem_H
#define RGTQGLCUDAImageItem_H

#include "qglfullcudaimageitem.h"
#include "CUDAImageMask.h"

class IImagePaletteHolder;
class CUDAImageTextureMapper;
class RGTQGLCUDAImageItem: public QGLFullCUDAImageItem{
	Q_OBJECT
public:
	RGTQGLCUDAImageItem(IImagePaletteHolder * isoSurfaceHolder,IImagePaletteHolder * attributeHolder,
			QGraphicsItem *parent=0, bool ApplyMask = false);
	~RGTQGLCUDAImageItem();
protected:
	virtual void preInitGL() override;
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix,
				const QRectF &exposed, int width, int height, int dpiX,int dpiY) override;

private:
	IImagePaletteHolder *m_isoSurfaceImage;
	CUDAImageTextureMapper * m_isoSurfaceMapper;
};

#endif
