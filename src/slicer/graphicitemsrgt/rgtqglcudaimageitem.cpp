#include "rgtqglcudaimageitem.h"

#include <QOpenGLTexture>
#include "iimagepaletteholder.h"
#include "cudaimagetexturemapper.h"

#define ISOVALUE_TEXTURE_UNIT 3
//
RGTQGLCUDAImageItem::RGTQGLCUDAImageItem(IImagePaletteHolder * isoSurfaceHolder,IImagePaletteHolder * attributeHolder,
		QGraphicsItem *parent, bool applyMask) :QGLFullCUDAImageItem(attributeHolder,parent,applyMask) {
	m_isoSurfaceImage=isoSurfaceHolder;
	m_isoSurfaceMapper=new CUDAImageTextureMapper(m_isoSurfaceImage,this);
}

RGTQGLCUDAImageItem::~RGTQGLCUDAImageItem() {
}

void RGTQGLCUDAImageItem::preInitGL()
{
	m_isoSurfaceMapper->bindTexture(ISOVALUE_TEXTURE_UNIT);
	m_isoSurfaceMapper->releaseTexture(ISOVALUE_TEXTURE_UNIT);

	if (texturemapper)
	{
		texturemapper->bindTexture(MASK_TEXTURE_UNIT);
		texturemapper->releaseTexture(MASK_TEXTURE_UNIT);
	}
}

void RGTQGLCUDAImageItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX,int dpiY)
{
	m_isoSurfaceMapper->bindTexture(ISOVALUE_TEXTURE_UNIT);
	m_isoSurfaceMapper->releaseTexture(ISOVALUE_TEXTURE_UNIT);

	QGLFullCUDAImageItem::drawGL(viewProjectionMatrix,exposed,width,height,dpiX,dpiY);
	//this is not needed here but if we d'ont do this the Iso value texture is not refreshed and the 3D view is not refreshed
}
