#ifndef GrayMaterialInitializer_H
#define GrayMaterialInitializer_H

#include <Qt3DRender/QParameter>
#include "genericmaterialinitializer.h"
#include "cudaimagetexture.h"
#include "colortabletexture.h"
#include <QVector2D>


class GrayMaterialInitializer: public GenericMaterialInitializer
{
private:
	Qt3DRender::QParameter * m_paletteAttrRangeParameter;
	CudaImageTexture * m_cudaGrayTexture;
	ImageFormats::QSampleType m_sampleTypeImage;

	QVector2D m_ratioGray;
	ColorTableTexture * m_colorTexture;


public:

	GrayMaterialInitializer(ImageFormats::QSampleType sampleTypeImage,QVector2D ratio,CudaImageTexture * cudaTexture,ColorTableTexture* colorTable);

	void rangeChanged( const QVector2D & value);
	void initMaterial(Qt3DRender::QMaterial *material, QString pathVertex) override;

	//item destructor by material
	void hide() override;

};

#endif
