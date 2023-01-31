#ifndef RGBInterleavedMaterialInitializer_H
#define RGBInterleavedMaterialInitializer_H

#include <Qt3DRender/QParameter>
#include "genericmaterialinitializer.h"
#include "cudaimagetexture.h"
#include <QVector2D>


class RGBInterleavedMaterialInitializer: public GenericMaterialInitializer
{
private:
	Qt3DRender::QParameter * m_paletteRedRangeParameter;
	Qt3DRender::QParameter * m_paletteGreenRangeParameter;
	Qt3DRender::QParameter * m_paletteBlueRangeParameter;
	Qt3DRender::QParameter * m_minimumValueActiveParameter;
	Qt3DRender::QParameter * m_minimumValueParameter;

	Qt3DRender::QParameter * m_textureRGBParameter;

	CudaImageTexture * m_cudaRgbTexture;


	ImageFormats::QSampleType m_sampleTypeImage;

	QVector2D m_ratioRed;
	QVector2D m_ratioGreen;
	QVector2D m_ratioBlue;
	bool m_minimumValueActive;
	float m_minimumValue;

public:

	RGBInterleavedMaterialInitializer(ImageFormats::QSampleType sampleTypeImage,QVector2D ratioRed,QVector2D ratioGreen,QVector2D ratioBlue,CudaImageTexture * cudaRGBTexture);
	void rangeChanged(unsigned int i, const QVector2D & value);
	void initMaterial(Qt3DRender::QMaterial *material, QString pathVertex) override;
	void updateTextureMaterial(CudaImageTexture * texture);
	//item destructor by material
	void hide() override;

	bool isMinimumValueActive() const;
	void setMinimumValueActive(bool active);
	float minimumValue() const;
	void setMinimumValue(float value);
};

#endif
