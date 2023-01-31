#ifndef RGBMaterialInitializer_H
#define RGBMaterialInitializer_H

#include <Qt3DRender/QParameter>
#include "genericmaterialinitializer.h"
#include "cudaimagetexture.h"
#include <QVector2D>


class RGBMaterialInitializer: public GenericMaterialInitializer
{
private:
	Qt3DRender::QParameter * m_paletteRedRangeParameter;
	Qt3DRender::QParameter * m_paletteGreenRangeParameter;
	Qt3DRender::QParameter * m_paletteBlueRangeParameter;
	Qt3DRender::QParameter * m_minimumValueActiveParameter;
	Qt3DRender::QParameter * m_minimumValueParameter;

	CudaImageTexture * m_cudaRedTexture;
	CudaImageTexture * m_cudaGreenTexture;
	CudaImageTexture * m_cudaBlueTexture;

	ImageFormats::QSampleType m_sampleTypeImage;

	QVector2D m_ratioRed;
	QVector2D m_ratioGreen;
	QVector2D m_ratioBlue;

	bool m_minimumValueActive;
	float m_minimumValue;

public:

	RGBMaterialInitializer(ImageFormats::QSampleType sampleTypeImage,QVector2D ratioRed,QVector2D ratioGreen,QVector2D ratioBlue,CudaImageTexture * cudaRedTexture,
			CudaImageTexture * cudaGreenTexture,CudaImageTexture * cudaBlueTexture);
	void rangeChanged(unsigned int i, const QVector2D & value);
	void initMaterial(Qt3DRender::QMaterial *material, QString pathVertex) override;

	//item destructor by material
	void hide() override;

	// hsv
	bool isMinimumValueActive() const;
	void setMinimumValueActive(bool active);
	float minimumValue() const;
	void setMinimumValue(float value);
};

#endif
