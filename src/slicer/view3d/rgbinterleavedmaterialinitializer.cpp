#include "rgbinterleavedmaterialinitializer.h"
#include "qt3dhelpers.h"


RGBInterleavedMaterialInitializer::RGBInterleavedMaterialInitializer(ImageFormats::QSampleType sampleTypeImage,QVector2D ratioRed,QVector2D ratioGreen,QVector2D ratioBlue, CudaImageTexture * cudaRGBTexture)
{
	m_paletteRedRangeParameter = nullptr;
	m_paletteGreenRangeParameter = nullptr;
	m_paletteBlueRangeParameter = nullptr;
	m_textureRGBParameter = nullptr;
	m_minimumValueActiveParameter = nullptr;
	m_minimumValueParameter = nullptr;

	m_sampleTypeImage= sampleTypeImage;
	m_ratioRed = ratioRed;
	m_ratioGreen = ratioGreen;
	m_ratioBlue = ratioBlue;

	m_cudaRgbTexture =cudaRGBTexture;

}

void  RGBInterleavedMaterialInitializer::rangeChanged(unsigned int i, const QVector2D & value)
{
	if (i == 0 && m_paletteRedRangeParameter!=nullptr)
			m_paletteRedRangeParameter->setValue(value);
		else if (i == 1 && m_paletteGreenRangeParameter!=nullptr)
			m_paletteGreenRangeParameter->setValue(value);
		else if (i == 2 && m_paletteBlueRangeParameter!=nullptr)
			m_paletteBlueRangeParameter->setValue(value);
}

void  RGBInterleavedMaterialInitializer::initMaterial(Qt3DRender::QMaterial *material, QString pathVertex)
{
	if (m_sampleTypeImage==ImageFormats::QSampleType::FLOAT32) {
		material->setEffect(Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/RGBColor_simple.frag",pathVertex));
	} else {
		material->setEffect(Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/rgbPhong.frag",pathVertex));
	}



	m_textureRGBParameter = new Qt3DRender::QParameter(QStringLiteral("rgbMap"),m_cudaRgbTexture);
	material->addParameter(m_textureRGBParameter);
	//material->addParameter(new Qt3DRender::QParameter(QStringLiteral("greenMap"),m_cudaGreenTexture));
	//material->addParameter(new Qt3DRender::QParameter(QStringLiteral("blueMap"),m_cudaBlueTexture));


	m_paletteRedRangeParameter = new Qt3DRender::QParameter(QStringLiteral("redRange"),m_ratioRed);
	material->addParameter(m_paletteRedRangeParameter);

	m_paletteGreenRangeParameter = new Qt3DRender::QParameter(QStringLiteral("greenRange"),m_ratioGreen);
	material->addParameter(m_paletteGreenRangeParameter);

	m_paletteBlueRangeParameter = new Qt3DRender::QParameter(QStringLiteral("blueRange"),m_ratioBlue);
	material->addParameter(m_paletteBlueRangeParameter);

	m_minimumValueActiveParameter = new Qt3DRender::QParameter(QStringLiteral("minValueActivated"),m_minimumValueActive);
	material->addParameter(m_minimumValueActiveParameter);

	m_minimumValueParameter = new Qt3DRender::QParameter(QStringLiteral("minValue"),m_minimumValue);
	material->addParameter(m_minimumValueParameter);

}

void RGBInterleavedMaterialInitializer::updateTextureMaterial(CudaImageTexture * texture)
{
	m_cudaRgbTexture = texture;
	m_textureRGBParameter->setValue(QVariant::fromValue(texture));
}

void  RGBInterleavedMaterialInitializer::hide()
{
	if(m_paletteRedRangeParameter != nullptr)
	{
		m_cudaRgbTexture->deleteLater();
		m_cudaRgbTexture = nullptr;
		m_paletteRedRangeParameter = nullptr;
		m_paletteGreenRangeParameter = nullptr;
		m_paletteBlueRangeParameter = nullptr;
		m_minimumValueActiveParameter= nullptr;
		m_minimumValueParameter =nullptr;
		m_textureRGBParameter= nullptr;
	}




}

bool RGBInterleavedMaterialInitializer::isMinimumValueActive() const
{
	return m_minimumValueActive;
}

void RGBInterleavedMaterialInitializer::setMinimumValueActive(bool active)
{
	m_minimumValueActive = active;
	if (m_minimumValueActiveParameter)
	{
		m_minimumValueActiveParameter->setValue(m_minimumValueActive);
	}
}

float RGBInterleavedMaterialInitializer::minimumValue() const
{
	return m_minimumValue;
}

void RGBInterleavedMaterialInitializer::setMinimumValue(float value)
{
	m_minimumValue = value;
	if (m_minimumValueParameter)
	{
		m_minimumValueParameter->setValue(m_minimumValue);
	}
}
