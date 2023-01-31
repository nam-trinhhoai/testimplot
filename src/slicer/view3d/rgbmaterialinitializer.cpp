#include "rgbmaterialinitializer.h"
#include "qt3dhelpers.h"


RGBMaterialInitializer::RGBMaterialInitializer(ImageFormats::QSampleType sampleTypeImage,QVector2D ratioRed,QVector2D ratioGreen,QVector2D ratioBlue, CudaImageTexture * cudaRedTexture,
		CudaImageTexture * cudaGreenTexture,CudaImageTexture * cudaBlueTexture)
{
	m_paletteRedRangeParameter = nullptr;
	m_paletteGreenRangeParameter = nullptr;
	m_paletteBlueRangeParameter = nullptr;
	m_minimumValueActiveParameter = nullptr;
	m_minimumValueParameter = nullptr;

	m_sampleTypeImage= sampleTypeImage;
	m_ratioRed = ratioRed;
	m_ratioGreen = ratioGreen;
	m_ratioBlue = ratioBlue;

	m_cudaRedTexture =cudaRedTexture;
	m_cudaGreenTexture =cudaGreenTexture;
	m_cudaBlueTexture =cudaBlueTexture;

	m_minimumValueActive = false;
	m_minimumValue = 0.0;
}

void  RGBMaterialInitializer::rangeChanged(unsigned int i, const QVector2D & value)
{
	if (i == 0 && m_paletteRedRangeParameter!=nullptr)
			m_paletteRedRangeParameter->setValue(value);
		else if (i == 1 && m_paletteGreenRangeParameter!=nullptr)
			m_paletteGreenRangeParameter->setValue(value);
		else if (i == 2 && m_paletteBlueRangeParameter!=nullptr)
			m_paletteBlueRangeParameter->setValue(value);
}

void  RGBMaterialInitializer::initMaterial(Qt3DRender::QMaterial *material, QString pathVertex)
{

	if (m_sampleTypeImage==ImageFormats::QSampleType::FLOAT32) {
		Qt3DRender::QEffect * eff = Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/RGBColor_simple.frag",pathVertex);

		material->setEffect(eff);
	} else {
		material->setEffect(Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/debugPhong.frag",pathVertex));
	}


	material->addParameter(new Qt3DRender::QParameter(QStringLiteral("redMap"),m_cudaRedTexture));
	material->addParameter(new Qt3DRender::QParameter(QStringLiteral("greenMap"),m_cudaGreenTexture));
	material->addParameter(new Qt3DRender::QParameter(QStringLiteral("blueMap"),m_cudaBlueTexture));



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

void  RGBMaterialInitializer::hide()
{
	m_paletteRedRangeParameter = nullptr;
	m_paletteGreenRangeParameter = nullptr;
	m_paletteBlueRangeParameter = nullptr;
}

bool RGBMaterialInitializer::isMinimumValueActive() const
{
	return m_minimumValueActive;
}

void RGBMaterialInitializer::setMinimumValueActive(bool active)
{
	m_minimumValueActive = active;
	if (m_minimumValueActiveParameter)
	{
		m_minimumValueActiveParameter->setValue(m_minimumValueActive);
	}
}

float RGBMaterialInitializer::minimumValue() const
{
	return m_minimumValue;
}

void RGBMaterialInitializer::setMinimumValue(float value)
{
	m_minimumValue = value;
	if (m_minimumValueParameter)
	{
		m_minimumValueParameter->setValue(m_minimumValue);
	}
}
