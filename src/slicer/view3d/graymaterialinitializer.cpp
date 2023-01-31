#include "graymaterialinitializer.h"
#include "qt3dhelpers.h"


GrayMaterialInitializer::GrayMaterialInitializer(ImageFormats::QSampleType sampleTypeImage,QVector2D ratio, CudaImageTexture * cudaTexture,ColorTableTexture* colorTexture)
{
	m_paletteAttrRangeParameter = nullptr;

	m_sampleTypeImage= sampleTypeImage;
	m_ratioGray = ratio;

	m_cudaGrayTexture =cudaTexture;
	m_colorTexture =colorTexture;

}

void  GrayMaterialInitializer::rangeChanged(const QVector2D & value)
{
	if (m_paletteAttrRangeParameter!=nullptr)
		m_paletteAttrRangeParameter->setValue(value);

}

void  GrayMaterialInitializer::initMaterial(Qt3DRender::QMaterial *material, QString pathVertex)
{

	if (m_sampleTypeImage==ImageFormats::QSampleType::FLOAT32) {
		material->setEffect(Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/grayDebugPhong.frag",pathVertex));
	} else {
		qDebug()<<" initMaterial igrayDebugPhong";
		material->setEffect(Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/igrayDebugPhong.frag",pathVertex));
	}


	material->addParameter(new Qt3DRender::QParameter(QStringLiteral("elementMap"),m_cudaGrayTexture));

	material->addParameter(new Qt3DRender::QParameter(QStringLiteral("colormap"),m_colorTexture));


	m_paletteAttrRangeParameter = new Qt3DRender::QParameter(QStringLiteral("paletteRange"),m_ratioGray);
	material->addParameter(m_paletteAttrRangeParameter);
}

void  GrayMaterialInitializer::hide()
{
	m_paletteAttrRangeParameter = nullptr;

}
