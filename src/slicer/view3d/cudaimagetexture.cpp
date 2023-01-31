#include "cudaimagetexture.h"
#include "colortableregistry.h"
#include <Qt3DRender/QAbstractTextureImage>
#include <Qt3DRender/QTextureImageDataGenerator>
#include <array>
#include "cudaimagepaletteholder.h"

namespace {

class SkinImageGenerator: public Qt3DRender::QTextureImageDataGenerator {
public:
	explicit SkinImageGenerator(const QByteArray &data,
			ImageFormats::QSampleType sampleType,ImageFormats::QColorFormat colorFormat, int width, int height) :
			m_data(data), m_sampleType(sampleType),m_colorFormat(colorFormat), m_width(width), m_height(height)
	{
	}

	~SkinImageGenerator() {
	}

	QT3D_FUNCTOR(SkinImageGenerator)

	Qt3DRender::QTextureImageDataPtr operator ()() override
	{
		Qt3DRender::QTextureImageDataPtr textureData =
				Qt3DRender::QTextureImageDataPtr::create();
		textureData->setTarget(QOpenGLTexture::Target2D);

		if(m_colorFormat ==ImageFormats::QColorFormat::GRAY)
		{
			switch (m_sampleType) {
			case ImageFormats::QSampleType::FLOAT32:
				textureData->setPixelFormat(QOpenGLTexture::Red);
				textureData->setFormat(QOpenGLTexture::R32F);
				textureData->setPixelType(QOpenGLTexture::Float32);
				break;
			case ImageFormats::QSampleType::INT8:
				textureData->setPixelFormat(QOpenGLTexture::Red_Integer);
				textureData->setFormat(QOpenGLTexture::R8I);
				textureData->setPixelType(QOpenGLTexture::Int8);
				break;
			case ImageFormats::QSampleType::UINT8:
				textureData->setPixelFormat(QOpenGLTexture::Red_Integer);
				textureData->setFormat(QOpenGLTexture::R8U);
				textureData->setPixelType(QOpenGLTexture::UInt8);
				break;
			case ImageFormats::QSampleType::INT16:
				textureData->setPixelFormat(QOpenGLTexture::Red_Integer);
				textureData->setFormat(QOpenGLTexture::R16I);
				textureData->setPixelType(QOpenGLTexture::Int16);
				break;
			case ImageFormats::QSampleType::UINT16:
				textureData->setPixelFormat(QOpenGLTexture::Red_Integer);
				textureData->setFormat(QOpenGLTexture::R16U);
				textureData->setPixelType(QOpenGLTexture::UInt16);
				break;
			case ImageFormats::QSampleType::INT32:
				textureData->setPixelFormat(QOpenGLTexture::Red_Integer);
				textureData->setFormat(QOpenGLTexture::R32I);
				textureData->setPixelType(QOpenGLTexture::Int32);
				break;
			case ImageFormats::QSampleType::UINT32:
				textureData->setPixelFormat(QOpenGLTexture::Red_Integer);
				textureData->setFormat(QOpenGLTexture::R32U);
				textureData->setPixelType(QOpenGLTexture::UInt32);
				break;
			default:
				qDebug() << "Invalid sample type SkinImageGenerator::operator ()" << QString::fromStdString(m_sampleType.str());
				textureData->setPixelFormat(QOpenGLTexture::Red_Integer);
				textureData->setFormat(QOpenGLTexture::R16I);
				textureData->setPixelType(QOpenGLTexture::Int16);
				break;
			}
		}
		else //RGB
		{
			switch (m_sampleType) {
			case ImageFormats::QSampleType::FLOAT32:
				textureData->setPixelFormat(QOpenGLTexture::RGB);
				textureData->setFormat(QOpenGLTexture::RGB32F);
				textureData->setPixelType(QOpenGLTexture::Float32);
				break;
			case ImageFormats::QSampleType::INT8:
				textureData->setPixelFormat(QOpenGLTexture::RGB_Integer);
				textureData->setFormat(QOpenGLTexture::RGB8I);
				textureData->setPixelType(QOpenGLTexture::Int8);
				break;
			case ImageFormats::QSampleType::UINT8:
				textureData->setPixelFormat(QOpenGLTexture::RGB_Integer);
				textureData->setFormat(QOpenGLTexture::RGB8U);
				textureData->setPixelType(QOpenGLTexture::UInt8);
				break;
			case ImageFormats::QSampleType::INT16:
				textureData->setPixelFormat(QOpenGLTexture::RGB_Integer);
				textureData->setFormat(QOpenGLTexture::RGB16I);
				textureData->setPixelType(QOpenGLTexture::Int16);
				break;
			case ImageFormats::QSampleType::UINT16:
				textureData->setPixelFormat(QOpenGLTexture::RGB_Integer);
				textureData->setFormat(QOpenGLTexture::RGB16U);
				textureData->setPixelType(QOpenGLTexture::UInt16);
				break;
			case ImageFormats::QSampleType::INT32:
				textureData->setPixelFormat(QOpenGLTexture::RGB_Integer);
				textureData->setFormat(QOpenGLTexture::RGB32I);
				textureData->setPixelType(QOpenGLTexture::Int32);
				break;
			case ImageFormats::QSampleType::UINT32:
				textureData->setPixelFormat(QOpenGLTexture::RGB_Integer);
				textureData->setFormat(QOpenGLTexture::RGB32U);
				textureData->setPixelType(QOpenGLTexture::UInt32);
				break;
			default:
				qDebug() << "Invalid sample type SkinImageGenerator::operator ()" << QString::fromStdString(m_sampleType.str());
				textureData->setPixelFormat(QOpenGLTexture::RGB_Integer);
				textureData->setFormat(QOpenGLTexture::RGB16I);
				textureData->setPixelType(QOpenGLTexture::Int16);
				break;
			}

		}

		textureData->setWidth(m_width);
		textureData->setHeight(m_height);
		textureData->setData(m_data, 1, false);

		return textureData;
	}

	bool operator ==(const Qt3DRender::QTextureImageDataGenerator &other) const
			override
			{
		const SkinImageGenerator *gen = functor_cast<SkinImageGenerator>(
				&other);
		if (gen == this)
			return true;
		return false;
	}

private:
	const QByteArray m_data;
	const int m_width;
	const int m_height;
	const ImageFormats::QSampleType m_sampleType;
	const ImageFormats::QColorFormat m_colorFormat;
};
using SkinImageGeneratorPtr = QSharedPointer<SkinImageGenerator>;

class SkinImage: public Qt3DRender::QAbstractTextureImage {
Q_OBJECT
public:

	explicit SkinImage(const SkinImageGeneratorPtr &generator,
			Qt3DCore::QNode *parent = nullptr) :
			Qt3DRender::QAbstractTextureImage(parent), m_generator(generator) {
	}

	~SkinImage() {
	}

protected:
	// QAbstractTextureImage interface
	Qt3DRender::QTextureImageDataGeneratorPtr dataGenerator() const override
	{
		return m_generator;
	}

private:
	SkinImageGeneratorPtr m_generator;
};
}

CudaImageTexture::CudaImageTexture(ImageFormats::QColorFormat colorFormat, ImageFormats::QSampleType sampleType,int width, int height,
		Qt3DCore::QNode *parent) :
		Qt3DRender::QTexture2D(parent) {
	m_sampleType = sampleType;
	m_colorFormat = colorFormat;
	m_width=width;
	m_height=height;
	//setMinificationFilter(Qt3DRender::QTexture2D::Linear);
	//setMagnificationFilter(Qt3DRender::QTexture2D::Linear);
	setGenerateMipMaps(false);
	setWidth(m_width);
	setHeight(m_height);
	setWrapMode(Qt3DRender::QTextureWrapMode(Qt3DRender::QTextureWrapMode::ClampToEdge));
	if(colorFormat ==ImageFormats::QColorFormat::GRAY)
	{
		switch (m_sampleType) {
		case ImageFormats::QSampleType::FLOAT32:
			setFormat(Qt3DRender::QTexture2D::TextureFormat::R32F);
			break;
		case ImageFormats::QSampleType::INT8:
			setFormat(Qt3DRender::QTexture2D::TextureFormat::R8I);
			break;
		case ImageFormats::QSampleType::UINT8:
			setFormat(Qt3DRender::QTexture2D::TextureFormat::R8U);
			break;
		case ImageFormats::QSampleType::INT16:
			setFormat(Qt3DRender::QTexture2D::TextureFormat::R16I);
			break;
		case ImageFormats::QSampleType::UINT16:
			setFormat(Qt3DRender::QTexture2D::TextureFormat::R16U);
			break;
		case ImageFormats::QSampleType::INT32:
			setFormat(Qt3DRender::QTexture2D::TextureFormat::R32I);
			break;
		case ImageFormats::QSampleType::UINT32:
			setFormat(Qt3DRender::QTexture2D::TextureFormat::R32U);
			break;
		default:
			qDebug() << "Invalid sample type SkinImageGenerator::operator () , format:" << QString::fromStdString(m_sampleType.str());
			setFormat(Qt3DRender::QTexture2D::TextureFormat::R16I);
			break;
		}

	}else //RGB
	{
		switch (m_sampleType) {
			case ImageFormats::QSampleType::FLOAT32:
				setFormat(Qt3DRender::QTexture2D::TextureFormat::RGB32F);
				break;
			case ImageFormats::QSampleType::INT8:
				setFormat(Qt3DRender::QTexture2D::TextureFormat::RGB8I);
				break;
			case ImageFormats::QSampleType::UINT8:
				setFormat(Qt3DRender::QTexture2D::TextureFormat::RGB8U);
				break;
			case ImageFormats::QSampleType::INT16:
				setFormat(Qt3DRender::QTexture2D::TextureFormat::RGB16I);
				break;
			case ImageFormats::QSampleType::UINT16:
				setFormat(Qt3DRender::QTexture2D::TextureFormat::RGB16U);
				break;
			case ImageFormats::QSampleType::INT32:
				setFormat(Qt3DRender::QTexture2D::TextureFormat::RGB32I);
				break;
			case ImageFormats::QSampleType::UINT32:
				setFormat(Qt3DRender::QTexture2D::TextureFormat::RGB32U);
				break;
			default:
				qDebug() << "Invalid sample type SkinImageGenerator::operator () , format:" << QString::fromStdString(m_sampleType.str());
				setFormat(Qt3DRender::QTexture2D::TextureFormat::RGB16I);
				break;
			}
	}
}

CudaImageTexture::~CudaImageTexture()
{

}

QByteArray CudaImageTexture::data() const {
	return m_data;
}

void CudaImageTexture::setData(const QByteArray &data) {
	const QVector<Qt3DRender::QAbstractTextureImage*> images = textureImages();
	for (Qt3DRender::QAbstractTextureImage *img : images) {

		removeTextureImage(img);
		img->deleteLater();
		//delete img;
	}

	//deep copy
//	m_data=QByteArray();
//	m_data.resize(data.size());
//	memcpy(m_data.data(),data.data(),data.size());

	// avoid deep copy
	m_data = data;

	auto generator = SkinImageGeneratorPtr::create(m_data, m_sampleType,m_colorFormat,m_width, m_height);
	SkinImage *img = new SkinImage(generator, this);
	addTextureImage(img);


	emit dataChanged();
}

ImageFormats::QSampleType CudaImageTexture::sampleType()
{
	return m_sampleType;
}



#include "cudaimagetexture.moc"
