#include "cudaimagetexturemapper.h"
#include <QOpenGLTexture>
#include <QOpenGLPixelTransferOptions>
#include "texturehelper.h"
#include "lookuptable.h"
#include "iimagepaletteholder.h"

CUDAImageTextureMapper::CUDAImageTextureMapper(IImagePaletteHolder *image,
		QObject *parent) {
	m_buffer=image;

	if(m_buffer == nullptr)
	{
		qDebug()<<" buffer est nullptr";
	}
	m_needInternalBufferUpdate=true;
	m_texture=nullptr;
	m_colorMapTexture=nullptr;
	m_needColorTableReload=false;
	m_internalBuffer=nullptr;

	connect(image,SIGNAL(lookupTableChanged(const LookupTable &)),this,SLOT(lookupTableChanged(const LookupTable &)));
	connect(image,SIGNAL(dataChanged()),this,SLOT(updateTexture()));
}

CUDAImageTextureMapper::~CUDAImageTextureMapper() {
	if(m_internalBuffer)
		delete [] m_internalBuffer;
}


void CUDAImageTextureMapper::lookupTableChanged(const LookupTable & table)
{
	m_needColorTableReload=true;
}



bool CUDAImageTextureMapper::textureInitialized() {
	return m_texture != nullptr;;
}

void CUDAImageTextureMapper::bindTexture(unsigned int unit) {
	if (m_texture == nullptr) {
		createTexture();
	}
	if (m_needInternalBufferUpdate) {
		QOpenGLTexture::PixelType sourceType;
		QOpenGLTexture::PixelFormat sourceFormat;
		switch(m_buffer->sampleType()) {
			case ImageFormats::QSampleType::FLOAT32:
				sourceType = QOpenGLTexture::Float32;
				sourceFormat = QOpenGLTexture::Red;
				break;
			case ImageFormats::QSampleType::INT8:
				sourceType = QOpenGLTexture::Int8;
				sourceFormat = QOpenGLTexture::Red_Integer;
				break;
			case ImageFormats::QSampleType::UINT8:
				sourceType = QOpenGLTexture::UInt8;
				sourceFormat = QOpenGLTexture::Red_Integer;
				break;
			case ImageFormats::QSampleType::UINT16:
				sourceType = QOpenGLTexture::UInt16;
				sourceFormat = QOpenGLTexture::Red_Integer;
				break;
			case ImageFormats::QSampleType::INT16:
				sourceType = QOpenGLTexture::Int16;
				sourceFormat = QOpenGLTexture::Red_Integer;
				break;
			case ImageFormats::QSampleType::UINT32:
				sourceType = QOpenGLTexture::UInt32;
				sourceFormat = QOpenGLTexture::Red_Integer;
				break;
			case ImageFormats::QSampleType::INT32:
				sourceType = QOpenGLTexture::Int32;
				sourceFormat = QOpenGLTexture::Red_Integer;
				break;
			default:
				qDebug() << "CUDAImageTextureMapper::bindTexture invalid format" << QString::fromStdString(m_buffer->sampleType().str());
				sourceType = QOpenGLTexture::Int16;
				sourceFormat = QOpenGLTexture::Red_Integer;
				break;
			}

		QOpenGLPixelTransferOptions uploadOptions;
		uploadOptions.setAlignment(1);
		size_t pointerSize = m_buffer->internalPointerSize();
		//defense copy
		m_buffer->lockPointer();
		memcpy(m_internalBuffer,m_buffer->constBackingPointer(),pointerSize);
		m_buffer->unlockPointer();
		m_texture->setData(sourceFormat, sourceType,
					(const char*) m_internalBuffer, &uploadOptions);

		m_needInternalBufferUpdate = false;

	}
	m_texture->bind(unit, QOpenGLTexture::TextureUnitReset::ResetTextureUnit);
}

void CUDAImageTextureMapper::releaseTexture(unsigned int unit) {
	m_texture->release(unit,
			QOpenGLTexture::TextureUnitReset::ResetTextureUnit);
}


void CUDAImageTextureMapper::createTexture() {
	m_texture = new QOpenGLTexture(QOpenGLTexture::Target2D);
	if (!m_texture && !m_texture->create()) {
		qDebug() << "Ooops! failed to create texture";
		return;
	}
	m_texture->setSize(m_buffer->width(), m_buffer->height());
	m_texture->setAutoMipMapGenerationEnabled(false);
	m_texture->setMipLevels(1);
	m_texture->setMinificationFilter(QOpenGLTexture::Linear);
	m_texture->setMagnificationFilter(QOpenGLTexture::Linear);
	m_texture->setWrapMode(QOpenGLTexture::ClampToEdge);

	QOpenGLTexture::PixelType sourceType;
	QOpenGLTexture::PixelFormat sourceFormat;
	switch(m_buffer->sampleType()) {
	case ImageFormats::QSampleType::FLOAT32:
		sourceFormat = QOpenGLTexture::Red;
		sourceType = QOpenGLTexture::Float32;
		m_texture->setFormat(QOpenGLTexture::R32F);
		break;
	case ImageFormats::QSampleType::INT8:
		sourceFormat = QOpenGLTexture::Red_Integer;
		sourceType = QOpenGLTexture::Int8;
		m_texture->setFormat(QOpenGLTexture::R8I);
		break;
	case ImageFormats::QSampleType::UINT8:
		sourceFormat = QOpenGLTexture::Red_Integer;
		sourceType = QOpenGLTexture::UInt8;
		m_texture->setFormat(QOpenGLTexture::R8U);
		break;
	case ImageFormats::QSampleType::UINT16:
		sourceFormat = QOpenGLTexture::Red_Integer;
		sourceType = QOpenGLTexture::UInt16;
		m_texture->setFormat(QOpenGLTexture::R16U);
		break;
	case ImageFormats::QSampleType::INT16:
		sourceType = QOpenGLTexture::Int16;
		sourceFormat = QOpenGLTexture::Red_Integer;
		m_texture->setFormat(QOpenGLTexture::R16I);
		break;
	case ImageFormats::QSampleType::UINT32:
		sourceFormat = QOpenGLTexture::Red_Integer;
		sourceType = QOpenGLTexture::UInt32;
		m_texture->setFormat(QOpenGLTexture::R32U);
		break;
	case ImageFormats::QSampleType::INT32:
		sourceType = QOpenGLTexture::Int32;
		sourceFormat = QOpenGLTexture::Red_Integer;
		m_texture->setFormat(QOpenGLTexture::R32I);
		break;
	default:
		qDebug() << "CUDAImageTextureMapper::createTexture invalid format" << QString::fromStdString(m_buffer->sampleType().str());
		sourceType = QOpenGLTexture::Int16;
		sourceFormat = QOpenGLTexture::Red_Integer;
		m_texture->setFormat(QOpenGLTexture::R16I);
		break;
	}
	m_texture->allocateStorage(sourceFormat, sourceType);
	m_internalBuffer=new char[m_buffer->internalPointerSize()];
}


void CUDAImageTextureMapper::updateTexture() {
	m_needInternalBufferUpdate = true;
}

void CUDAImageTextureMapper::bindLUTTexture(unsigned int unit) {

	if(m_needColorTableReload)
	{
		if(m_colorMapTexture!=nullptr)
			m_colorMapTexture->destroy();
		m_colorMapTexture=nullptr;
		m_needColorTableReload=false;
	}
	if (m_colorMapTexture != nullptr) {
		m_colorMapTexture->bind(unit,
				QOpenGLTexture::TextureUnitReset::ResetTextureUnit);
	} else {
		m_colorMapTexture = TextureHelper::generateLUTTexture(m_buffer->lookupTable());
		m_colorMapTexture->bind(unit,
				QOpenGLTexture::TextureUnitReset::ResetTextureUnit);
	}
}

void CUDAImageTextureMapper::releaseLUTTexture(unsigned int unit) {
	m_colorMapTexture->release(unit,
			QOpenGLTexture::TextureUnitReset::ResetTextureUnit);
}





