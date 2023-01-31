#include "texturehelper.h"
#include <QOpenGLTexture>
#include <QOpenGLPixelTransferOptions>
#include <QDebug>

QOpenGLTexture* TextureHelper::generateLUTTexture(const LookupTable &lut) {

	unsigned char buffer[4 * lut.size()];
	for (int i = 0; i < lut.size(); i++) {
		const std::array<int, 4> colors = lut.getColors(i);
		buffer[4 * i] = (unsigned char) colors[0];
		buffer[4 * i + 1] = (unsigned char) colors[1];
		buffer[4 * i + 2] = (unsigned char) colors[2];
		buffer[4 * i + 3] = (unsigned char) colors[3];
	}

	QOpenGLTexture *colorMapTexture = new QOpenGLTexture(
			QOpenGLTexture::Target1D);
	if (!colorMapTexture && !colorMapTexture->create()) {
		qDebug() << "Failed to create Color Map Texture";
		return nullptr;
	}
	colorMapTexture->setAutoMipMapGenerationEnabled(false);
	colorMapTexture->setSize(lut.size());
	colorMapTexture->setFormat(QOpenGLTexture::RGBA8_UNorm);
	if (lut.size() != 0) {
		colorMapTexture->setMinificationFilter(QOpenGLTexture::Nearest);
		colorMapTexture->setMagnificationFilter(QOpenGLTexture::Nearest);
	}
	colorMapTexture->setWrapMode(QOpenGLTexture::ClampToEdge);
	colorMapTexture->allocateStorage(QOpenGLTexture::RGBA,
			QOpenGLTexture::UInt8);
	colorMapTexture->setData(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8,
			(const void*) buffer);

	return colorMapTexture;
}

QOpenGLTexture* TextureHelper::generateTexture(const void *data, int width,
		int height, ImageFormats::QSampleType sampleType,
		ImageFormats::QColorFormat colorFormat) {
	QOpenGLTexture *tile = new QOpenGLTexture(QOpenGLTexture::Target2D);
	if (!tile && !tile->create()) {
		qDebug() << "Ooops!";
		return nullptr;
	}
	tile->setSize(width, height);
	tile->setAutoMipMapGenerationEnabled(false);
	tile->setMipLevels(1);
	tile->setMinificationFilter(QOpenGLTexture::Linear);
	tile->setMagnificationFilter(QOpenGLTexture::Linear);
	tile->setWrapMode(QOpenGLTexture::ClampToEdge);

	//For mono channel images, palette is defined across the shader
	//Texture are not normalize

	//For RGB cases, we handle transparency in the shader and updload texture in normalized mode
	QOpenGLTexture::PixelType sourceType = QOpenGLTexture::UInt8;
	QOpenGLTexture::PixelFormat sourceFormat = QOpenGLTexture::Red_Integer;

	if (sampleType == ImageFormats::QSampleType::UINT8) {
		sourceType = QOpenGLTexture::UInt8;
		tile->setFormat(QOpenGLTexture::R8U);

		if (colorFormat == ImageFormats::QColorFormat::RGB_INTERLEAVED) {
			sourceFormat = QOpenGLTexture::RGB;
			tile->setFormat(QOpenGLTexture::RGBA8_UNorm);

		} else if (colorFormat
				== ImageFormats::QColorFormat::RGBA_INTERLEAVED) {
			sourceFormat = QOpenGLTexture::RGBA;
			tile->setFormat(QOpenGLTexture::RGBA8_UNorm);
		}
	} else if (sampleType == ImageFormats::QSampleType::INT8) {
		sourceType = QOpenGLTexture::Int8;
		tile->setFormat(QOpenGLTexture::R8I);
		if (colorFormat == ImageFormats::QColorFormat::RGB_INTERLEAVED) {
			sourceFormat = QOpenGLTexture::RGB;
			tile->setFormat(QOpenGLTexture::RGBA8_SNorm);

		} else if (colorFormat
				== ImageFormats::QColorFormat::RGBA_INTERLEAVED) {
			sourceFormat = QOpenGLTexture::RGBA;
			tile->setFormat(QOpenGLTexture::RGBA8_SNorm);
		}
	} else if (sampleType == ImageFormats::QSampleType::INT16) {
		sourceType = QOpenGLTexture::Int16;
		tile->setFormat(QOpenGLTexture::R16I);
		if (colorFormat == ImageFormats::QColorFormat::RGB_INTERLEAVED) {
			sourceFormat = QOpenGLTexture::RGB;
			tile->setFormat(QOpenGLTexture::RGBA16_SNorm);

		} else if (colorFormat
				== ImageFormats::QColorFormat::RGBA_INTERLEAVED) {
			sourceFormat = QOpenGLTexture::RGBA;
			tile->setFormat(QOpenGLTexture::RGBA16_SNorm);
		}
	} else if (sampleType == ImageFormats::QSampleType::UINT16) {
		sourceType = QOpenGLTexture::UInt16;
		tile->setFormat(QOpenGLTexture::R16U);
		if (colorFormat == ImageFormats::QColorFormat::RGB_INTERLEAVED) {
			sourceFormat = QOpenGLTexture::RGB;
			tile->setFormat(QOpenGLTexture::RGBA16_UNorm);

		} else if (colorFormat
				== ImageFormats::QColorFormat::RGBA_INTERLEAVED) {
			sourceFormat = QOpenGLTexture::RGBA;
			tile->setFormat(QOpenGLTexture::RGBA16_UNorm);
		}
	} else if (sampleType == ImageFormats::QSampleType::INT32) {
		sourceType = QOpenGLTexture::Int32;
		tile->setFormat(QOpenGLTexture::R32I);
	} else if (sampleType == ImageFormats::QSampleType::UINT32) {
		sourceType = QOpenGLTexture::UInt32;
		tile->setFormat(QOpenGLTexture::R32U);
	} else if (sampleType == ImageFormats::QSampleType::FLOAT32) {
		sourceFormat = QOpenGLTexture::Red;
		sourceType = QOpenGLTexture::Float32;
		tile->setFormat(QOpenGLTexture::R32F);
	}

	qDebug() << sourceFormat << "," << sourceType << "," << tile->format();
	tile->allocateStorage(sourceFormat, sourceType);

	QOpenGLPixelTransferOptions uploadOptions;
	uploadOptions.setAlignment(1);
	tile->setData(sourceFormat, sourceType, data, &uploadOptions);
	return tile;
}

bool TextureHelper::setValue(void *data, int i, int j, int width,
		ImageFormats::QSampleType sampleType, double value)
{
	if (sampleType == ImageFormats::QSampleType::UINT8) {
		((unsigned char*) data)[j * width + i] = value;
		return true;
	} else if (sampleType == ImageFormats::QSampleType::INT8) {
		((char*) data)[j * width + i] = value;
		return true;
	} else if (sampleType == ImageFormats::QSampleType::UINT16) {
		((unsigned short*) data)[j * width + i] = value;
		return true;
	} else if (sampleType == ImageFormats::QSampleType::INT16) {
		((short*) data)[j * width + i] = value;
		return true;

	} else if (sampleType == ImageFormats::QSampleType::UINT32) {
		((unsigned int*) data)[j * width + i] = value;
		return true;
	} else if (sampleType == ImageFormats::QSampleType::INT32) {
		((int*) data)[j * width + i] = value;
		return true;

	} else if (sampleType == ImageFormats::QSampleType::FLOAT32) {
		((float*) data)[j * width + i] = value;
		return true;
	}
	return false;
}

bool TextureHelper::valueAt(const void *data, int i, int j, int width,
		ImageFormats::QSampleType sampleType, double &value) {
	if (sampleType == ImageFormats::QSampleType::UINT8) {
		value = ((const unsigned char*) data)[j * width + i];
		return true;
	} else if (sampleType == ImageFormats::QSampleType::INT8) {
		value = ((const char*) data)[j * width + i];
		return true;
	} else if (sampleType == ImageFormats::QSampleType::UINT16) {
		value = ((const unsigned short*) data)[j * width + i];
		return true;
	} else if (sampleType == ImageFormats::QSampleType::INT16) {
		value = ((const short*) data)[j * width + i];
		return true;

	} else if (sampleType == ImageFormats::QSampleType::UINT32) {
		value = ((const unsigned int*) data)[j * width + i];
		return true;
	} else if (sampleType == ImageFormats::QSampleType::INT32) {
		value = ((const int*) data)[j * width + i];
		return true;

	} else if (sampleType == ImageFormats::QSampleType::FLOAT32) {
		value = ((const float*) data)[j * width + i];
		return true;
	}
	return false;
}

void TextureHelper::valuesAlongJ(const void *data,int xoffset, int j, bool *valid,
		double *values, int width, int height,
		ImageFormats::QSampleType sampleType) {
	if (sampleType == ImageFormats::QSampleType::UINT8) {
		const unsigned char *internalBuffer = (const unsigned char*) data;
		for (int i = 0; i < width; i++) {
			values[i+xoffset] = internalBuffer[j * width + i];
			valid[i+xoffset] = true;
		}
	} else if (sampleType == ImageFormats::QSampleType::INT8) {
		const char *internalBuffer = (const char*) data;
		for (int i = 0; i < width; i++) {
			values[i+xoffset] = internalBuffer[j * width + i];
			valid[i+xoffset] = true;
		}
	} else if (sampleType == ImageFormats::QSampleType::UINT16) {
		const unsigned short *internalBuffer = (const unsigned short*) data;
		for (int i = 0; i < width; i++) {
			values[i+xoffset] = internalBuffer[j * width + i];
			valid[i+xoffset] = true;
		}
	} else if (sampleType == ImageFormats::QSampleType::INT16) {
		const short *internalBuffer = (const short*) data;
		for (int i = 0; i < width; i++) {
			values[i+xoffset] = internalBuffer[j * width + i];
			valid[i+xoffset] = true;
		}

	} else if (sampleType == ImageFormats::QSampleType::UINT32) {
		const unsigned int *internalBuffer = (const unsigned int*) data;
		for (int i = 0; i < width; i++) {
			values[i+xoffset] = internalBuffer[j * width + i];
			valid[i+xoffset] = true;
		}
	} else if (sampleType == ImageFormats::QSampleType::INT32) {
		const int *internalBuffer = (const int*) data;
		for (int i = 0; i < width; i++) {
			values[i+xoffset] = internalBuffer[j * width + i];
			valid[i+xoffset] = true;
		}

	} else if (sampleType == ImageFormats::QSampleType::FLOAT32) {
		const float *internalBuffer = (const float*) data;
		for (int i = 0; i < width; i++) {
			values[i+xoffset] = internalBuffer[j * width + i];
			valid[i+xoffset] = true;
		}
	} else {
		for (int i = 0; i < width; i++) {
			valid[i+xoffset] = false;
		}
	}
}

void TextureHelper::valuesAlongI(const void *data, int i,int yoffset, bool *valid,
		double *values, int width, int height,
		ImageFormats::QSampleType sampleType) {
	if (sampleType == ImageFormats::QSampleType::UINT8) {
		const unsigned char *internalBuffer = (const unsigned char*) data;
		for (int j = 0; j < height; j++) {
			values[j+yoffset] = internalBuffer[j * width + i];
			valid[j+yoffset] = true;
		}
	} else if (sampleType == ImageFormats::QSampleType::INT8) {
		const char *internalBuffer = (const char*) data;
		for (int j = 0; j < height; j++) {
			values[j+yoffset] = internalBuffer[j * width + i];
			valid[j+yoffset] = true;
		}
	} else if (sampleType == ImageFormats::QSampleType::UINT16) {
		const unsigned short *internalBuffer = (const unsigned short*) data;
		for (int j = 0; j < height; j++) {
			values[j+yoffset] = internalBuffer[j * width + i];
			valid[j+yoffset] = true;
		}
	} else if (sampleType == ImageFormats::QSampleType::INT16) {
		const short *internalBuffer = (const short*) data;
		for (int j = 0; j < height; j++) {
			values[j+yoffset] = internalBuffer[j * width + i];
			valid[j+yoffset] = true;
		}
	} else if (sampleType == ImageFormats::QSampleType::UINT32) {
		const unsigned int *internalBuffer = (const unsigned int*) data;
		for (int j = 0; j < height; j++) {
			values[j+yoffset] = internalBuffer[j * width + i];
			valid[j+yoffset] = true;
		}
	} else if (sampleType == ImageFormats::QSampleType::INT32) {
		const int *internalBuffer = (const int*) data;
		for (int j = 0; j < height; j++) {
			values[j+yoffset] = internalBuffer[j * width + i];
			valid[j+yoffset] = true;
		}
	} else if (sampleType == ImageFormats::QSampleType::FLOAT32) {
		const float *internalBuffer = (const float*) data;
		for (int j = 0; j < height; j++) {
			values[j+yoffset] = internalBuffer[j * width + i];
			valid[j+yoffset] = true;
		}
	} else {
		for (int j = 0; j < height; j++) {
			valid[j+yoffset] = false;
		}
	}
}
