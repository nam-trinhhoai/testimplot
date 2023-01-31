#ifndef TextureHelper_h
#define TextureHelper_h

#include "lookuptable.h"
#include "imageformats.h"

class QOpenGLTexture;
class TextureHelper {
public:
	static QOpenGLTexture* generateLUTTexture(const LookupTable &lut);
	static QOpenGLTexture* generateTexture(const void *data, int width,
			int height, ImageFormats::QSampleType sampleType,
			ImageFormats::QColorFormat colorFormat);

	static bool setValue(void *data, int i, int j, int width,
			ImageFormats::QSampleType sampleType, double value);

	static bool valueAt(const void *data, int i, int j, int width,
			ImageFormats::QSampleType sampleType, double &value);

	static void valuesAlongI(const void *data, int i,int yoffset, bool *valid, double *values,
			int width, int height, ImageFormats::QSampleType sampleType) ;

	static void valuesAlongJ(const void *data,int xoffset, int j, bool *valid, double *values,
			int width, int height, ImageFormats::QSampleType sampleType) ;
private:
	TextureHelper() {
	}
};

#endif
