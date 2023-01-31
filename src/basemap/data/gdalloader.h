#ifndef GDALLoader_h
#define GDALLoader_h

#include "imageformats.h"

class GDALDataset;
class GDALRasterBand;

class GDALLoader {
public:
	virtual ~GDALLoader();
	static ImageFormats::QColorFormat getColorFormatType(GDALDataset * dataset);
	static ImageFormats::QSampleType getSampleType(GDALRasterBand *hBand);
private :
	GDALLoader();
};

#endif
