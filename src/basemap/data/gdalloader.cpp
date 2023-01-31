#include "gdalloader.h"
#include "gdal_priv.h"

ImageFormats::QColorFormat GDALLoader::getColorFormatType(
		GDALDataset *dataset) {
	int numBands = dataset->GetRasterCount();
	if (numBands == 0)
		return ImageFormats::GRAY;
	GDALRasterBand *band = (GDALRasterBand*) dataset->GetRasterBand(1);
	if (numBands == 1) {
		//Beware could be a palette
		switch (band->GetColorInterpretation()) {
		case GDALColorInterp::GCI_PaletteIndex:
			return ImageFormats::RGBA_INDEXED;
		default:
			return ImageFormats::GRAY;
		}
	} else if (numBands == 3 || numBands == 4) {
		const char *value = dataset->GetMetadataItem("INTERLEAVE",
				"IMAGE_STRUCTURE");
		if (value == nullptr && numBands == 3)
			return ImageFormats::RGB_INTERLEAVED;
		else if (value == nullptr && numBands == 4)
			return ImageFormats::RGBA_INTERLEAVED;
		else if (value == nullptr)
			return ImageFormats::GRAY;

		std::string vString = value;
		if (vString == "PIXEL") {
			if (numBands == 3)
				return ImageFormats::RGB_INTERLEAVED;
			else if (numBands == 4)
				return ImageFormats::RGBA_INTERLEAVED;

		} else {
			if (numBands == 3)
				return ImageFormats::RGB_PLANAR;
			else if (numBands == 4)
				return ImageFormats::RGBA_PLANAR;
		}
	}
	return ImageFormats::GRAY;
}
ImageFormats::QSampleType GDALLoader::getSampleType(GDALRasterBand *hBand)
{
	ImageFormats::QSampleType sampleType = ImageFormats::QSampleType::INT8;
		if (GDALGetRasterDataType(hBand) == GDT_Byte) {
			sampleType = ImageFormats::QSampleType::UINT8;
		} else if (GDALGetRasterDataType(hBand) == GDT_UInt16) {
			sampleType = ImageFormats::QSampleType::UINT16;
		} else if (GDALGetRasterDataType(hBand) == GDT_Int16) {
			sampleType = ImageFormats::QSampleType::INT16;
		} else if (GDALGetRasterDataType(hBand) == GDT_UInt32) {
			sampleType = ImageFormats::QSampleType::UINT32;
		} else if (GDALGetRasterDataType(hBand) == GDT_Int32) {
			sampleType = ImageFormats::QSampleType::INT32;
		} else if (GDALGetRasterDataType(hBand) == GDT_Float32) {
			sampleType = ImageFormats::QSampleType::FLOAT32;
		}
		return sampleType;
}

