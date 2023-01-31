/*
 * Cultural.cpp
 *
 *  Created on: Apr 2, 2020
 *      Author: Georges
 *
 *   Warning we are only adressing the mono survey3D problems
 *
 */

#include "cultural.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <png.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

#include <boost/algorithm/string.hpp>


#include "smtopo3ddesc.h"
#include <QByteArray>
#include <QRect>
#include <QFileInfo>
#include <QDir>
#include <QDebug>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <boost/filesystem.hpp>

#include "sismagedbmanager.h"
#include "smsurvey3D.h"
#include "utils/ioutil.h"
#include "utils/stringutil.h"
#include "LayerSlice.h"
#include "rgblayerslice.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "cudargbimage.h"
#include "cudaimagepaletteholder.h"
#include "culturalcategory.h"
#include "sampletypebinder.h"

namespace fs = boost::filesystem;


std::string PNG(".png");
std::string GRI(".gri");

/**
 * "desc" file describe Cultural attributes. It was located in LAYER DESC DIRECTORY
 *
 * dirName is the directory of Cultural inside one single survey3D
 */
Cultural::Cultural(const std::string& cdatFilePath) : m_cdatFilePath( cdatFilePath) {
	// correct name
	QFileInfo culturalFileInfo(QString::fromStdString(cdatFilePath));
	std::string culturalName = culturalFileInfo.baseName().toStdString();
	std::string correctName = SismageDBManager::fixCulturalName(culturalName);
	correctName += "." + culturalFileInfo.completeSuffix().toStdString();
	m_cdatFilePath = culturalFileInfo.dir().absoluteFilePath(QString::fromStdString(correctName)).toStdString();

	readCdatFile(m_cdatFilePath);

	fs::path p(m_cdatFilePath);
	p.replace_extension(PNG);
	m_pngFilePath = p.c_str();

	fs::path pgri(m_cdatFilePath);
	pgri.replace_extension(GRI);
	m_griFilePath = pgri.c_str();
}

/**
 * For new cultural that will be created by NextVision.
 * Specification NK (3/12/2020) file name should be equal to the name of the cultural
 */
Cultural::Cultural(const std::string& sismageCulturalsPath,
		const std::string& culturalUIName, CulturalCategory& culturalCategory) :
				m_sismageCulturalsPath( sismageCulturalsPath) {

	fs::path genericPath(sismageCulturalsPath);
	m_name = SismageDBManager::fixCulturalName(culturalUIName);
	genericPath /= m_name;
	std::cout << "Generic PATH= "<< genericPath.c_str() << std::endl;

	m_cdatFilePath = genericPath.c_str();
	m_cdatFilePath.append( ".cdat" );
	std::cout << "cdat PATH= "<< m_cdatFilePath << std::endl;

	m_griFilePath = genericPath.c_str();
	m_griFilePath.append( ".gri");
	std::cout << "gri PATH= "<< m_griFilePath << std::endl;

	m_pngFilePath = genericPath.c_str();
	m_pngFilePath.append( ".png");
	std::cout << "png PATH= "<< m_pngFilePath << std::endl;

	// .cdat
	writeCdatFile(m_cdatFilePath, culturalCategory);

	//.gri will be created by saveInto
	//fs::copy_file(refCultural->getGriFilePath(), m_griFilePath);
}

Cultural::~Cultural() {
	// TODO Auto-generated destructor stub
}

/**
 * Example:
RGB_2
GEOREF IMAGE
Sismage Main	IDL:DmCultural/ExistingCategoryFactory:1.0	dKxEyhb9

 *
 */
bool Cultural::readCdatFile(const std::string& filename) {

	std::string line;
	std::ifstream myfile( filename.c_str() );
	if (myfile.is_open()) {
		std::getline (myfile,line);
		m_name = line.c_str();

		std::getline (myfile,line);
		m_type = line.c_str();

		std::getline (myfile,line);
		std::string m_categoryLine = line.c_str();
		if (!m_categoryLine.empty()) {
			std::vector<std::string> strs;
			boost::split(strs, m_categoryLine, boost::is_any_of("\t "));

			if (strs.size() > 1) {
				m_category = strs.at(strs.size()-1);
			}
		}

		myfile.close();
	}
	else {
		return false;
		//throw io::CubeIOException(std::string("Unable to open file ") + filename.c_str());
	}
	return true;
}

/**
 *
 */
void Cultural::writeCdatFile(const std::string& filename, CulturalCategory& existingCategory) {

	std::ofstream myFile( filename.c_str() );
	if (myFile.is_open()) {
		myFile << m_name << std::endl;
		myFile << "GEOREF IMAGE" << std::endl;
		myFile << "Sismage Main\tIDL:DmCultural/ExistingCategoryFactory:1.0\t" <<
				existingCategory.getSismageId() << std::endl;

		myFile.close();
	}
}


static pixel_t * pixel_at (bitmap_t *bitmap, int x, int y) {
        return bitmap->pixels + x + bitmap->width * y;
}

int Cultural::save_png_to_file (bitmap_t *bitmap, const char* path) {
        FILE *p_file;
        png_structp     p_png = nullptr;
        png_infop p_info = nullptr;
        size_t x, y;
        png_byte ** row_pointers = nullptr;
        int status = -1;
        int pixel_size = 3;
        int depth = 8;

        p_file = fopen(path, "wb");
        if (!p_file) {
                goto fopen_failed;
        }

        p_png = png_create_write_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if (p_png == NULL) {
                goto png_create_write_struct_failed;
        }

        p_info = png_create_info_struct (p_png);
        if (p_info == NULL) {
                goto png_create_info_struct_failed;
        }

        if (setjmp (png_jmpbuf (p_png))) {
                goto png_failure;
        }

        png_set_IHDR (  p_png,
						p_info,
						bitmap->width,
						bitmap->height,
						depth,
						PNG_COLOR_TYPE_RGB,
						PNG_INTERLACE_NONE,
						PNG_COMPRESSION_TYPE_DEFAULT,
						PNG_FILTER_TYPE_DEFAULT);
        row_pointers = (png_byte**)png_malloc (p_png, bitmap->height * sizeof (png_byte*));

        for (y = 0; y < bitmap->height; ++y) {
                png_byte *row = (png_byte*)png_malloc (p_png, sizeof (uint8_t) * bitmap->width * pixel_size);
                row_pointers[y] = row;
                for (x = 0; x < bitmap->width; ++x) {
                        pixel_t *pixel = pixel_at (bitmap, x, y);
                        *row++ = pixel->red;
                        *row++ = pixel->green;
                        *row++ = pixel->blue;
                }
        }

        png_init_io (p_png, p_file);
        png_set_rows (p_png, p_info, row_pointers);
        png_write_png (p_png, p_info, PNG_TRANSFORM_IDENTITY, NULL);
        status = 0;

        for (y = 0; y < bitmap->height; y++) {
                png_free (p_png, row_pointers[y]);
        }

        png_free (p_png, row_pointers);

png_failure:

png_create_info_struct_failed:
        png_destroy_write_struct(&p_png, &p_info);

png_create_write_struct_failed:
        fclose(p_file);

fopen_failed:
        return status;

}

unsigned char Cultural::getComponent(float val, const QVector2D& range) {
	val = 255.0 * (val - range.x()) / (range.y() - range.x());
	if (val<0) {
		val = 0;
	} else if (val>255) {
		val = 255;
	}
	return static_cast<unsigned char>(val);
}


template<typename DataType>
struct CopyCUDARGBDataIntoPngImageKernel {
	static void run(bitmap_t& bitmap, CUDAImagePaletteHolder* red, CUDAImagePaletteHolder* green,
			CUDAImagePaletteHolder* blue, int minValue, long dimW, long dimH) {
		QVector2D redRange = red->range();
		QVector2D greenRange = green->range();
		QVector2D blueRange = blue->range();

		pixel_t *pixel_out = bitmap.pixels;

		red->lockPointer();
		green->lockPointer();
		blue->lockPointer();
		DataType* redBuf = static_cast<DataType*>(red->backingPointer());
		DataType* greenBuf = static_cast<DataType*>(green->backingPointer());
		DataType* blueBuf = static_cast<DataType*>(blue->backingPointer());

		for (int h = 0; h < dimH; h++) {
			for (int w = 0; w < dimW; w++) {
				int iIn = (h * dimW + w);
				int iOut = (w * dimH + h);

				QColor color = QColor::fromRgb(Cultural::getComponent(redBuf[iIn], redRange),
						Cultural::getComponent(greenBuf[iIn], greenRange),
						Cultural::getComponent(blueBuf[iIn], blueRange));

				int hue;
				int saturation;
				int value;
				color.getHsv(&hue, &saturation, &value);
				value = std::max(value, minValue);

				QColor saturatedColor = QColor::fromHsv(hue, saturation, value);
				pixel_out[iOut].red = saturatedColor.red();
				pixel_out[iOut].green = saturatedColor.green();
				pixel_out[iOut].blue = saturatedColor.blue();
			}
		}
		red->unlockPointer();
		green->unlockPointer();
		blue->unlockPointer();
	}
};

/**
 * Save all processed maps on Sismage Layer
 *
 * TODO: Should be revisited when SeismicAddon will be completed
 */
void Cultural::saveInto(RGBLayerSlice* visual, bool create) {

//	gui::ToBeAnalysedImage2* imageRgb = visual->getCube();
//	if(!imageRgb)
//		return;

	//const data::ToBeAnalysedBaseData2* data = imageRgb->getData();
	int dimW = visual->layerSlice()->width();
	int dimH = visual->layerSlice()->depth();
	QRect sourceRegion(0, 0, dimW, dimH);

//	RawImage1 * rawImage  = imageRgb->ReadData1( sourceRegion );
//	QByteArray& buf = rawImage->buffer;
//	unsigned char* charTab = static_cast<unsigned char*> (static_cast<void*>(buf.data()));
	CUDARGBImage* image = visual->image();
	CUDAImagePaletteHolder* red = image->get(0);
	CUDAImagePaletteHolder* green = image->get(1);
	CUDAImagePaletteHolder* blue = image->get(2);

	QVector2D redRange = red->range();
	QVector2D greenRange = green->range();
	QVector2D blueRange = blue->range();

	bitmap_t bitmap;
    bitmap.width = dimH; // reversed
    bitmap.height = dimW;
	bitmap.pixels = new pixel_t[dimW * dimH];

	//uint8_t *pixel_in = charTab;
	pixel_t *pixel_out = bitmap.pixels;

	image->lockPointer();
	float* redBuf = static_cast<float*>(red->backingPointer());
	float* greenBuf = static_cast<float*>(green->backingPointer());
	float* blueBuf = static_cast<float*>(blue->backingPointer());

	int minValue = 0;
	if (visual->isMinimumValueActive()) {
		minValue = std::floor(255 * visual->minimumValue());
	}

	for (int h = 0; h < dimH; h++) {
		for (int w = 0; w < dimW; w++) {
			int iIn = (h * dimW + w);
			int iOut = (w * dimH + h);
			QColor color = QColor::fromRgb(getComponent(redBuf[iIn], redRange),
					getComponent(greenBuf[iIn], greenRange),
					getComponent(blueBuf[iIn], blueRange));

			int hue;
			int saturation;
			int value;
			color.getHsv(&hue, &saturation, &value);
			value = std::max(value, minValue);

			QColor saturatedColor = QColor::fromHsv(hue, saturation, value);

			pixel_out[iOut].red  = saturatedColor.red();
			pixel_out[iOut].green = saturatedColor.green();
			pixel_out[iOut].blue   = saturatedColor.blue();
		}
	}
	image->unlockPointer();

	int retval = save_png_to_file (&bitmap, m_pngFilePath.c_str());

	if (retval != 0) {
			printf ("Error saving PNG.\n");
			//goto free_val;
	}

	//????free(buffer);
	delete (bitmap.pixels);

	if (create) {
		// Maintenant le .gri
		std::ofstream myFile( m_griFilePath.c_str() );
		if (myFile.is_open()) {
			myFile << m_name << std::endl;
			myFile << "GeorefImage" << std::endl;
			myFile << "3" << std::endl;

			// dimH -> nbInlines -> j dans la transfo
			// dimW -> nbXlines -> i dans la transfo
			const Affine2DTransformation *ijToXYTransfo = visual->layerSlice()->seismic()->ijToXYTransfo();
			double worldX, worldY;
			ijToXYTransfo->imageToWorld(0, 0, worldX, worldY);
			myFile << std::fixed;
			myFile << std::setprecision(9);
			myFile << worldX << "\t" << worldY << "\t0.0" << std::endl;
			myFile << "0.0" << "\t0.0" << "\t0.0" << std::endl;
			ijToXYTransfo->imageToWorld( 0, dimH - 1, worldX, worldY);
			myFile << worldX << "\t" << worldY << "\t0.0" << std::endl;
			myFile << std::setprecision(1);
			myFile << (dimH - 1) << "\t0.0" << "\t0.0" << std::endl;
			ijToXYTransfo->imageToWorld( dimW - 1, 0, worldX, worldY);
			myFile << std::setprecision(9);
			myFile << worldX << "\t" << worldY << "\t0.0" << std::endl;
			myFile << std::setprecision(1);
			myFile << "0.0" << "\t" << (dimW - 1) << "\t0.0" << std::endl;
		}
		myFile.close();
	}
}

template<typename DataType>
struct CopyGrayDataIntoPngImageKernel {
	static void run(bitmap_t& bitmap, CPUImagePaletteHolder* image, long dimW, long dimH) {
		QVector2D redRange = image->range();

		pixel_t *pixel_out = bitmap.pixels;

		image->lockPointer();
		DataType* imageBuf = static_cast<DataType*>(image->backingPointer());

		for (int h = 0; h < dimH; h++) {
			for (int w = 0; w < dimW; w++) {
				int iIn = (h * dimW + w);
				int iOut = (w * dimH + h);

				int val = Cultural::getComponent(imageBuf[iIn], redRange);
				pixel_out[iOut].red = val;
				pixel_out[iOut].green = val;
				pixel_out[iOut].blue = val;
			}
		}
		image->unlockPointer();
	}
};

void Cultural::saveInto(CPUImagePaletteHolder* image,
		const Affine2DTransformation* ijToXYTransfo, bool create) {
	int dimW = image->width();
	int dimH = image->height();
	QRect sourceRegion(0, 0, dimW, dimH);

	bitmap_t bitmap;
	bitmap.width = dimH; // reversed
	bitmap.height = dimW;
	bitmap.pixels = new pixel_t[dimW * dimH];

	SampleTypeBinder binder(image->sampleType());
	binder.bind<CopyGrayDataIntoPngImageKernel>(bitmap, image, dimW, dimH);

	int retval = save_png_to_file (&bitmap, m_pngFilePath.c_str());

	if (retval != 0) {
			printf ("Error saving PNG.\n");
	}

	delete[] (bitmap.pixels);

	if (create) {
		// Maintenant le .gri
		std::ofstream myFile( m_griFilePath.c_str() );
		if (myFile.is_open()) {
			myFile << m_name << std::endl;
			myFile << "GeorefImage" << std::endl;
			myFile << "3" << std::endl;

			// dimH -> nbInlines -> j dans la transfo
			// dimW -> nbXlines -> i dans la transfo
			double worldX, worldY;
			ijToXYTransfo->imageToWorld(0, 0, worldX, worldY);
			myFile << std::fixed;
			myFile << std::setprecision(9);
			myFile << worldX << "\t" << worldY << "\t0.0" << std::endl;
			myFile << "0.0" << "\t0.0" << "\t0.0" << std::endl;
			ijToXYTransfo->imageToWorld( 0, dimH - 1, worldX, worldY);
			myFile << worldX << "\t" << worldY << "\t0.0" << std::endl;
			myFile << std::setprecision(1);
			myFile << (dimH - 1) << "\t0.0" << "\t0.0" << std::endl;
			ijToXYTransfo->imageToWorld( dimW - 1, 0, worldX, worldY);
			myFile << std::setprecision(9);
			myFile << worldX << "\t" << worldY << "\t0.0" << std::endl;
			myFile << std::setprecision(1);
			myFile << "0.0" << "\t" << (dimW - 1) << "\t0.0" << std::endl;
		}
		myFile.close();
	}
}

/**
 * Save all processed maps on Sismage Layer
 *
 * TODO: Should be revisited when SeismicAddon will be completed
 */
void Cultural::saveInto(CUDAImagePaletteHolder *red,
		CUDAImagePaletteHolder *green, CUDAImagePaletteHolder *blue,
		int minValue, const Affine2DTransformation *ijToXYTransfo,
		bool create) {

	int dimW = red->width();
	int dimH = red->height();
	std::cout << "dimW= " << dimW << "dimH= " << dimH << std::endl;
	QRect sourceRegion(0, 0, dimW, dimH);


	QVector2D redRange = red->range();
	QVector2D greenRange = green->range();
	QVector2D blueRange = blue->range();

	bitmap_t bitmap;
    bitmap.width = dimH; // reversed
    bitmap.height = dimW;
	bitmap.pixels = new pixel_t[dimW * dimH];

	//uint8_t *pixel_in = charTab;
	pixel_t *pixel_out = bitmap.pixels;

	SampleTypeBinder binder(red->sampleType());
	binder.bind<CopyCUDARGBDataIntoPngImageKernel>(bitmap, red, green, blue, minValue, dimW, dimH);

	int retval = save_png_to_file (&bitmap, m_pngFilePath.c_str());

	if (retval != 0) {
			printf ("Error saving PNG.\n");
			//goto free_val;
	}

	//????free(buffer);
	delete (bitmap.pixels);
	// Autre chose a detruire ? GS 7/12/2020

	// Maintenant le .gri
	if (create) {
		std::ofstream myFile( m_griFilePath.c_str() );
		if (myFile.is_open()) {
			myFile << m_name << std::endl;
			myFile << "GeorefImage" << std::endl;
			myFile << "3" << std::endl;

			// dimH -> nbInlines -> j dans la transfo
			// dimW -> nbXlines -> i dans la transfo
			double worldX, worldY;
			ijToXYTransfo->imageToWorld(0, 0, worldX, worldY);
			myFile << std::fixed;
			myFile << std::setprecision(9);
			myFile << worldX << "\t" << worldY << "\t0.0" << std::endl;
			myFile << "0.0" << "\t0.0" << "\t0.0" << std::endl;
			ijToXYTransfo->imageToWorld( 0, dimH - 1, worldX, worldY);
			myFile << worldX << "\t" << worldY << "\t0.0" << std::endl;
			myFile << std::setprecision(1);
			myFile << (dimH - 1) << "\t0.0" << "\t0.0" << std::endl;
			ijToXYTransfo->imageToWorld( dimW - 1, 0, worldX, worldY);
			myFile << std::setprecision(9);
			myFile << worldX << "\t" << worldY << "\t0.0" << std::endl;
			myFile << std::setprecision(1);
			myFile << "0.0" << "\t" << (dimW - 1) << "\t0.0" << std::endl;
		}
		myFile.close();
	}
}
