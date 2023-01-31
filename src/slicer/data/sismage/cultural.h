/*
 * Cultural.h
 *
 *  Created on: Apr 2, 2020
 *      Author: l0222891
 */

#ifndef TARUMAPP_SRC_DATA_CULTURAL_H_
#define TARUMAPP_SRC_DATA_CULTURAL_H_

#include <string>

#include <QVector2D>

class RGBLayerSlice;
class FixedRGBLayersFromDatasetAndCube;
class CPUImagePaletteHolder;
class CUDAImagePaletteHolder;
class CulturalCategory;
class Affine2DTransformation;
//class Seismic3DDataset;

typedef struct {
        uint8_t red;
        uint8_t green;
        uint8_t blue;
} pixel_t;

typedef struct {
        pixel_t *pixels;
        size_t width;
        size_t height;
} bitmap_t;

class Cultural {
public:
	Cultural(const std::string& dirName);
	Cultural(const std::string& sismageCulturalsPath,
			const std::string& culturalUIName, CulturalCategory& culturalCategory);
	virtual ~Cultural();

	// apply of rgb -> hsv -> value minimal -> rgb does not create an image identical to shader rendering
	// this need to be investigated
	void saveInto(RGBLayerSlice* visual, bool create=false);
	void saveInto(CPUImagePaletteHolder* image, const Affine2DTransformation* ijToXYTransfo,
			bool create=false);
	void saveInto(CUDAImagePaletteHolder *red,
			CUDAImagePaletteHolder *green, CUDAImagePaletteHolder *blue,
			int minValue, const Affine2DTransformation *ijToXYTransfo, bool create=false);

	const std::string& getName() const {
		return m_name;
	}

	const std::string& getType() const {
		return m_type;
	}

	int getDimH() const {
		return m_dimH;
	}

	int getDimW() const {
		return m_dimW;
	}

	const std::string& getCategory() const {
		return m_category;
	}

	static int save_png_to_file (bitmap_t *bitmap, const char* path);
	static unsigned char getComponent(float val, const QVector2D& range);

	const std::string& getGriFilePath() const {
		return m_griFilePath;
	}

private:
	bool init();
	bool readCdatFile(const std::string& filename);
	void writeCdatFile(const std::string& filename, CulturalCategory& culturalCategory);

	std::string m_cdatFilePath;
	std::string m_griFilePath;
	std::string m_pngFilePath;

	std::string m_sismageCulturalsPath;

	std::string m_name;
	std::string m_type;
	std::string m_category;
	int m_dimW = 1;
	int m_dimH = 1;
};


#endif /* TARUMAPP_SRC_DATA_CULTURAL_H_ */
