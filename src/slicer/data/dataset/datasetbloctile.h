#ifndef DatasetBlocTile_H
#define DatasetBlocTile_H

#include <string>
#include <sstream>
#include <boost/iostreams/device/mapped_file.hpp>
#include <QString>
#include "datasethashkey.h"
#include "imageformats.h"

class DatasetBlocTile {
public:
	DatasetBlocTile(const std::string &path, int channel, size_t headerLength, int w, int h,
			int d0, int d1, int dimV, ImageFormats::QSampleType sampleType);
	~DatasetBlocTile();

	int d0() const {
		return m_d0;
	}

	int d1() const {
		return m_d1;
	}

	size_t blocSize() const {
		return m_blocSize;
	}

	int depth() const {
		return m_depth;
	}

	DatasetHashKey key() const;

	size_t memoryCost();

	size_t dataStartOffset() const {
		return m_delta;
	}

	ImageFormats::QSampleType sampleType() const {
		return m_sampleType;
	}

	void* buffer();

private:
	size_t roundDown(size_t numToRound, size_t multiple);
	size_t roundUp(size_t numToRound, size_t multiple);
private:
	int m_d0;
	int m_d1;
	int m_h;
	int m_w;
	int m_depth;
	int m_dimV;
	int m_channel;
	ImageFormats::QSampleType m_sampleType;


	std::string m_path;
	bool m_fileOpened;
	//boost::iostreams::mapped_file seismic_file;

	//Mapped IO implies to align correctly memory, we need to apply a small delta to correct alignment if needed
	size_t m_delta;

	size_t m_realOffset;
	size_t m_blocSize;

	size_t m_memoryCost;

	void *m_buffer;
};

#endif
