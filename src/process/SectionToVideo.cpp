
#include <QDir>
#include <QDebug>

#include "SectionToVideo.h"

#include "Xt.h"
#include <freeHorizonManager.h>
#include "datasetrelatedstorageimpl.h"
#include "nextvisiondbmanager.h"
#include <rgtSpectrumHeader.h>

#include "CubeIO.h"

#include <cstdio>

SectionToVideo::SectionToVideo() {
	m_numInline = 0;
	m_numXline = 0;
	m_sectionIndex = -1;
	m_sizeDefined = false;
	m_sampleStep = 1;
	m_sampleOrigin = 0;
}

SectionToVideo::~SectionToVideo() {

}

bool SectionToVideo::setDatasetPath(const QString& inDatasetPath) {
	inri::Xt xtFile(inDatasetPath.toStdString());

	bool valid = xtFile.is_valid();
	if (valid && !m_rgb2Path.isNull() && !m_rgb2Path.isEmpty()) {
		valid = compatible(inDatasetPath, m_rgb2Path);
	}

	if (valid) {
		m_datasetPath = inDatasetPath;
		m_sampleOrigin = xtFile.startSamples();
		m_sampleStep = xtFile.stepSamples();
		if (!m_sizeDefined) {
			m_numInline = xtFile.nSlices();
			m_numXline = xtFile.nRecords();
			m_sizeDefined = true;
		}
	}

	return valid;
}

bool SectionToVideo::setSection(int sectionIndex, SliceDirection dir) {
	bool out = m_sizeDefined;
	if (out) {
		if (dir==SliceDirection::Inline) {
			out = sectionIndex>=0 && sectionIndex<m_numInline;
		} else if (dir==SliceDirection::XLine) {
			out = sectionIndex>=0 && sectionIndex<m_numXline;
		}
	}

	if (out) {
		m_sectionIndex = sectionIndex;
		m_direction = dir;
	}

	return out;
}

bool SectionToVideo::setRgb2(const QString& inRgb2Path) {
	bool valid = true;
	if (!m_datasetPath.isNull() && !m_datasetPath.isEmpty()) {
		valid = compatible(m_datasetPath, inRgb2Path);
	}

	if (valid) {
		m_rgb2Path = inRgb2Path;
		if (!m_sizeDefined) {
			QString rgb2BaseDataset = getDatasetPathFromRgb2(inRgb2Path);
			inri::Xt xtFile(rgb2BaseDataset.toStdString()); // if valid is true, the xt file should be valid
			m_numInline = xtFile.nSlices();
			m_numXline = xtFile.nRecords();
			m_sizeDefined = true;
		}
	}

	return valid;
}


bool SectionToVideo::setRgb2(const QString& seismicPath, const QString& inRgb2Path) {
	if (!m_sizeDefined) {
		inri::Xt xtFile(seismicPath.toStdString()); // if valid is true, the xt file should be valid
		m_numInline = xtFile.nSlices();
		m_numXline = xtFile.nRecords();
		m_sizeDefined = true;
	}
	return true;
	/*
	bool valid = true;
	if (!m_datasetPath.isNull() && !m_datasetPath.isEmpty()) {
		valid = compatible(m_datasetPath, inRgb2Path);
	}

	if (valid) {
		m_rgb2Path = inRgb2Path;
		if (!m_sizeDefined) {
			QString rgb2BaseDataset = getDatasetPathFromRgb2(inRgb2Path);
			inri::Xt xtFile(rgb2BaseDataset.toStdString()); // if valid is true, the xt file should be valid
			m_numInline = xtFile.nSlices();
			m_numXline = xtFile.nRecords();
			m_sizeDefined = true;
		}
	}

	return valid;
	 */
}


void SectionToVideo::setIsoPath(const std::vector<std::string> &isoPath) {
	m_isoPath = isoPath;
}

void SectionToVideo::setSeismicName(const QString& name)
{
	m_seismicName = name;
}



void SectionToVideo::setOutputPath(const QString& outputVideoPath) {
	m_outputPath = outputVideoPath;
}

QString SectionToVideo::getDatasetPathFromRgb2(const QString& rgb2Path) {
	QString surveyPath = QString::fromStdString(NextVisionDBManager::getSurvey3DPathFromRgb2Path(rgb2Path.toStdString()));
	QString seismicSismageName = QString::fromStdString(NextVisionDBManager::getSeismicSismageNameFromRgb2Path(rgb2Path.toStdString()));
	QString rgb2BaseDataset = DatasetRelatedStorageImpl::getDatasetRealPath(surveyPath, seismicSismageName);
	return rgb2BaseDataset;
}

bool SectionToVideo::compatible(const QString& seismicPath, const QString& rgb2Path) {
	QString rgb2BaseDataset = getDatasetPathFromRgb2(rgb2Path);

	bool isCubeCompatible = !seismicPath.isNull() && !seismicPath.isEmpty() &&
			!rgb2BaseDataset.isNull() && !rgb2BaseDataset.isEmpty() && seismicPath.compare(rgb2BaseDataset)==0;

	// check compatibility on carte (inline & xline)
	if (!isCubeCompatible) {
		std::size_t oriNbTraces, oriNbProfiles;
		float oriFirstTraces, oriPasTraces, oriFirstProfiles, oriPasProfiles;
		std::size_t rgb2NbTraces, rgb2NbProfiles;
		float rgb2FirstTraces, rgb2PasTraces, rgb2FirstProfiles, rgb2PasProfiles;

		inri::Xt oriXt(seismicPath.toStdString().c_str());
		if (!oriXt.is_valid()) {
			isCubeCompatible = false;
		} else {
			oriNbTraces = oriXt.nRecords();
			oriNbProfiles = oriXt.nSlices();
			oriFirstTraces = oriXt.startRecord();
			oriPasTraces = oriXt.stepRecords();
			oriFirstProfiles = oriXt.startSlice();
			oriPasProfiles = oriXt.stepSlices();

			inri::Xt rgb2Xt(rgb2BaseDataset.toStdString().c_str());
			if (!rgb2Xt.is_valid()) {
				isCubeCompatible = false;
			} else {
				rgb2NbTraces = rgb2Xt.nRecords();
				rgb2NbProfiles = rgb2Xt.nSlices();
				rgb2FirstTraces = rgb2Xt.startRecord();
				rgb2PasTraces = rgb2Xt.stepRecords();
				rgb2FirstProfiles = rgb2Xt.startSlice();
				rgb2PasProfiles = rgb2Xt.stepSlices();
			}
		}

		isCubeCompatible = oriNbTraces==rgb2NbTraces && oriNbProfiles==rgb2NbProfiles && oriFirstProfiles==rgb2FirstProfiles &&
				oriPasProfiles==rgb2PasProfiles && oriFirstTraces==rgb2FirstTraces && oriPasTraces==rgb2PasTraces;
	}
	return isCubeCompatible;
}

bool SectionToVideo::run(const QString& inDatasetPath, int sectionIndex,
		SliceDirection dir, const QString& inRgb2Path, const QString& outVideoPath) {
	SectionToVideo process;
	bool valid = process.setDatasetPath(inDatasetPath);
	valid = valid && process.setSection(sectionIndex, dir);
	valid = valid && process.setRgb2(inRgb2Path);
	process.setOutputPath(outVideoPath);

	valid = valid && process.computeVideo(QColor(Qt::green), 2);

	return valid;
}

bool SectionToVideo::run2(const std::vector<std::string>& isoPath, const QString& inDatasetPath, const QString &seismicName, int sectionIndex,
		SliceDirection dir, const QString& inRgb2Path,
		const QString& outVideoPath) {
	SectionToVideo process;
	bool valid = process.setDatasetPath(inDatasetPath);
	valid = valid && process.setSection(sectionIndex, dir);
	valid = valid && process.setRgb2(inDatasetPath, inRgb2Path);
	process.setOutputPath(outVideoPath);
	process.setIsoPath(isoPath);
	process.setSeismicName(seismicName);
	valid = valid && process.computeVideo2(QColor(Qt::green), 2);
	return valid;
}


template<typename InputType>
struct ExtractSectionToRgbBufferKernel {
	static void run(std::vector<unsigned char>& rgbBuf, const QString& datasetPath, int index, SliceDirection dir) {
		std::unique_ptr<const murat::io::InputOutputCube<InputType>> cube;
		cube.reset(murat::io::openCube<InputType>(datasetPath.toStdString()));
		murat::io::CubeDimension dim = cube->getDim();

		InputType* tab = nullptr;
		std::size_t axisSize = 0;
		if (dir==SliceDirection::Inline) {
			tab = cube->readSubVolume(0, 0, index, dim.getI(), dim.getJ(), 1);
			axisSize = dim.getJ();
		} else {
			tab = cube->readSubVolume(0, index, 0, dim.getI(), 1, dim.getK());
			axisSize = dim.getK();
		}
		std::size_t N = dim.getI() * axisSize;

		InputType min = std::numeric_limits<InputType>::max();
		InputType max = std::numeric_limits<InputType>::lowest();

		double fileMin, fileMax;
		bool fileRangeValid = SectionToVideo::extractRange(datasetPath.toStdString(), fileMin, fileMax);

		if (!fileRangeValid) {
			// search min max
			for (std::size_t i=0; i<N; i++) {
				InputType val = tab[i];
				if (val<min) {
					min = val;
				}
				if (val>max) {
					max = val;
				}
			}
			if (min==max) {
				if (min<=std::numeric_limits<InputType>::max()-1) {
					max = min + 1;
				} else {
					min = max -1;
				}
			} else {
				InputType symMax = SectionToVideo::getSymetricMax(min, max);
				max = symMax;
				min = -symMax;
			}
		} else {
			if (fileMax>std::numeric_limits<InputType>::max()) {
				max = std::numeric_limits<InputType>::max();
			} else {
				max = fileMax;
			}
			if (fileMin<std::numeric_limits<InputType>::lowest()) {
				min = std::numeric_limits<InputType>::lowest();
			} else {
				min = fileMin;
			}
		}

		rgbBuf.resize(N*3);
		for (std::size_t j=0; j<axisSize; j++) {
			for (std::size_t i=0; i<dim.getI(); i++) {
				// swap to have video presented correctly
				std::size_t inK = i + j * dim.getI();
				std::size_t outK = j + i * axisSize;
				double val = tab[inK];
				// for symetric min max : 0 -> 128
				// max -> 255
				// min -> 1
				val = (val - min) / (max - min) * 254.0f+1;
				// and reverse
				// max -> 1
				// min -> 255
				val = std::round(256.0-val);
				if (val<0.0) {
					val = 0.0;
				} else if (val>255.0) {
					val = 255.0;
				}
				unsigned char charVal = static_cast<unsigned char>(val);
				rgbBuf[outK*3] = charVal;
				rgbBuf[outK*3+1] = charVal;
				rgbBuf[outK*3+2] = charVal;
			}
		}
	}
};

bool SectionToVideo::computeVideo(const QColor& curveColor, int penSize) {
	bool valid = !m_datasetPath.isNull() && !m_datasetPath.isEmpty() &&
			!m_rgb2Path.isNull() && !m_rgb2Path.isEmpty() &&
			!m_outputPath.isNull() && !m_outputPath.isEmpty() &&
			m_sectionIndex>=0 && penSize>=1;

	const murat::io::Cube* cube = murat::io::openCube<short>(m_datasetPath.toStdString());
	murat::io::SampleType sampleType = cube->getNativeType();

	long numSamples = cube->getDim().getI();
	long sectionAxeSize;
	if (m_direction==SliceDirection::Inline) {
		sectionAxeSize = cube->getDim().getJ();
	} else {
		sectionAxeSize = cube->getDim().getK();
	}

	delete cube;

	std::vector<unsigned char> rgbSection;

	// create
	murat::io::SampleTypeBinder binder(sampleType);
	binder.bind<ExtractSectionToRgbBufferKernel>(rgbSection, m_datasetPath, m_sectionIndex, m_direction);

	long numMaps = getRgb2NumMaps();
	//	std::unique_ptr<murat::io::InputOutputCube<unsigned char>> video;
	//	video.reset(murat::io::openOrCreateCube<unsigned char>(outputPath.toStdString(), 3*numSamples, sectionAxeSize, numMaps));

	FILE* fp = fopen(m_outputPath.toStdString().c_str(), "w");
	if (fp!=nullptr) {
		for (long i=0; i<numMaps; i++) {
			std::vector<int> isoTab = getRgb2IsoTab(i);

			std::vector<unsigned char> restoreCache;
			restoreCache.resize(isoTab.size()*penSize);

			// prepare buffer
			for (long j=0; j<sectionAxeSize; j++) {
				long index = isoTab[j];
				if (index<0) {
					index = 0;
				} else if (index>numSamples-1) {
					index = numSamples - 1;
				}
				for (int penOffset=0; penOffset<penSize; penOffset++) {

					long indexOffset = index + penOffset - penSize/2;
					if (indexOffset>=0 && indexOffset<numSamples) {
						unsigned char oldVal = rgbSection[(j+indexOffset*sectionAxeSize)*3];
						restoreCache[j*penSize+penOffset] = oldVal;
						rgbSection[(j+indexOffset*sectionAxeSize)*3] = curveColor.red();
						rgbSection[(j+indexOffset*sectionAxeSize)*3+1] = curveColor.green();
						rgbSection[(j+indexOffset*sectionAxeSize)*3+2] = curveColor.blue();
					}
				}
			}

			// write
			fwrite(rgbSection.data(), sizeof(unsigned char), rgbSection.size(), fp);

			// restore
			for (long j=0; j<sectionAxeSize; j++) {
				long index = isoTab[j];
				if (index<0) {
					index = 0;
				} else if (index>numSamples-1) {
					index = numSamples - 1;
				}
				for (int penOffset=0; penOffset<penSize; penOffset++) {
					long indexOffset = index + penOffset - penSize/2;
					if (indexOffset>=0 && indexOffset<numSamples) {
						rgbSection[(j+indexOffset*sectionAxeSize)*3] = restoreCache[j*penSize+penOffset];
						rgbSection[(j+indexOffset*sectionAxeSize)*3+1] = restoreCache[j*penSize+penOffset];
						rgbSection[(j+indexOffset*sectionAxeSize)*3+2] = restoreCache[j*penSize+penOffset];
					}
				}
			}
		}
		fclose(fp);
	}


	return valid;
}

bool SectionToVideo::computeVideo2(const QColor& curveColor, int penSize) {
	bool valid = !m_datasetPath.isNull() && !m_datasetPath.isEmpty() &&
			!m_rgb2Path.isNull() && !m_rgb2Path.isEmpty() &&
			!m_outputPath.isNull() && !m_outputPath.isEmpty() &&
			m_sectionIndex>=0 && penSize>=1;

	const murat::io::Cube* cube = murat::io::openCube<short>(m_datasetPath.toStdString());
	murat::io::SampleType sampleType = cube->getNativeType();

	long numSamples = cube->getDim().getI();
	long sectionAxeSize;
	if (m_direction==SliceDirection::Inline) {
		sectionAxeSize = cube->getDim().getJ();
	} else {
		sectionAxeSize = cube->getDim().getK();
	}

	delete cube;

	std::vector<unsigned char> rgbSection;

	// create
	murat::io::SampleTypeBinder binder(sampleType);
	binder.bind<ExtractSectionToRgbBufferKernel>(rgbSection, m_datasetPath, m_sectionIndex, m_direction);

	long numMaps = m_isoPath.size();
	//	std::unique_ptr<murat::io::InputOutputCube<unsigned char>> video;
	//	video.reset(murat::io::openOrCreateCube<unsigned char>(outputPath.toStdString(), 3*numSamples, sectionAxeSize, numMaps));

	FILE* fp = fopen(m_outputPath.toStdString().c_str(), "w");
	if (fp!=nullptr) {
		for (long i=0; i<numMaps; i++) {
			std::vector<int> isoTab = getRgb2IsoTab2(QString::fromStdString(m_isoPath[i]));

			std::vector<unsigned char> restoreCache;
			restoreCache.resize(isoTab.size()*penSize);

			// prepare buffer
			for (long j=0; j<sectionAxeSize; j++) {
				long index = isoTab[j];
				if (index<0) {
					index = 0;
				} else if (index>numSamples-1) {
					index = numSamples - 1;
				}
				for (int penOffset=0; penOffset<penSize; penOffset++) {

					long indexOffset = index + penOffset - penSize/2;
					if (indexOffset>=0 && indexOffset<numSamples) {
						unsigned char oldVal = rgbSection[(j+indexOffset*sectionAxeSize)*3];
						restoreCache[j*penSize+penOffset] = oldVal;
						rgbSection[(j+indexOffset*sectionAxeSize)*3] = curveColor.red();
						rgbSection[(j+indexOffset*sectionAxeSize)*3+1] = curveColor.green();
						rgbSection[(j+indexOffset*sectionAxeSize)*3+2] = curveColor.blue();
					}
				}
			}

			// write
			fwrite(rgbSection.data(), sizeof(unsigned char), rgbSection.size(), fp);

			// restore
			for (long j=0; j<sectionAxeSize; j++) {
				long index = isoTab[j];
				if (index<0) {
					index = 0;
				} else if (index>numSamples-1) {
					index = numSamples - 1;
				}
				for (int penOffset=0; penOffset<penSize; penOffset++) {
					long indexOffset = index + penOffset - penSize/2;
					if (indexOffset>=0 && indexOffset<numSamples) {
						rgbSection[(j+indexOffset*sectionAxeSize)*3] = restoreCache[j*penSize+penOffset];
						rgbSection[(j+indexOffset*sectionAxeSize)*3+1] = restoreCache[j*penSize+penOffset];
						rgbSection[(j+indexOffset*sectionAxeSize)*3+2] = restoreCache[j*penSize+penOffset];
					}
				}
			}
		}
		fclose(fp);
	}


	return valid;
}


long SectionToVideo::getRgb2NumMaps() {
	if (m_rgb2Path.isNull() || m_rgb2Path.isEmpty() || !m_sizeDefined) {
		return 0;
	}

	FILE* fp = fopen(m_rgb2Path.toStdString().c_str(), "r");
	fseek(fp, 0L, SEEK_END);
	long sz = ftell(fp);
	fclose(fp);

	long size = sz / (m_numInline * m_numXline * sizeof(short) * 4);
	return size;
}

std::vector<int> SectionToVideo::getRgb2IsoTab(int mapIndex) {
	if (m_rgb2Path.isNull() || m_rgb2Path.isEmpty() || !m_sizeDefined) {
		return std::vector<int>();
	}
	long sectionAxeSize;
	long firstPosition = (mapIndex * m_numInline * m_numXline * 4 + 3) * sizeof(short);
	long positionStep;
	if (m_direction==SliceDirection::Inline) {
		sectionAxeSize = m_numXline;
		firstPosition += (m_sectionIndex * 4 * m_numXline) * sizeof(short);
		positionStep = 3 * sizeof(short);
	} else {
		sectionAxeSize = m_numInline;
		firstPosition += (m_sectionIndex * 4) * sizeof(short);
		positionStep = ((m_numXline * 4) - 1) * sizeof(short);
	}

	std::vector<int> tab;
	tab.resize(sectionAxeSize, 0);

	FILE* fp = fopen(m_rgb2Path.toStdString().c_str(), "r");

	fseek(fp, firstPosition, SEEK_SET);
	for (int i=0; i<sectionAxeSize; i++) {
		if (i>0) {
			fseek(fp, positionStep, SEEK_CUR);
		}
		short val;
		fread(&val, sizeof(short), 1, fp);

		tab[i] = std::round((val - m_sampleOrigin) / m_sampleStep);
	}

	fclose(fp);

	return tab;
}

std::vector<int> SectionToVideo::getRgb2IsoTab2(QString isoPath) {

	// todo
	/*
	if (m_rgb2Path.isNull() || m_rgb2Path.isEmpty() || !m_sizeDefined) {
		return std::vector<int>();
	}
	 */

	float *tmp = (float*)calloc(m_numInline * m_numXline, sizeof(float));
	if ( tmp == nullptr ) return std::vector<int>();
	QString filename = isoPath + "/" + ISODATA_NAME + ".iso";
	FreeHorizonManager::read(filename.toStdString(), tmp);
	/*
	// qDebug() << filename;
	FILE *pf = fopen((char*)filename.toStdString().c_str(), "r");
	if ( pf == nullptr ) { free(tmp); return std::vector<int>(); }
	fread(tmp, sizeof(float), m_numInline * m_numXline, pf);
	 */
	std::vector<int> tab;
	if (m_direction==SliceDirection::Inline) {
		tab.resize(m_numXline, 0);
		for (int y=0; y<m_numXline; y++)
			tab[y] = std::round((tmp[m_numXline*m_sectionIndex+y]-m_sampleOrigin) / m_sampleStep);
	}
	else {
		tab.resize(m_numInline, 0);
		for (int z=0; z<m_numXline; z++)
			tab[z] = std::round((tmp[m_numXline*z+m_sectionIndex]-m_sampleOrigin) / m_sampleStep);
	}
	// fclose(pf);
	free(tmp);
	return tab;
}

template<>
signed char SectionToVideo::getSymetricMax<signed char>(signed char min, signed char max) {
	return getSymetricMaxSigned<signed char>(min, max);
}

template<>
unsigned char SectionToVideo::getSymetricMax<unsigned char>(unsigned char min, unsigned char max) {
	return getSymetricMaxUnsigned<unsigned char>(min, max);
}

template<>
short SectionToVideo::getSymetricMax<short>(short min, short max) {
	return getSymetricMaxSigned<short>(min, max);
}

template<>
unsigned short SectionToVideo::getSymetricMax<unsigned short>(unsigned short min, unsigned short max) {
	return getSymetricMaxUnsigned<unsigned short>(min, max);
}

template<>
int SectionToVideo::getSymetricMax<int>(int min, int max) {
	return getSymetricMaxSigned<int>(min, max);
}

template<>
unsigned int SectionToVideo::getSymetricMax<unsigned int>(unsigned int min, unsigned int max) {
	return getSymetricMaxUnsigned<unsigned int>(min, max);
}

template<>
long SectionToVideo::getSymetricMax<long>(long min, long max) {
	return getSymetricMaxSigned<long>(min, max);
}

template<>
unsigned long SectionToVideo::getSymetricMax<unsigned long>(unsigned long min, unsigned long max) {
	return getSymetricMaxUnsigned<unsigned long>(min, max);
}

template<>
float SectionToVideo::getSymetricMax<float>(float min, float max) {
	return getSymetricMaxFloat<float>(min, max);
}

template<>
double SectionToVideo::getSymetricMax<double>(double min, double max) {
	return getSymetricMaxFloat<double>(min, max);
}

bool SectionToVideo::extractRange(const std::string& xtFile, double& resMin, double& resMax) {
	// get dynamic
	bool minSet = false;
	bool maxSet = false;
	float min=0, max = 1;

	FILE *pFile = fopen(xtFile.c_str(), "r");
	if ( pFile == NULL ) return false;
	char str[10000];
	fseek(pFile, 0x4c, SEEK_SET);
	int n = 0, cont = 1;
	while ( cont )
	{
		int nbre = fscanf(pFile, "VMIN=\t%f\n", &min);
		if ( nbre > 0 ) {
			cont = 0;
			minSet = true;
		} else
			fgets(str, 10000, pFile);
		n++;
		if ( n > 40 )
		{
			cont = 0;
			strcpy(str, "Other");
		}
	}
	fseek(pFile, 0x4c, SEEK_SET);
	n = 0, cont = 1;
	while ( cont )
	{
		int nbre = fscanf(pFile, "VMAX=\t%f\n", &max);
		if ( nbre > 0 ) {
			cont = 0;
			maxSet = true;
		} else
			fgets(str, 10000, pFile);
		n++;
		if ( n > 40 )
		{
			cont = 0;
			strcpy(str, "Other");
		}
	}
	fclose(pFile);

	if (minSet && maxSet) {
		resMin = min;
		resMax = max;
	}
	return minSet && maxSet;
}
