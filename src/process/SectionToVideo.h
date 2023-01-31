#ifndef PROCESS_SECTIONTOVIDEO_H
#define PROCESS_SECTIONTOVIDEO_H

#include "sliceutils.h"

#include <QString>
#include <QColor>

#include <cmath>

class SectionToVideo {
public:
	SectionToVideo();
	~SectionToVideo();

	// need to be compatible with rgb2, should be an xt file with defined min/max else limits will be used
	bool setDatasetPath(const QString& inDatasetPath);
	// need to be compatible with dataset and rgb2 and one of them should already be defined
	bool setSection(int sectionIndex, SliceDirection dir);
	// need to be compatible with dataset
	bool setRgb2(const QString& inRgb2Path);
	bool setRgb2(const QString& seismicPath, const QString& inRgb2Path);
	void setIsoPath(const std::vector<std::string> &isoPath);
	void setSeismicName(const QString& name);



	// video is in raw format like rg1
	void setOutputPath(const QString& outputVideoPath);

	static QString getDatasetPathFromRgb2(const QString& rgb2Path);
	static bool compatible(const QString& seismicPath, const QString& rgb2Path);
	static bool run(const QString& inDatasetPath, int sectionIndex, SliceDirection dir, const QString& inRgb2Path, const QString& outVideoPath);

	// raw to avi with directories by iso
	static bool run2(const std::vector<std::string>& isoPath, const QString& inDatasetPath,
			const QString& seismicName,
			int sectionIndex, SliceDirection dir, const QString& inRgb2Path, const QString& outVideoPath);

	/*
	 * Pen size should be greater or equal to 1
	 */
	bool computeVideo(const QColor& curveColor, int penSize);
	bool computeVideo2(const QColor& curveColor, int penSize);

	// max = max(abs(maxi), abs(mini))
	// return max compatible with max of ValueType and -max with min of ValueType
	template<typename ValueType>
	static ValueType getSymetricMax(ValueType min, ValueType max);

	template<typename ValueType>
	static ValueType getSymetricMaxFloat(ValueType min, ValueType max);
	template<typename ValueType>
	static ValueType getSymetricMaxSigned(ValueType min, ValueType max);
	template<typename ValueType>
	static ValueType getSymetricMaxUnsigned(ValueType min, ValueType max);

	static bool extractRange(const std::string& xtFile, double& resMin, double& resMax);

private:
	long getRgb2NumMaps();

	// return iso in index, function does the time to index conversion
	std::vector<int> getRgb2IsoTab(int mapIndex);
	std::vector<int> getRgb2IsoTab2(QString isoPath);

	QString m_datasetPath;
	int m_sectionIndex;
	SliceDirection m_direction;

	QString m_rgb2Path;

	QString m_outputPath;

	// define by rgb2 and dataset
	long m_numInline;
	long m_numXline;
	bool m_sizeDefined;

	float m_sampleStep;
	float m_sampleOrigin;
	std::vector<std::string> m_isoPath;
	QString m_seismicName = "";
};

template<typename ValueType>
ValueType SectionToVideo::getSymetricMaxFloat(ValueType min, ValueType max) {
	ValueType symMax = std::max(std::fabs(min), std::fabs(max));
	return symMax;
}

template<typename ValueType>
ValueType SectionToVideo::getSymetricMaxSigned(ValueType min, ValueType max) {
	// see note in https://en.cppreference.com/w/cpp/numeric/math/abs
	if (min==std::numeric_limits<ValueType>::min()) {
		min = -std::numeric_limits<ValueType>::max();
	}
	ValueType symMax = std::max(std::abs(min), std::abs(max));
	return symMax;
}

template<typename ValueType>
ValueType SectionToVideo::getSymetricMaxUnsigned(ValueType min, ValueType max) {
	ValueType symMax = max;
	return symMax;
}

#endif
