#ifndef SRC_SLICER_DATA_SISMAGE_NEXTVISIONDBMANAGER_H_
#define SRC_SLICER_DATA_SISMAGE_NEXTVISIONDBMANAGER_H_

#include <list>
#include <string>

// Like in SismageDBManager, try to avoid Qt classes and prefer std and boost possibilities
class NextVisionDBManager {
public:
	static std::string getNextVisionNameFromSismageHorizonName(const std::string& horizonSismageName,
			const std::string& sismageVersion, const std::string& datasetName);
	static std::string getSurvey3DPathFromHorizonPath(const std::string& horizonPath);
	static std::string getSeismicSismageNameFromHorizonPath(const std::string& horizonPath);
	static std::string getSurvey3DPathFromRgb2Path(const std::string& rgb2Path);
	static std::string getSeismicSismageNameFromRgb2Path(const std::string& rgb2Path);
	static std::string nvDatasetDir2RGTToRGBDir(const std::string& nvDatasetDir);
	static std::string horizonDir2SurveyPath(const std::string& horizonDir);
	static std::string surveyPath2HorizonDir(const std::string& surveyPath);
	static std::string surveyPath2NvDatasetsDir(const std::string& surveyPath);

	static std::list<std::string> searchNextVisionHorizonForSismageHorizon(const std::string& horizonSismageName,
			const std::string& sismageVersion, const std::string& horizonPath);
};

#endif
