#ifndef SismageDBManager_H
#define SismageDBManager_H

#include <string>

class SismageDBManager {
public:
	~SismageDBManager();

	static std::string survey3DPathFromDatasetPath(const std::string& datasetPath);
	static std::string topoExchangePathFromSurveyPath(const std::string& survey3dDirStr);

	static std::string datasetPath2LayerPath(const std::string& datasetPath);
	static std::string datasetPath2CulturalPath(const std::string& datasetPath);
	static std::string surveyPath2CulturalPath(const std::string& surveyPath);
	static std::string surveyPath2HorizonsPath(const std::string& surveyPath);
	static std::string getTopo3dDescfromLayeringPath(const std::string& layeringPath);
	static std::string getTopo3dDescfromSurveyPath(const std::string& surveyPath);

	static std::string projectPath2CulturalPath(const std::string& projectPath);
	static std::string projectPath2NeuronPath(const std::string& projectPath);
	static std::string surveyPath2DatasetPath(const std::string& surveyPath);
	static std::string surveyPath2LayerPath(const std::string& surveyPath);

	static std::string rgt2rgbPath2SurveyPath(const std::string& rgt2rgbPath);

	static std::string getCulturalRegex();
	static std::string fixCulturalName(const std::string& culturalName);

	static std::string projectPathFromSurveyPath(const std::string& surveyPath);
	static std::string dirProjectPathFromProjectPath(const std::string& projectPath);

	static std::string datasetNameFromDatasetPath(const std::string& surveyPath);

	/**
	 * dataset name is not the sismage name, it is the filename without "seismic3d." (optional) and ".xt"
	 * If the file without "seismic3d." does not exists, return a path with "seismic3d."
	 * Try surveyPath/DATA/SEISMIC/datasetName.xt then surveyPath/DATA/SEISMIC/seismic3d.datasetName.xt
	 */
	static std::string datasetPathFromDatasetFileNameAndSurveyPath(const std::string& datasetName,
			const std::string& surveyPath);
	static std::string surveyNameFromSurveyPath(const std::string& surveyPath);

private:
	static std::string culturalRegex;
	static std::string notCulturalCharRegex;

	SismageDBManager();
};

#endif
