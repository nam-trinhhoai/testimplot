#include "nextvisiondbmanager.h"
#include <freeHorizonManager.h>
#include <boost/filesystem.hpp>
#include <geotimepath.h>


namespace fs = boost::filesystem;

std::string NextVisionDBManager::getNextVisionNameFromSismageHorizonName(const std::string& horizonSismageName,
		const std::string& sismageVersion, const std::string& datasetName) {
	std::string nvName;
	if (sismageVersion.size()>0) {
		nvName = horizonSismageName+ "_" + sismageVersion + "_sismage";
		// nvName = horizonSismageName+ "_" + sismageVersion + "_sismage_(" + datasetName + ")";
	} else {
		nvName = horizonSismageName + "_sismage";
		// nvName = horizonSismageName + "_sismage_(" + datasetName + ")";
	}
	return nvName;
}

std::string NextVisionDBManager::getSurvey3DPathFromHorizonPath(const std::string& horizonPath) {
	fs::path fsHorizonPath(horizonPath);
	fs::path horizonDir = fsHorizonPath.parent_path();// HORIZON_GRIDS folder
	fs::path datasetDir = horizonDir.parent_path();
	fs::path ijkDir = datasetDir.parent_path(); // IJK
	fs::path importExportDir = ijkDir.parent_path(); // ImportExport
	fs::path surveyDir = importExportDir.parent_path(); // survey dir

	return surveyDir.c_str();
}

std::string NextVisionDBManager::getSeismicSismageNameFromHorizonPath(const std::string& horizonPath) {
	fs::path fsHorizonPath(horizonPath);
	fs::path horizonDir = fsHorizonPath.parent_path();// HORIZON_GRIDS folder
	fs::path datasetDir = horizonDir.parent_path();

	return datasetDir.filename().c_str();
}

std::string NextVisionDBManager::getSurvey3DPathFromRgb2Path(const std::string& rgb2Path) {
	fs::path fsRgb2Path(rgb2Path);
	fs::path rgb2Dir = fsRgb2Path.parent_path();// cubeRgt2Rgb folder
	fs::path datasetDir = rgb2Dir.parent_path();
	fs::path ijkDir = datasetDir.parent_path(); // IJK
	fs::path importExportDir = ijkDir.parent_path(); // ImportExport
	fs::path surveyDir = importExportDir.parent_path(); // survey dir

	return surveyDir.c_str();
}

std::string NextVisionDBManager::getSeismicSismageNameFromRgb2Path(const std::string& rgb2Path) {
	fs::path fsRgb2Path(rgb2Path);
	fs::path rgb2Dir = fsRgb2Path.parent_path();// cubeRgt2Rgb folder
	fs::path datasetDir = rgb2Dir.parent_path();

	return datasetDir.filename().c_str();
}

std::string NextVisionDBManager::nvDatasetDir2RGTToRGBDir(const std::string& nvDatasetDir) {
	fs::path nvDatasetDirPath(nvDatasetDir);

	nvDatasetDirPath /= "/cubeRgt2RGB/";

	return nvDatasetDirPath.c_str();
}

std::string NextVisionDBManager::horizonDir2SurveyPath(const std::string& horizonDir) {
	fs::path fsHorizonPath(horizonDir); // NV-HORIZON

	fs::path fsNvPath = fsHorizonPath.parent_path(); // NEXTVISION
	fs::path fsIimportExportPath = fsNvPath.parent_path(); // ImportExport
	fs::path fsSurveyPath = fsIimportExportPath.parent_path(); // survey

	return fsSurveyPath.c_str();
}

std::string NextVisionDBManager::surveyPath2HorizonDir(const std::string& surveyPath) {
	fs::path fsSurveyPath(surveyPath);

	// fsSurveyPath /= "/ImportExport/IJK/HORIZONS/" + FreeHorizonManager::BaseDirectory + "/";
	fsSurveyPath /= "/" + GeotimePath::NEXTVISION_NVHORIZON_PATH + "/";

	return fsSurveyPath.c_str();
}

std::string NextVisionDBManager::surveyPath2NvDatasetsDir(const std::string& surveyPath) {
	fs::path fsSurveyPath(surveyPath);

	fsSurveyPath /= "/ImportExport/IJK/";

	return fsSurveyPath.c_str();
}


std::list<std::string> NextVisionDBManager::searchNextVisionHorizonForSismageHorizon(const std::string& horizonSismageName,
		const std::string& sismageVersion, const std::string& horizonPath) {
	std::list<std::string> out;
	fs::path hp(horizonPath);
	if (horizonPath.empty() || !fs::is_directory(hp)) {
		return out;
	}

	std::string horizonBaseName;
	if (sismageVersion.size()>0) {
		horizonBaseName = horizonSismageName+ "_" + sismageVersion + "_sismage_(";
	} else {
		horizonBaseName = horizonSismageName + "_sismage_(";
	}

	for (const auto & entry : fs::directory_iterator(hp)) {
		fs::path dp(entry);
		std::string dirFilename = dp.filename().c_str();

		int index = dirFilename.find(horizonBaseName);
		bool foundHorizon = index==0;
		if (foundHorizon) {
			out.push_back(dp.c_str());
		}
	}
	return out;
}
