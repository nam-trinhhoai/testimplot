#include "sismagedbmanager.h"

#include "ProjectManagerNames.h"
#include "SeismicManager.h"

#include <boost/filesystem.hpp>
#include <iostream>

// for regular expressions only, can be replaced by std regex
#include <vector>
#include <QString>
#include <QRegularExpression>

namespace fs = boost::filesystem;

std::string SismageDBManager::culturalRegex = "^[a-zA-Z0-9_\\-+()]+$";
std::string SismageDBManager::notCulturalCharRegex = "[^a-zA-Z0-9_\\-+()]+";

SismageDBManager::SismageDBManager() {
}

SismageDBManager::~SismageDBManager() {
}

std::string SismageDBManager::survey3DPathFromDatasetPath(
		const std::string &datasetStr) {
	fs::path dp(datasetStr);
	const std::string folderPath = dp.parent_path().c_str();
	fs::path descPath = folderPath;
	fs::path dataDirPath = descPath.parent_path();
	fs::path survey3dPath = dataDirPath.parent_path();
	return survey3dPath.c_str();
}

std::string SismageDBManager::topoExchangePathFromSurveyPath(
		const std::string &survey3dDirStr) {
	fs::path topoFilePath(survey3dDirStr);
	topoFilePath /= "/ImportExport/IJK/TOPO3D.ijk";
	return topoFilePath.c_str();
}

std::string SismageDBManager::datasetPath2LayerPath(const std::string& datasetPath) {
	std::cout << "SismageDBManager::datasetPath2SurveyPath SEISMIC PATH= "<< datasetPath << std::endl;
	fs::path dp(datasetPath);
	const std::string folderPath = dp.parent_path().c_str();
	fs::path descPath = folderPath;
	std::cout << descPath.c_str() << std::endl;
	//Time to be simplify
	fs::path dataDirPath = descPath.parent_path();
	dataDirPath /= "/LAYERS/";
	std::cout << "LAYER PATH= "<< dataDirPath.c_str() << std::endl;
	return dataDirPath.c_str();
}

std::string SismageDBManager::getTopo3dDescfromLayeringPath(
                const std::string& layeringPath) {
        std::cout << "SismageDBManager::getTopo3dDescfromLayerPath LAYER PATH= "<< layeringPath << std::endl;
        fs::path dp(layeringPath);
        const std::string folderPath = dp.parent_path().c_str();
        fs::path descPath = folderPath;
        std::cout << descPath.c_str() << std::endl;
        //TODO Should be simplify
        descPath /= "/../TOPOS/topo3d.desc";
        std::cout << "topo3d.desc PATH= "<< descPath.c_str() << std::endl;
        return descPath.c_str();
}

std::string SismageDBManager::getTopo3dDescfromSurveyPath(
                const std::string& surveyPath) {
        std::cout << "SismageDBManager::getTopo3dDescfromSurveyPath= "<< surveyPath << std::endl;
        std::string topo3DPath = surveyPath;
        topo3DPath.append("/DATA/TOPOS/topo3d.desc");
        std::cout << "topo3d.desc PATH= "<< topo3DPath << std::endl;
        return topo3DPath;
}

std::string SismageDBManager::datasetPath2CulturalPath(const std::string& datasetPath) {
	std::cout << "SismageDBManager::datasetPath2CulturalPath SEISMIC PATH= "<< datasetPath << std::endl;
	fs::path dp(datasetPath);
	const std::string folderPath = dp.parent_path().c_str();
	fs::path descPath = folderPath;
	std::cout << descPath.c_str() << std::endl;
	//Time to be simplify
	fs::path dataDirPath = descPath.parent_path();
	fs::path survey3dDirPath = dataDirPath.parent_path();
	fs::path d3DirPath = survey3dDirPath.parent_path();
	fs::path projectDataDirPath = d3DirPath.parent_path();
	// attention au nom...
	projectDataDirPath /= "/CULTURAL/";
	std::cout << "CULTURAL PATH= "<< projectDataDirPath.c_str() << std::endl;
	return projectDataDirPath.c_str();
}

std::string SismageDBManager::surveyPath2CulturalPath(const std::string& surveyPath) {
	std::cout << "SismageDBManager::datasetPath2CulturalPath SURVEY PATH= "<< surveyPath << std::endl;
	fs::path sp(surveyPath);
	fs::path d3DirPath = sp.parent_path();
	fs::path projectDataDirPath = d3DirPath.parent_path();
	// attention au nom...
	projectDataDirPath /= "/CULTURAL/";
	std::cout << "CULTURAL PATH= "<< projectDataDirPath.c_str() << std::endl;
	return projectDataDirPath.c_str();
}

std::string SismageDBManager::surveyPath2HorizonsPath(const std::string& surveyPath) {
	std::cout << "SismageDBManager::datasetPath2CulturalPath SURVEY PATH= "<< surveyPath << std::endl;
	fs::path sp(surveyPath);
	sp /= "/DATA/HORIZONS/";

	std::cout << "HORIZONS PATH= "<< sp.c_str() << std::endl;
	return sp.c_str();
}

std::string SismageDBManager::projectPath2CulturalPath(const std::string& projectPath) {
	std::cout << "SismageDBManager::projectPath2CulturalPath PROJECT PATH= "<< projectPath << std::endl;
	fs::path sp(projectPath);
	sp /= "/DATA/CULTURAL/";

	std::cout << "CULTURAL PATH= "<< sp.c_str() << std::endl;
	return sp.c_str();
}

std::string SismageDBManager::surveyPath2DatasetPath(const std::string& surveyPath) {
	std::cout << "SismageDBManager::surveyPath2DatasetPath SURVEY PATH= "<< surveyPath << std::endl;
	fs::path sp(surveyPath);
	sp /= "/DATA/SEISMIC/";

	std::cout << "SEISMIC PATH= "<< sp.c_str() << std::endl;
	return sp.c_str();
}

std::string SismageDBManager::projectPath2NeuronPath(const std::string& projectPath) {
	fs::path sp(projectPath);

	sp /= "/DATA/NEURONS/neurons2/LogInversion2Problem3/";

	return sp.c_str();
}

std::string SismageDBManager::surveyPath2LayerPath(const std::string& surveyPath) {
	std::cout << "SismageDBManager::surveyPath2LayerPath SURVEY PATH= "<< surveyPath << std::endl;
	fs::path sp(surveyPath);
	sp /= "/DATA/LAYERS/";

	std::cout << "LAYER PATH= "<< sp.c_str() << std::endl;
	return sp.c_str();
}

std::string SismageDBManager::rgt2rgbPath2SurveyPath(const std::string& rgt2rgbPath) {
	std::cout << "SismageDBManager::rgt2rgbPath2SurveyPath RGT2RGB PATH= "<< rgt2rgbPath << std::endl;
	fs::path dp(rgt2rgbPath);
	const std::string folderPath = dp.parent_path().c_str(); // get rgt2rgb dir

	std::string nextVisionTag ="ImportExport/NEXTVISION";
	// old path
	if ( folderPath.find(nextVisionTag) == std::string::npos )
	{
		fs::path descPath = folderPath;
		fs::path dataDirPath = descPath.parent_path(); // dataset dir
		fs::path ijkDirPath = dataDirPath.parent_path(); // ijk dir
		fs::path importExportDirPath = ijkDirPath.parent_path(); // import dir
		fs::path survey3dPath = importExportDirPath.parent_path(); // survey dir
		return survey3dPath.c_str();
	}
	else
	{
		fs::path descPath = folderPath;
		fs::path dataDirPath = descPath.parent_path(); // dataset dir
		fs::path importExportDirPath = dataDirPath.parent_path(); // import dir
		fs::path survey3dPath = importExportDirPath.parent_path(); // survey dir
		return survey3dPath.c_str();
	}
}

std::string SismageDBManager::getCulturalRegex() {
	return culturalRegex;
}

std::string SismageDBManager::fixCulturalName(const std::string& culturalName) {
	std::vector<QChar> chars;
	QString regExpStr = QString::fromStdString(notCulturalCharRegex);
	QString subject = QString::fromStdString(culturalName);
	QRegularExpression regExp(regExpStr);
	QRegularExpressionMatchIterator it = regExp.globalMatch(subject);
	long firstValid = 0;
	while (it.hasNext()) {
		QRegularExpressionMatch match = it.next();
		long lastValid = match.capturedStart()-1;
		for (long i=firstValid; i<=lastValid; i++) {
			chars.push_back(culturalName.at(i));
		}
		firstValid = match.capturedEnd();
	}
	if (firstValid < culturalName.size()) {
	long lastValid = culturalName.size()-1;
		for (long i=firstValid; i<=lastValid; i++) {
			chars.push_back(culturalName.at(i));
		}
	}
	QString out(chars.data(), chars.size());
	return out.toStdString();
}

std::string SismageDBManager::projectPathFromSurveyPath(const std::string& surveyPath) {
	std::cout << "SismageDBManager::projectFromSurveyPath SURVEY PATH= "<< surveyPath << std::endl;
	fs::path sp(surveyPath);
	sp = sp.parent_path().parent_path().parent_path();
	std::cout << "PROJECT PATH= "<< sp.c_str() << std::endl;
	return sp.c_str();
}

std::string SismageDBManager::dirProjectPathFromProjectPath(const std::string& projectPath) {
	std::cout << "SismageDBManager::dirProjectPathFromprojectPath PROJECT PATH= "<< projectPath << std::endl;
	fs::path sp(projectPath);
	sp = sp.parent_path();
	std::cout << "DIR PROJECT PATH= "<< sp.c_str() << std::endl;
	return sp.c_str();
}

std::string SismageDBManager::datasetNameFromDatasetPath(const std::string& datasetPath) {
	return SeismicManager::seismicFullFilenameToTinyName(QString::fromStdString(datasetPath)).toStdString();
}

std::string SismageDBManager::datasetPathFromDatasetFileNameAndSurveyPath(const std::string& datasetName,
		const std::string& surveyPath) {
	std::string datasetDirPath = surveyPath2DatasetPath(surveyPath);
	std::string datasetPath = datasetDirPath + "/" + datasetName + ".xt";
	fs::path datasetBoostPath(datasetPath);
	if (!fs::exists(datasetBoostPath)) {
		datasetPath = datasetDirPath + "/seismic3d." + datasetName + ".xt";
	}

	return datasetPath;
}

std::string SismageDBManager::surveyNameFromSurveyPath(const std::string& surveyPath) {
	fs::path sp(surveyPath);
	std::string surveyFilename = sp.filename().c_str();

	fs::path surveyDescFile = sp / ("/" + surveyFilename + ".desc");
	std::string surveyDescFilePath = surveyDescFile.c_str();

	std::string surveyName;
	QString foundName = ProjectManagerNames::getKeyFromFilename(QString::fromStdString(surveyDescFilePath), "name=");
	if (!foundName.isNull() && !foundName.isEmpty()) {
		surveyName = foundName.toStdString();
	} else {
		surveyName = surveyFilename;
	}

	return surveyName;
}
