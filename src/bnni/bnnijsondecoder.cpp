#include "bnnijsondecoder.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/writer.h>

#include <QStringList>

#include <fstream>

typedef rapidjson::GenericDocument<rapidjson::ASCII<> > WDocument;
typedef rapidjson::GenericValue<rapidjson::ASCII<> > WValue;

std::vector<QString> BnniJsonDecoder::jsonExtractSeismics(const QString& jsonPath, const QString& projectPath) {
	std::vector<QString> seismicsOriStr, seismicsPath;

	std::ifstream ifs(jsonPath.toStdString().c_str());
	rapidjson::IStreamWrapper isw(ifs);
	WDocument document;
	document.ParseStream(isw);

	bool valid = !document.HasParseError() && document.IsObject() &&
			document.HasMember("seismicParameters") &&
			document["seismicParameters"].IsObject();

	if (valid) {
		WValue& parseValue = document["seismicParameters"];
		valid = parseValue.IsObject() && parseValue.HasMember("datasets") &&
				parseValue["datasets"].IsArray();

		if (valid) {
			WValue& array = parseValue["datasets"];
			for (unsigned int i=0; i<array.Size();i++) {
				WValue& e = array[i];
				bool datasetValid = e.IsObject() && e.HasMember("dataset") && e["dataset"].IsString();
				if (datasetValid) {
					seismicsOriStr.push_back(e["dataset"].GetString());
				}
			}
		}
	}
	if (valid) {
		for (const QString& oriStr : seismicsOriStr) {
			QString name = oriStr.split("/").last();
			QString survey_name = oriStr.split("\t").last().split("/").first();
			seismicsPath.push_back(projectPath+"/DATA/3D/"+survey_name+"/DATA/SEISMIC/seismic3d."+name);
		}
	}
	return seismicsPath;
}
