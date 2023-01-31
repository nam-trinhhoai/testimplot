#include "bnniubjsondecoder.h"
#include "bnnimainwindow.h"

#include <QDebug>
#include <QObject>
#include <QDir>
#include <QFileInfo>

#include <nlohmann/json.hpp>

#include <fstream>

bool BnniUbjsonDecoder::load(QString filename, BnniMainWindow* mainWindow, float& trainSampleRate, QVector<Parameter>& seismics,
		QVector<LogParameter>& logs, QVector<Well>& wells, QString& survey_name) {
    std::ifstream ifs(filename.toStdString().c_str());
//    ifs.seekg(0, std::io_base::end);
//    long long fileSize = ifs.tellg();
//    ifs.seekg(0);
//    std::vector<std::uint8> ubjsonData;
//    ubjsonData.resize(fileSize);

    nlohmann::json document;
    try {
    	document = nlohmann::json::from_ubjson(ifs);
    }
    catch(nlohmann::json::parse_error e) {
    	return false;
    }

//    rapidjson::IStreamWrapper isw(ifs);
//    WDocument document;
//    document.ParseStream(isw);
//    qDebug() <<document.HasParseError();

    bool isValid = true;

    if (isValid && !document.is_object()) {
        qWarning() << QObject::tr("Unexpected format, could not get root object");
        isValid = false;
    }

    if (isValid && !(document.contains("seismicParameters") && document["seismicParameters"].is_object())) {
        qWarning() << QObject::tr("Unexpected format, could not get seismicParameters");
        isValid = false;
    } else if(isValid) {
    	nlohmann::json& parseValue = document["seismicParameters"];

        if (isValid && !(parseValue.is_object() && parseValue.contains("datasets"))) {
            qWarning() << QObject::tr("Unexpected format, could not find correct 'seismicParameters'");
            isValid = false;
        } else if (isValid) {
            nlohmann::json& array = parseValue["datasets"];

            if (isValid && !array.is_array()) {
                qWarning() << QObject::tr("Unexpected format, could not find 'datasets' in 'seismicParameters'");
                isValid = false;
            } else if(isValid){
                // fill seismic parameters
                for (unsigned int i=0; i<array.size();i++) {
                    nlohmann::json& _e = array[i];
                    if (isValid && !(_e.is_object() && _e.contains("dataset") && _e["dataset"].is_string())) {
                        qWarning() << QObject::tr("Unexpected format, could not find correct 'dataset' in 'seismicParameters'");
                        isValid = false;
                        break;
                    }

                    Parameter param;
                    param.name = QString::fromStdString(_e["dataset"].get<std::string>());
                    QString tmp_survey = param.name.split("\t").last().split("/").first();
                    if (i==0) {
                        survey_name = tmp_survey;
                        qDebug() << "Found survey : " << survey_name;
                    }

                    if (isValid && !(_e.contains("dynamic") && _e["dynamic"].is_array() &&_e["dynamic"].size()==2 &&
                                     _e["dynamic"][0].is_number_float() && _e["dynamic"][1].is_number_float())) {
                        qWarning() << QObject::tr("Unexpected format, could not find correct 'dynamic' in 'datasets' array from 'seismicParameters'");
                        isValid = false;
                        break;
                    }
                    param.min = _e["dynamic"][0].get<double>();
                    param.max = _e["dynamic"][1].get<double>();
                    param.InputMin = param.min;
                    param.InputMax = param.max;
                    param.OutputMin = -0.5;
                    param.OutputMax = 0.5;
                    if (_e.contains("samplingRate") && _e["samplingRate"].is_number_float()) {
                        trainSampleRate = _e["samplingRate"].get<double>();
                    } else {
                        trainSampleRate = BnniMainWindow::DEFAULT_SAMPLE_RATE;
                    }

                    seismics.append(param);

                    nlohmann::json& halfWindow = parseValue["halfWindowHeight"];
                    if (isValid && !halfWindow.is_number_integer()) {
                        qWarning() << QObject::tr("Unexpected format, could not find 'halfWindowHeight' in 'seismicParameters'");
                        isValid = false;

                    } else if(isValid){
                        mainWindow->setHalfWindow(halfWindow.get<int>());
                    }
                }
            }


        }


    }

    if (isValid && !(document.contains("logsParameters") && document["logsParameters"].is_object() &&
                     document["logsParameters"].contains("logColumns") && document["logsParameters"]["logColumns"].is_array() &&
                     document["logsParameters"].contains("logsDynamics") && document["logsParameters"]["logsDynamics"].is_array() &&
                     document["logsParameters"]["logColumns"].size() == document["logsParameters"]["logsDynamics"].size())) {
        qWarning() << QObject::tr("Unexpected format, could not find correct 'logsParameters'");
        isValid = false;
    } else if (isValid) {
        // fill log parameters
        for (unsigned int i=0; i<document["logsParameters"]["logColumns"].size(); i++) {
            LogParameter param;

            nlohmann::json& column = document["logsParameters"]["logColumns"][i];
            nlohmann::json& dynamic = document["logsParameters"]["logsDynamics"][i];

            if (isValid && !(column.is_object() && (column.contains("kind") && column["kind"].is_string() ||
                                                   column.contains("name") && column["name"].is_string()) &&
                             dynamic.is_array()) && dynamic.size()==2 && dynamic[0].is_number_float() && dynamic[1].is_number_float()) {
                qWarning() << QObject::tr("Unexpected format, could not find correct 'logsParameters'");
                break;
            }

            if (column.contains("kind")) {
                param.name = QString::fromStdString(column["kind"].get<std::string>());
            } else {
                param.name = QString::fromStdString(column["name"].get<std::string>());
            }
            param.min = dynamic[0].get<double>();
            param.max = dynamic[1].get<double>();

            param.InputMin = param.min;
            param.InputMax = param.max;
            param.OutputMin = 0.25;
            param.OutputMax = 0.75;
            param.preprocessing = LogNone;

            logs.append(param);
        }
    }

    if (isValid && !(document.contains("logsParameters") && document["logsParameters"].is_object() &&
                     document["logsParameters"].contains("wellbores") && document["logsParameters"]["wellbores"].is_array())) {
        qWarning() << QObject::tr("Unexpected format, could not find correct 'wellbores'");
        isValid = false;
    } else if(isValid) {
        for (unsigned int i=0; i<document["logsParameters"]["wellbores"].size(); i++) {
            nlohmann::json& w = document["logsParameters"]["wellbores"][i];
            if (isValid && !(w.is_string())) {
                qWarning() << QObject::tr("Unexpected format, could not find correct 'wellbores'");
                break;
            }

            Well well;
            well.name = QString::fromStdString(w.get<std::string>());
            wells.append(well);
        }
    }


    if (isValid && document.contains("samples") && document["samples"].is_object()) {
         for (int i=0; i<wells.size(); i++) {
             char name[1000];
             strcpy(name, wells[i].name.replace("\t","").toUtf8().toStdString().c_str());
             qDebug() << name;
             qDebug() << wells[i].name.replace("\t","");
             qDebug() << wells[i].name.replace("\t","").toUtf8().toStdString().c_str();

             if (!(document["samples"].contains(name) &&
                   document["samples"][name].is_array())) {

                 qWarning() << QObject::tr("Unexpected format, could not find correct expected well in 'samples'") << name;
                 //isValid = false;
                 continue;
             }

             if (isValid) {
                 for (int k=0; k<document["samples"][name].size(); k++) {
                     nlohmann::json& sample = document["samples"][name][k];
                     if (!(sample.is_array() && sample.size()==4 && sample[0].is_array() && sample[1].is_array())) {
                         qWarning() << QObject::tr("Unexpected format, could not find correct 'samples'");
                         isValid = false;
                         break;
                     }
                     LogSample example;
                     for (int j=0; j<sample[0].size(); j++) {
                         nlohmann::json& val = sample[0][j];
                         if (!(val.is_number_float())) {
                             qWarning() << QObject::tr("Unexpected format, could not find correct 'samples'");
                             isValid = false;
                             break;
                         }
                         example.seismicVals.append(val.get<double>());
                     }
                     if (isValid) {
                         for (int j=0; j<sample[1].size(); j++) {
                             nlohmann::json& val = sample[1][j];
                             if (!(val.is_number_float())) {
                                 qWarning() << QObject::tr("Unexpected format, could not find correct 'samples'");
                                 isValid = false;
                                 break;
                             }
                             example.logVals.append(val.get<double>());
                         }
                     }
                     if (isValid) {
                         isValid = sample[2].is_number_float();
                         if (isValid) {
                             example.depth = sample[2].get<double>();
                         }
                     }
                     if (isValid) {
                         isValid = sample[3].is_array() && sample[3].size()==3 && sample[3][0].is_number_float() &&
                                 sample[3][1].is_number_float() && sample[3][2].is_number_float();
                         if (isValid) {
                             example.x = sample[3][0].get<double>();
                             example.y = sample[3][1].get<double>();
                             example.z = sample[3][2].get<double>();
                         }
                     }
                     if (!isValid) {
                         break;
                     } else {
                         wells[i].samples.append(example);
                     }

                 }
             }

             if (isValid) {
                 wells[i].ranges.resize(1);
                 wells[i].ranges[0].min = 0;
                 wells[i].ranges[0].max = wells[i].samples.length()-1;

                 // Check if well is vertical
                 bool isVertical = true;
                 int indexLoop = 1;
                 double xDir = 0; // well is vertical if it goes forward in the same direction for all axises
                 double yDir = 0;

                 while (isVertical & indexLoop<wells[i].samples.count()) {
                	 isVertical = wells[i].samples[indexLoop].z - wells[i].samples[indexLoop-1].z > 0;

                	 if (isVertical && xDir==0) {
                		 xDir = wells[i].samples[indexLoop].x - wells[i].samples[indexLoop-1].x;
                	 }
                	 isVertical = isVertical && (wells[i].samples[indexLoop].x - wells[i].samples[indexLoop-1].x)*xDir >= 0;

                	 if (isVertical && yDir==0) {
                		 yDir = wells[i].samples[indexLoop].y - wells[i].samples[indexLoop-1].y;
                	 }
                	 isVertical = isVertical && (wells[i].samples[indexLoop].y - wells[i].samples[indexLoop-1].y)*yDir >= 0;

                	 indexLoop ++;
                 }
                 wells[i].isVertical = isVertical;
             }
         }
    }

    return isValid;
}

std::vector<QString> BnniUbjsonDecoder::ubjsonExtractSeismics(const QString& ubjsonPath, const QString& projectPath) {
	std::vector<QString> seismicsOriStr, seismicsPath;

	std::ifstream ifs(ubjsonPath.toStdString().c_str());
	nlohmann::json document;
	try {
		document = nlohmann::json::from_ubjson(ifs);
	}
	catch(nlohmann::json::parse_error e) {
		return seismicsPath;
	}

	bool valid = document.is_object() &&
			document.contains("seismicParameters") &&
			document["seismicParameters"].is_object();

	if (valid) {
		nlohmann::json& parseValue = document["seismicParameters"];
		valid = parseValue.is_object() && parseValue.contains("datasets") &&
				parseValue["datasets"].is_array();

		if (valid) {
			nlohmann::json& array = parseValue["datasets"];
			for (unsigned int i=0; i<array.size();i++) {
				nlohmann::json& e = array[i];
				bool datasetValid = e.is_object() && e.contains("dataset") && e["dataset"].is_string();
				if (datasetValid) {
					seismicsOriStr.push_back(QString::fromStdString(e["dataset"].get<std::string>()));
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
