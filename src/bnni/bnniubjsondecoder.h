#ifndef SRC_BNNI_BNNIUBJSONDECODER_H
#define SRC_BNNI_BNNIUBJSONDECODER_H

#include "structures.h"

#include <QString>

#include <vector>

class BnniMainWindow;

class BnniUbjsonDecoder {
public:
	static std::vector<QString> ubjsonExtractSeismics(const QString& jsonPath, const QString& projectPath);

	static bool load(QString filename, BnniMainWindow* mainWindow, float& trainSampleRate, QVector<Parameter>& seismics,
			QVector<LogParameter>& logs, QVector<Well>& wells, QString& survey_name);
};

#endif
