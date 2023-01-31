#ifndef SRC_BNNI_BNNIJSONDECODER_H
#define SRC_BNNI_BNNIJSONDECODER_H

#include <QString>

#include <vector>

class BnniJsonDecoder {
public:
	static std::vector<QString> jsonExtractSeismics(const QString& jsonPath, const QString& projectPath);
};

#endif
