
#ifndef __FREEHORIZONUTIL__
#define __FREEHORIZONUTIL__

/*
#define FREEHORIZON_JSONFILENAME "/desc.json"
static const QString freehorizondimsKey = QStringLiteral("dims");
static const QString freehorizontdebKey = QStringLiteral("tdeb");
static const QString freehorizontpasechKey = QStringLiteral("pasech");

QString freeHorizonWriteDesc(QString path, int dimy, int dimz);

std::pair<int, int> freeHorizonGetDims(QString path);
std::pair<float, float> freeHorizonGetTdebPasEch(QString path);
*/

#include <string>

std::string freeHorizonSaveWithoutTdebAndPasech(std::string filename, float *data, int dimy, int dimz, float tdeb, float pasech);

std::pair<int, int> freeHorizonGetDims(std::string filename);

std::string freeHorizonRead(std::string filename, void *data);


#endif
