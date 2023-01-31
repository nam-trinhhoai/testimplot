#ifndef SRC_BNNI_CONFIGREADER_H
#define SRC_BNNI_CONFIGREADER_H

#include "structures.h"

// For now only use as utils for accessing some config parameters
// Could later be used to retrieve parameters from the config file
class ConfigReader {
public:
    static NeuralNetwork getNetworkFromFile(const QString& configFile, bool* ok);
    static QString separator;
};

#endif // SRC_BNNI_CONFIGREADER_H
