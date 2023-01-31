#include "configreader.h"

#include <QFile>
#include <QTextStream>
#include <QString>
#include <QStringList>

QString ConfigReader::separator = ",";

NeuralNetwork ConfigReader::getNetworkFromFile(const QString& configFile, bool* ok) {
    NeuralNetwork output = NeuralNetwork::Dense;
    *ok = false;

    QFile file(configFile);

    if (!configFile.isNull() && !configFile.isEmpty() && file.open(QIODevice::ReadOnly)) {
        QTextStream stream(&file);
        QString line;
        QString key;
        QStringList values;

        while (!(*ok) && stream.readLineInto(&line)) {
            values = line.split("\n")[0].split(separator);
            key = values[0];
            values.pop_front();

			// Ignore line if there is no
            if (values.size()==1 && key.compare("network")==0) {
                QString value = values[0];
                if (value.toLower().compare("dense")==0) {
                	output = NeuralNetwork::Dense;
                	*ok = true;
                } else if (value.toLower().compare("dnn")==0) {
                	output = NeuralNetwork::Dnn;
                	*ok = true;
                } else if (value.toLower().compare("xgboost")==0) {
                	output = NeuralNetwork::Xgboost;
                	*ok = true;
                }
            }
        }
    }
    file.close();

    return output;
}
