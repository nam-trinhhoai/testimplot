#include "wellhead.h"
#include "wellbore.h"
#include <cmath>
#include <iostream>
#include "gdal.h"
#include "wellbore.h"
#include "wellheadgraphicrepfactory.h"

#include <QTextStream>
#include <QFile>
#include <QDir>
#include <QDebug>

WellHead::WellHead(WorkingSetManager *workingSet, const QString &name,
		double X, double Y, double Z, const QString& idPath,QString date, QObject *parent) :
		IData(workingSet, parent), IFileBasedData(idPath) {
	m_name = name;
	m_x = X;
	m_y = Y;
	m_z = Z;
	m_uuid = QUuid::createUuid();
	m_descFile = idPath;
	m_date=date;
	m_repFactory = new WellHeadGraphicRepFactory(this);

}

IGraphicRepFactory* WellHead::graphicRepFactory() {
	return m_repFactory;
}

QUuid WellHead::dataID() const {
	return m_uuid;
}

WellHead::~WellHead() {

}

double WellHead::x() const {
	return m_x;
}

double WellHead::y() const {
	return m_y;
}

double WellHead::z() const {
	return m_z;
}

void WellHead::addWellBore(WellBore *wellBore) {
	m_wellBores.push_back(wellBore);
	emit wellBoreAdded(wellBore);
}

void WellHead::removeWellBore(WellBore *wellBore) {
	m_wellBores.removeOne(wellBore);
	emit wellBoreRemoved(wellBore);
}

QList<WellBore*> WellHead::wellBores() {
	return m_wellBores;
}



WellHead* WellHead::getWellHeadFromDescFile(QString descFile, WorkingSetManager * workingSet, QObject *parent) {
	QString wellHeadName;
	double X, Y, Z;
	QString date;
	int numberOfOptionsSet = 0;
	QFile file(descFile);
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		qDebug() << "WellHead : cannot read desc file in text format " << descFile;
		return nullptr;
	}

	QTextStream in(&file);
	bool nameFound = false;
	while (!in.atEnd() && numberOfOptionsSet<4) {
		QString line = in.readLine();
		QStringList lineSplit = line.split("\t");
		if(lineSplit.size()>1) {
			if (lineSplit.first().compare("Name")==0) {
				wellHeadName = lineSplit[1];
				numberOfOptionsSet++;
			} else if (lineSplit.first().compare("X")==0) {
				bool ok;
				X = lineSplit[1].toDouble(&ok);
				if (ok) {
					numberOfOptionsSet++;
				}
			} else if (lineSplit.first().compare("Y")==0) {
				bool ok;
				Y = lineSplit[1].toDouble(&ok);
				if (ok) {
					numberOfOptionsSet++;
				}
			} else if (lineSplit.first().compare("Z")==0) {
				bool ok;
				Z = lineSplit[1].toDouble(&ok);
				if (ok) {
					numberOfOptionsSet++;
				}
			}
			else if (lineSplit.first().compare("Date")==0) {

				date = lineSplit[1];
			}
		}
	}

	if (numberOfOptionsSet==4) {
		return new WellHead(workingSet, wellHeadName, X, Y, Z, descFile,date, parent);
	} else if (numberOfOptionsSet==3 && (wellHeadName.isNull()  || wellHeadName.isEmpty())) {
		QDir dir = QFileInfo(descFile).absoluteDir();
		wellHeadName = dir.dirName();
		return new WellHead(workingSet, wellHeadName, X, Y, Z, descFile,date, parent);
	} else {
		qDebug() << "WellHead : unsupported desc file " << descFile;
		return nullptr;
	}
}

QString WellHead::getDirName() {
	QFileInfo info(m_descFile);
	QDir dir = info.dir();
	return dir.dirName();
}

//double WellHead::displayDistance() const {
//	return m_displayDistance;
//}
//
//void WellHead::setDisplayDistance(double val) {
//	if (m_displayDistance!=val) {
//		m_displayDistance = val;
//		displayDistanceChanged(m_displayDistance);
//	}
//}
