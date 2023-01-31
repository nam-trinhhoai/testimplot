#include "wellpick.h"
#include "wellbore.h"
#include "marker.h"
#include "workingsetmanager.h"
#include "wellpickgraphicrepfactory.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "affinetransformation.h"
#include "affine2dtransformation.h"
#include "cubeseismicaddon.h"
#include "folderdata.h"
#include "sampletypebinder.h"

#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <QRegularExpression>

#include <cmath>

WellPick::WellPick(WellBore* parentWell, WorkingSetManager * workingSet,const QString &name, const QString& kind, double value,
		const QString& idPath, QObject *parent) : IData(workingSet, parent), m_wellBore(parentWell), m_markerName(name),
		m_kind(kind), m_value(value), IFileBasedData(idPath) {
	m_repFactory = new WellPickGraphicRepFactory(this);
	m_uuid = QUuid::createUuid();
}

WellPick::~WellPick() {}

//IData
IGraphicRepFactory *WellPick::graphicRepFactory() {
	return m_repFactory;
}

QUuid WellPick::dataID() const {
	return m_uuid;
}

QString WellPick::name() const {
	return m_wellBore->name();
}

QString WellPick::markerName() const {
	return m_markerName;
}

double WellPick::value() const {
	return m_value;
}

QString WellPick::kind() const {
	return m_kind;
}

WellUnit WellPick::kindUnit() const {
	WellUnit unit = WellUnit::UNDEFINED_UNIT;
	if (m_kind.compare("MD")==0) {
		unit = WellUnit::MD;
	} else if (m_kind.compare("TVD")==0) {
		unit = WellUnit::TVD;
	} else if (m_kind.compare("TWT")==0) {
		unit = WellUnit::TWT;
	}
	return unit;
}

WellBore* WellPick::wellBore() const {
	return m_wellBore;
}

Marker* WellPick::currentMarker() const {
	return m_currentMarker;
}

void WellPick::setCurrentMarker(Marker* marker) {
	m_currentMarker = marker;
}

// return nullptr if could not read file
WellPick* WellPick::getWellPickFromDescFile(WellBore* parentWell, QColor color, QString pickFile, WorkingSetManager * workingSet, QObject *parent) {
	bool isValid = false;
	QString name, kind;
	double value;
	//QColor color(0, 0, 0); // default color

	bool nameFound = false;
	bool kindFound = false;
	bool valueFound = false;

	QFile file(pickFile);
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		qDebug() << "WellPick : cannot read pick file in text format " << pickFile;
		return nullptr;
	}

	QTextStream in(&file);
	while (!in.atEnd()) {
		QString line = in.readLine();
		QStringList lineSplit = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
		if(lineSplit.size()>1) {
			if (lineSplit.first().compare("Name")==0) {
				name = lineSplit[1];
				nameFound = true;
			} else if (lineSplit.first().compare("Kind")==0) {
				kind = lineSplit[1];
				kindFound = true;
			} else if (lineSplit.first().compare("Value")==0) {
				value = lineSplit[1].toDouble(&valueFound);
			}/* else if (lineSplit.first().compare("Color")==0 && lineSplit.size()==4) {
				bool okR, okG, okB;
				int r = lineSplit[1].toInt(&okR);
				int g = lineSplit[2].toInt(&okG);
				int b = lineSplit[3].toInt(&okB);
				if (okR && okG && okB) {
					color = QColor(r, g, b);
				}
			}*/
		}
	}

	WellPick* out = nullptr;
	if (nameFound && kindFound && valueFound) {
		out = new WellPick(parentWell, workingSet, name, kind, value, pickFile, parent);

		// search marker to use
		QList<IData*> datas = workingSet->folders().markers->data();
		std::size_t idxMarker=0;
		while (idxMarker<datas.count() && (dynamic_cast<Marker*>(datas[idxMarker])==nullptr || datas[idxMarker]->name().compare(name)!=0)) {
			idxMarker++;
		}
		Marker* marker = nullptr;
		if (idxMarker>=datas.count()) {
			marker = new Marker(workingSet, name, parent);
			marker->setColor(color);
			workingSet->addMarker(marker);
		} else {
			marker = dynamic_cast<Marker*>(datas[idxMarker]);
		}

		out->setCurrentMarker(marker);
		marker->addWellPick(out);
	}
	return out;
}

std::pair<RgtSeed, bool> WellPick::getProjectionOnDataset(Seismic3DAbstractDataset* dataset, SampleUnit sampleUnit, WellBore* wellBore, double unitVal, WellUnit wellUnit) {
	RgtSeed out;
	bool isValid = false;

	double depth, x, y;
	x = wellBore->getXFromWellUnit(unitVal, wellUnit, &isValid);
	if (isValid) {
		y = wellBore->getYFromWellUnit(unitVal, wellUnit, &isValid);
	}
	if (isValid) {
		depth = wellBore->getDepthFromWellUnit(unitVal, wellUnit, sampleUnit, &isValid);
	}

	if (isValid) {
		double val, imageX, imageY;
		dataset->sampleTransformation()->indirect(depth, val);
		out.x = std::round(val);
		dataset->ijToXYTransfo()->worldToImage(x, y, imageX, imageY);
		out.y = std::round(imageX);
		out.z = std::round(imageY);

		isValid = out.x>=0 && out.x<dataset->height() &&
				out.y>=0 && out.y<dataset->width() &&
				out.z>=0 && out.z<dataset->depth();
	}

	return std::pair<RgtSeed, bool>(out, isValid);
}

template<typename InputType>
struct GetProjectionOnDatasetKernel {
	static double run(Seismic3DDataset* dataset, int channel, long x1, long x2, long y, long z) {
		std::vector<InputType> tab;
		tab.resize(dataset->dimV());
		dataset->readSubTraceAndSwap(tab.data(), x1, x2, y, z);
		return tab[channel];
	}
};

std::pair<RgtSeed, bool> WellPick::getProjectionOnDataset(Seismic3DAbstractDataset* dataset, int channel, SampleUnit sampleUnit) {
	RgtSeed out;
	CubeSeismicAddon addon = dataset->cubeSeismicAddon();
	if (sampleUnit!=addon.getSampleUnit() || channel<0 || channel>=dataset->dimV()) {
		return std::pair<RgtSeed, bool>(out, false);
	}

	bool isValid = false;

	WellUnit wellUnit = kindUnit();
	std::pair<RgtSeed, bool> pair = getProjectionOnDataset(dataset, sampleUnit, m_wellBore, m_value, wellUnit);
	isValid = pair.second;
	out = pair.first;

	if (isValid) {
		if (Seismic3DDataset* cpuDataset = dynamic_cast<Seismic3DDataset*>(dataset)) { // TODO code function for GPU
			SampleTypeBinder binder(cpuDataset->sampleType());
			out.rgtValue = binder.bind<GetProjectionOnDatasetKernel>(cpuDataset, channel, out.x, out.x+1, out.y, out.z);
		} else {
			isValid = false;
		}
	}
	return std::pair<RgtSeed, bool>(out, isValid);
}

void WellPick::removeGraphicsRep(){
	emit deletedMenu();
}
