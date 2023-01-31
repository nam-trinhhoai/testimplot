#include "ijkhorizon.h"
#include "ijkhorizongraphicrepfactory.h"
#include "seismic3dabstractdataset.h"
#include "Xt.h"

IJKHorizon::IJKHorizon(QString name, QString path, QString seismicOriginPath,
		WorkingSetManager *workingSet, QObject* parent) : IData(workingSet, parent),
		IFileBasedData(path) {
	m_name = name;
	m_path = path;
	m_seismicOriginPath = seismicOriginPath;
	m_uuid = QUuid::createUuid();
	m_repFactory.reset(new IJKHorizonGraphicRepFactory(this));
}

IJKHorizon::~IJKHorizon() {

}

//IData
IGraphicRepFactory* IJKHorizon::graphicRepFactory() {
	return m_repFactory.get();
}

QUuid IJKHorizon::dataID() const {
	return m_uuid;
}

QString IJKHorizon::name() const {
	return m_name;
}

QString IJKHorizon::path() const {
	return m_path;
}

QString IJKHorizon::seismicOriginPath() const {
	return m_seismicOriginPath;
}

bool IJKHorizon::filterHorizon(IJKHorizon* horizon, Seismic3DAbstractDataset* dataset) {
	// check dataset caracteristics
	// get seismic used for extraction of horizon
	QString horizonExtractionDatasetPath = horizon->seismicOriginPath();

	bool isCubeCompatible = false;

	// check compatibility on carte (inline & xline)
	CubeSeismicAddon addon = dataset->cubeSeismicAddon();

	std::size_t nbTraces, nbProfiles;
	float oriTraces, pasTraces, oriProfiles, pasProfiles;

	inri::Xt xt(horizonExtractionDatasetPath.toStdString().c_str());
	if (!xt.is_valid()) {
		isCubeCompatible = false;
	} else {
		nbTraces = xt.nRecords();
		nbProfiles = xt.nSlices();
		oriTraces = xt.startRecord();
		pasTraces = xt.stepRecords();
		oriProfiles = xt.startSlice();
		pasProfiles = xt.stepSlices();

		isCubeCompatible = nbTraces==dataset->width() && nbProfiles==dataset->depth() && oriProfiles==addon.getFirstInline() &&
					pasProfiles==addon.getInlineStep() && oriTraces==addon.getFirstXline() && pasTraces==addon.getXlineStep();
	}
	return isCubeCompatible;
}

MemoryIsochrone* IJKHorizon::getIsochrone() {
	MemoryIsochrone* isochrone = nullptr;
	bool valid;

	long dimy, dimz;
	inri::Xt xt(m_seismicOriginPath.toStdString().c_str());
	if (xt.is_valid()) {
		dimy = xt.nRecords();
		dimz = xt.nSlices();
		valid = true;
	} else {
		valid = false;
	}

	float* tab = nullptr;
	FILE* file = nullptr;
	if (valid) {
		file = fopen(m_path.toStdString().c_str(), "r");
		qDebug() << "get isochrone from : " << m_path;
		valid = file != nullptr;
	}
	if (valid) {
		tab = (float*) malloc(dimy*dimz*sizeof(float));
		long count = fread(tab, sizeof(float), dimy*dimz, file);
		qDebug() << "count : " << count;
		fclose(file);

		isochrone = new MemoryIsochrone(tab, dimy, dimz, true);
	}

	return isochrone;
}

MemoryIsochrone::MemoryIsochrone(float* oriTab, long numTraces, long numProfils, bool takeOwnerShip) {
	m_ownBuffer = takeOwnerShip;
	m_numTraces = numTraces;
	m_numProfils = numProfils;

	if (!m_ownBuffer) {
		m__vectorBuffer.resize(m_numTraces * m_numProfils);
		m_buffer = m__vectorBuffer.data();
		memcpy(m_buffer, oriTab, m_numTraces * m_numProfils * sizeof(float));
	} else {
		m_buffer = oriTab;
	}
}

MemoryIsochrone::~MemoryIsochrone() {
	if (m_ownBuffer) {
		free(m_buffer);
	}
}

int MemoryIsochrone::getNumTraces() const {
	return m_numTraces;
}

int MemoryIsochrone::getNumProfils() const {
	return m_numProfils;
}

float MemoryIsochrone::getValue(long i, long j, bool* ok) {
	(*ok) = i>=0 && i<m_numTraces && j>=0 && j<m_numProfils;
	float val;
	if (*ok) {
		val = m_buffer[i + j * m_numTraces];
	}
	return val;
}

float* MemoryIsochrone::getTab() {
	return m_buffer;
}

