#include "bnnitrainingset.h"

#include <QFileInfo>

BnniTrainingSet::BnniTrainingSet(QObject* parent) : QObject(parent) {
	m_trainingSetName = DEFAULT_TRAINING_SET_NAME;
}

BnniTrainingSet::~BnniTrainingSet() {

}

const std::map<long, BnniTrainingSet::SeismicParameter>& BnniTrainingSet::seismics() const {
	return m_seismics;
}

long BnniTrainingSet::createNewSeismic() {
	long id = nextId();
	SeismicParameter params;
	m_seismics[id] = params;
	return id;
}

void BnniTrainingSet::deleteSeismic(long id) {
	m_seismics.erase(id);
	updateSeismicUnitFromSeismics();
}

bool BnniTrainingSet::changeSeismic(long id, BnniTrainingSet::SeismicParameter param) {
	bool contain = m_seismics.find(id)!=m_seismics.end();
	if (contain && m_seismics.size()>1) {
		// reject param if sampleUnit is invalid
		contain = param.unit==m_seismicUnit || m_seismicUnit==SampleUnit::NONE || param.unit==SampleUnit::NONE;
	}
	if (contain) {
		m_seismics[id] = param;
		updateSeismicUnitFromSeismics();
	}
	return contain;
}

float BnniTrainingSet::targetSampleRate() const {
	return m_pasSampleSurrechantillon;
}

void BnniTrainingSet::setTargetSampleRate(float val) {
	m_pasSampleSurrechantillon = val;
}

int BnniTrainingSet::halfWindow() const {
	return m_halfWindow;
}

void BnniTrainingSet::setHalfWindow(int val) {
	m_halfWindow = val;
}

const std::map<long, BnniTrainingSet::WellParameter>& BnniTrainingSet::wellBores() const {
	return m_wellBores;
}

long BnniTrainingSet::createNewWell() {
	long id = nextId();
	BnniTrainingSet::WellParameter well;
	std::map<long, BnniWellHeader>::const_iterator it = m_wellHeaders.begin();
	while (it!=m_wellHeaders.end()) {
		well.logsPathAndName[it->first] = std::pair<QString, QString>("", "");
		it++;
	}
	m_wellBores[id] = well;
	return id;
}

const std::map<long, BnniTrainingSet::BnniWellHeader>& BnniTrainingSet::wellHeaders() const {
	return m_wellHeaders;
}

long BnniTrainingSet::createNewKind() {
	long id = nextId();
	BnniTrainingSet::BnniWellHeader kind;
	std::map<long, WellParameter>::iterator it = m_wellBores.begin();
	while (it!=m_wellBores.end()) {
		it->second.logsPathAndName[id] = std::pair<QString, QString>("", "");
		it++;
	}
	m_wellHeaders[id] = kind;
	return id;
}

bool BnniTrainingSet::changeWellBore(long id, WellParameter wellParam) {
	bool contain = wellParam.logsPathAndName.size()==m_wellHeaders.size() &&
			m_wellBores.find(id)!=m_wellBores.end();
	if (contain) {
		std::map<long, std::pair<QString, QString>>::const_iterator newIt = wellParam.logsPathAndName.begin();
		std::map<long, BnniWellHeader>::const_iterator oriIt = m_wellHeaders.begin();
		while (oriIt!=m_wellHeaders.end() && oriIt->first==newIt->first) {// work because keys are sorted
			newIt++;
			oriIt++;
		}
		contain = oriIt==m_wellHeaders.end();
	}
	if (contain) {
		m_wellBores[id] = wellParam;
	}
	return contain;
}

bool BnniTrainingSet::changeKind(long id, BnniWellHeader header) {
	bool contain = m_wellHeaders.find(id)!=m_wellHeaders.end();

	if (contain) {
		m_wellHeaders[id] = header;
	}
	return contain;
}

void BnniTrainingSet::deleteKind(long kindId) {
	m_wellHeaders.erase(kindId);
	std::map<long, WellParameter>::iterator it = m_wellBores.begin();
	while (it!=m_wellBores.end()) {
		it->second.logsPathAndName.erase(kindId);
		it++;
	}
}

void BnniTrainingSet::deleteWell(long wellId) {
	m_wellBores.erase(wellId);
}

const QString& BnniTrainingSet::tfpFilter() const {
	return m_tfpFilter;
}

void BnniTrainingSet::setTfpFilter(const QString& newTfpFilter) {
	m_tfpFilter = newTfpFilter;
}

bool BnniTrainingSet::useBandPassHighFrequency() const {
	return m_useBandPassHighFrequency;
}

float BnniTrainingSet::bandPassHighFrequency() const {
	return m_bandPassHighFrequency;
}

void BnniTrainingSet::activateBandPassHighFrequency(float freq) {
	m_bandPassHighFrequency = freq;
	m_useBandPassHighFrequency = true;
}

void BnniTrainingSet::deactivateBandPassHighFrequency() {
	m_useBandPassHighFrequency = false;
}

float BnniTrainingSet::mdSamplingRate() const {
	return m_mdSamplingRate;
}

void BnniTrainingSet::setMdSamplingRate(float val) {
	m_mdSamplingRate = val;
}

bool BnniTrainingSet::useAugmentation() const {
	return m_useAugmentation;
}

void BnniTrainingSet::setUseAugmentation(bool val) {
	m_useAugmentation = val;
}

int BnniTrainingSet::augmentationDistance() const {
	return m_augmentationDistance;
}

void BnniTrainingSet::setAugmentationDistance(int dist) {
	m_augmentationDistance = dist;
}

float BnniTrainingSet::gaussianNoiseStd() const {
	return m_gaussianNoiseStd;
}

void BnniTrainingSet::setGaussianNoiseStd(float val) {
	m_gaussianNoiseStd = val;
}

bool BnniTrainingSet::useCnxAugmentation() const {
	return m_useCnxAugmentation;
}

void BnniTrainingSet::toggleCnxAugmentation(bool val) {
	m_useCnxAugmentation = val;
}

QString BnniTrainingSet::projectPath() const {
	return m_projectPath;
}

void BnniTrainingSet::setProjectPath(const QString& projectPath) {
	m_projectPath = projectPath;
	searchNewTrainingSetName();
}

QString BnniTrainingSet::trainingSetName() const {
	return m_trainingSetName;
}

void BnniTrainingSet::setTrainingSetName(const QString& trainingSetName) {
	m_trainingSetName = trainingSetName;
	setOutputJsonFile(computeOutputJsonFile(m_projectPath, m_trainingSetName));
}

QString BnniTrainingSet::outputJsonFile() const {
	return m_outputJsonFile;
}

void BnniTrainingSet::setOutputJsonFile(const QString& jsonPath) {
	m_outputJsonFile = jsonPath;
	emit outputJsonFileChanged(m_outputJsonFile);
}

void BnniTrainingSet::searchNewTrainingSetName() {
	QString trainingSetName = DEFAULT_TRAINING_SET_NAME;
	QString newJsonPath = computeOutputJsonFile(m_projectPath, trainingSetName);
	int n = 0;
	while (QFileInfo(newJsonPath).exists()) {
		n++;
		trainingSetName = DEFAULT_TRAINING_SET_NAME + "_" + QString::number(n);
		newJsonPath = computeOutputJsonFile(m_projectPath, trainingSetName);
	}
	setTrainingSetName(trainingSetName);
}

QString BnniTrainingSet::computeOutputJsonFile(const QString& projectPath,
		const QString& trainingSetName) {
	return projectPath + "/" + DEFAULT_RELATIVE_PATH + "/" + trainingSetName +
			"/" + DEFAULT_JSON_NAME;
}

long BnniTrainingSet::nextId() const {
	return m_nextId++;
}

bool BnniTrainingSet::isLogEmpty(long logId, std::map<long, std::pair<QString, QString>> logMap) {
	std::map<long, std::pair<QString, QString>>::const_iterator it = logMap.find(logId);
	return it==logMap.end() || it->second.first.isNull() || it->second.first.isEmpty() ||
							it->second.second.isNull() || it->second.second.isEmpty();
}

const std::map<long, BnniTrainingSet::HorizonIntervalParameter>& BnniTrainingSet::intervals() const {
	return m_horizonIntervals;
}

long BnniTrainingSet::createNewInterval() { // return index
	long id = nextId();
	HorizonIntervalParameter params;
	m_horizonIntervals[id] = params;
	return id;
}

void BnniTrainingSet::deleteInterval(long id) {
	m_horizonIntervals.erase(id);
}

bool BnniTrainingSet::changeInterval(long id, HorizonIntervalParameter param) {
	bool contain = m_horizonIntervals.find(id)!=m_horizonIntervals.end();
	if (contain) {
		m_horizonIntervals[id] = param;
	}
	return contain;
}

SampleUnit BnniTrainingSet::seismicUnit() const {
	return m_seismicUnit;
}

void BnniTrainingSet::setSeismicUnit(SampleUnit sampleUnit) {
	if (sampleUnit!=m_seismicUnit) {
		m_seismicUnit = sampleUnit;
		emit seismicUnitChanged(m_seismicUnit);
	}
}

void BnniTrainingSet::updateSeismicUnitFromSeismics() {
	SampleUnit out = SampleUnit::NONE;

	// search first valid sample unit, trust the first one and others should be of the same sample unit
	std::map<long, SeismicParameter>::const_iterator it = m_seismics.begin();
	while (out==SampleUnit::NONE && it!=m_seismics.end()) {
		out = it->second.unit;
		it++;
	}
	setSeismicUnit(out);
}

