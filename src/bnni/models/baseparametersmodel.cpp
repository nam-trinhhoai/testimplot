#include "treeparametersmodel.h"

#include "fileSelectorDialog.h"

#include <QMessageBox>

BaseParametersModel::BaseParametersModel(QObject* parent) : QObject(parent) {

}

BaseParametersModel::~BaseParametersModel() {

}

void BaseParametersModel::editReferenceCheckpoint() {
	QString newCheckpoint;

	bool changeCheckpoint = true;
	QDir dir(m_checkpointDir);
	QString referenceAbsoluteFilePath = dir.absoluteFilePath(CHECKPOINT_REFERENCE + "." + getExtension());
	QFileInfo referenceInfo(referenceAbsoluteFilePath);
	if (referenceInfo.exists() && referenceInfo.isReadable()) {
		std::vector<QString> allCheckpoints = {"no reference", CHECKPOINT_REFERENCE};
		FileSelectorDialog dialog(&allCheckpoints, "Select checkpoint");
		int res = dialog.exec();
		changeCheckpoint = res==QDialog::Accepted;
		if (changeCheckpoint) {
			int index = dialog.getSelectedIndex();
			changeCheckpoint = index>=0 && index<allCheckpoints.size();
			if (changeCheckpoint && index>0) {
				newCheckpoint = allCheckpoints[index];
			} else if (changeCheckpoint) {
				newCheckpoint = "";
			}
		}
	} else {
		QMessageBox::information(nullptr, "Checkpoint selection", "There is no references available.");
	}
	if (changeCheckpoint) {
		setReferenceCheckpointPrivate(newCheckpoint);
	}
}

std::vector<QString> BaseParametersModel::getAvailableCheckpoints() const {
	std::vector<QString> result;
	if (m_checkpointDir.isNull() || m_checkpointDir.isEmpty()) {
		return result;
	}
	QDir dir(m_checkpointDir);
	if (!dir.exists() || !dir.isReadable()) {
		return result;
	}

	QFileInfoList list = dir.entryInfoList(QStringList() << "*."+getExtension(), QDir::Files, QDir::Time);
	result.resize(list.size(), "");
	for (int i=0; i<list.size(); i++) {
		result[i] = list[i].baseName();
	}
	return result;
}

bool BaseParametersModel::validateArguments() {
	bool preprocessingValid = (m_hatParameter>0 && m_seismicPreprocessing==SeismicPreprocessing::SeismicHat) ||
			m_seismicPreprocessing==SeismicPreprocessing::SeismicNone;
	bool postprocessingValid = m_wellPostprocessing==WellPostprocessing::WellNone ||
			(m_wellFilterFrequency>=std::numeric_limits<float>::min() && m_wellPostprocessing==WellPostprocessing::WellFilter);

	bool test = m_nGpus < 99 && !m_savePrefix.isNull() && !m_savePrefix.isEmpty() &&
		m_epochSaveStep>0 && preprocessingValid && postprocessingValid && m_learningRate>0;

	return test;
}

QStringList BaseParametersModel::warningForArguments() {
	QStringList warnings;
	if (m_nGpus >= 99) {
		warnings << "Too many gpus";
	}
	if (m_savePrefix.isNull() || m_savePrefix.isEmpty()) {
		warnings << "Save prefix not defined";
	}
	if (m_epochSaveStep<=0) {
		warnings << "Epoch Saving Step is too low, please use an integer greater than 0";
	}
	bool preprocessingValid = (m_hatParameter>0 && m_seismicPreprocessing==SeismicPreprocessing::SeismicHat) ||
				m_seismicPreprocessing==SeismicPreprocessing::SeismicNone;
	if (!preprocessingValid) {
		warnings << "Hat is too low, please use an integer greater than 0";
	}
	bool postprocessingValid = m_wellPostprocessing==WellPostprocessing::WellNone ||
			(m_wellFilterFrequency>=std::numeric_limits<float>::min() && m_wellPostprocessing==WellPostprocessing::WellFilter);
	if (!postprocessingValid) {
		warnings << "Filter frequency is too low, please use a real greater than 0";
	}
	if (m_learningRate<=0) {
		warnings << "Learning rate is too low, please use a real greater than 0";
	}
	return warnings;
}

double BaseParametersModel::getLearningRate() const {
	return m_learningRate;
}

bool BaseParametersModel::loadLearningRate(QString txt) {
	bool ok;
	float val = txt.toFloat(&ok);
	ok = ok && val > 0;
	if (ok) {
		setLearningRate(val);
	}
	return ok;
}

void BaseParametersModel::setLearningRate(double val) {
	if (m_learningRate!=val) {
		m_learningRate = val;
		emit learningRateChanged(m_learningRate);
	}
}

unsigned int BaseParametersModel::getEpochSaveStep() const {
	return m_epochSaveStep;
}

bool BaseParametersModel::loadEpochSaveStep(QString txt) {
	bool ok;
	unsigned int val = txt.toUInt(&ok);
	ok = ok && val > 0;
	if (ok) {
		setEpochSaveStep(val);
	}
	return ok;
}

void BaseParametersModel::setEpochSaveStep(unsigned int val) {
	if (m_epochSaveStep!=val) {
		m_epochSaveStep = val;
		emit epochSaveStepChanged(m_epochSaveStep);
	}
}

unsigned int BaseParametersModel::getNGpus() const {
	return m_nGpus;
}

bool BaseParametersModel::loadNGpus(QString txt) {
	bool ok;
	unsigned int val = txt.toUInt(&ok);
	if (ok) {
		setNGpus(val);
	}
	return ok;
}

void BaseParametersModel::setNGpus(unsigned int val) {
	if (m_nGpus!=val) {
		m_nGpus = val;
		emit nGpusChanged(m_nGpus);
	}
}

QString BaseParametersModel::getSavePrefix() const {
	return m_savePrefix;
}

bool BaseParametersModel::loadSavePrefix(QString txt) {
	setSavePrefix(txt);
	return true;
}

void BaseParametersModel::setSavePrefix(const QString& txt) {
	if (m_savePrefix.compare(txt)!=0) {
		m_savePrefix = txt;
		emit savePrefixChanged(m_savePrefix);
	}
}

SeismicPreprocessing BaseParametersModel::getSeismicPreprocessing() const {
	return m_seismicPreprocessing;
}

bool BaseParametersModel::loadSeismicPreprocessing(QString txt) {
	bool out = true;
	if (txt.compare("hat")==0) {
		setSeismicPreprocessing(SeismicPreprocessing::SeismicHat);
	} else if(txt.compare("normal")==0) {
		setSeismicPreprocessing(SeismicPreprocessing::SeismicNone);
	} else {
		out = false;
	}
	return out;
}

void BaseParametersModel::setSeismicPreprocessing(SeismicPreprocessing val) {
	if (m_seismicPreprocessing!=val) {
		m_seismicPreprocessing = val;
		emit seismicPreprocessingChanged(m_seismicPreprocessing);
	}
}

int BaseParametersModel::getHatParameter() const {
	return m_hatParameter;
}

bool BaseParametersModel::loadHatParameter(QString txt) {
	bool ok;
	unsigned int val = txt.toUInt(&ok);
	ok = ok && val >= 1;
	if (ok) {
		setHatParameter(val);
	}
	return ok;
}

void BaseParametersModel::setHatParameter(int val) {
	if (m_hatParameter!=val) {
		m_hatParameter = val;
		emit hatParameterChanged(m_hatParameter);
	}
}

WellPostprocessing BaseParametersModel::getWellPostprocessing() const {
	return m_wellPostprocessing;
}

bool BaseParametersModel::loadWellPostprocessing(QString txt) {
	bool out = true;
	if (txt.compare("filter")==0) {
		setWellPostprocessing(WellPostprocessing::WellFilter);
	} else if(txt.compare("normal")==0) {
		setWellPostprocessing(WellPostprocessing::WellNone);
	} else {
		out = false;
	}
	return out;
}

void BaseParametersModel::setWellPostprocessing(WellPostprocessing val) {
	if (m_wellPostprocessing!=val) {
		m_wellPostprocessing = val;
		emit wellPostprocessingChanged(m_wellPostprocessing);
	}
}

float BaseParametersModel::getWellFilterFrequency() const {
	return m_wellFilterFrequency;
}

bool BaseParametersModel::loadWellFilterFrequency(QString txt) {
	bool ok;
	float val = txt.toFloat(&ok);
	ok = ok && val >= std::numeric_limits<float>::min();
	if (ok) {
		setWellFilterFrequency(val);
	}
	return ok;
}

void BaseParametersModel::setWellFilterFrequency(float val) {
	if (m_wellFilterFrequency!=val) {
		m_wellFilterFrequency = val;
		emit wellFilterFrequencyChanged(m_wellFilterFrequency);
	}
}

bool BaseParametersModel::hasReferenceCheckpoint() const {
	return !m_referenceCheckpoint.isNull() && !m_referenceCheckpoint.isEmpty();
}

QString BaseParametersModel::getReferenceCheckpoint() const {
	return m_referenceCheckpoint;
}

bool BaseParametersModel::loadReferenceCheckpoint(QString txt) {
	return setReferenceCheckpoint(txt);
}

bool BaseParametersModel::setReferenceCheckpoint(const QString& txt) {
	if (txt.isNull() || txt.isEmpty()) {
		setReferenceCheckpointPrivate(txt);
		return true;
	}

	std::vector<QString> allCheckpoints = getAvailableCheckpoints();
	auto it = std::find(allCheckpoints.begin(), allCheckpoints.end(), txt);

	bool res = it!=allCheckpoints.end();
	if (res) {
		setReferenceCheckpointPrivate(txt);
	}
	return res;
}

void BaseParametersModel::setReferenceCheckpointPrivate(const QString& txt) {
	if (m_referenceCheckpoint!=txt) {
		m_referenceCheckpoint = txt;
		emit referenceCheckpointChanged(m_referenceCheckpoint);
	}
}

QString BaseParametersModel::getCheckpointDir() const {
	return m_checkpointDir;
}

void BaseParametersModel::setCheckpointDir(const QString& checkpointDir) {
	if (m_checkpointDir.compare(checkpointDir)!=0) {
		m_checkpointDir = checkpointDir;
		emit checkpointDirChanged(m_checkpointDir);

		// check
		bool res = setReferenceCheckpoint(m_referenceCheckpoint);
		if (!res) {
			setReferenceCheckpointPrivate("");
		}
	}
}
