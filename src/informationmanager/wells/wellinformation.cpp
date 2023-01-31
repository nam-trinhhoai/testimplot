#include "wellinformation.h"

#include "DataSelectorDialog.h"
#include "folderdata.h"
#include "nurbinformationmetadatawidget.h"
#include "ProjectManagerNames.h"
#include "propertyfiltersparser.h"
#include "wellbore.h"
#include "wellhead.h"
#include "wellinformationpanelwidget.h"
#include "wellpick.h"
#include "WellUtil.h"
#include "workingsetmanager.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <algorithm>


WellInformation::WellInformation(const QString& wellBoreDir, WorkingSetManager* manager, QObject* parent) : IInformation(parent),
		m_wellBoreDir(wellBoreDir), m_workingSetManager(manager) {
	searchLoadedData();
}

WellInformation::~WellInformation() {

}

bool WellInformation::isDeletable() const {
	return false;
}

bool WellInformation::deleteStorage(QString* errorMsg) {
	// does not support deletion
	if (errorMsg) {
		*errorMsg = tr("Wells cannot be deleted");
	}
	return false;
}

bool WellInformation::isSelectable() const {
	return !m_workingSetManager.isNull() && m_workingSetManager->getManagerWidget()!=nullptr;
}

bool WellInformation::isSelected() const {
	return isSelectable() && !m_loadedData.isNull();
}

void WellInformation::toggleSelection(bool toggle) {
	if (!isSelectable() || toggle==isSelected()) {
		return;
	}

	// this cannot be threaded as is, it would need a synchronization point
	bool valid = true;
	if (toggle) {
		valid = selectWell(m_wellBoreDir, m_workingSetManager);
	} else {
		valid = unselectWell(m_wellBoreDir, m_workingSetManager);
	}

	if (valid) {
		if (!toggle) {
			m_loadedData = nullptr;
		} else {
			searchLoadedData();
		}

		if (m_workingSetManager && m_workingSetManager->getManagerWidget()) {
			m_workingSetManager->getManagerWidget()->save_to_default_session();
		}
	}
}

void WellInformation::searchLoadedData() {
	if (m_workingSetManager==nullptr || m_loadedData!=nullptr) {
		return;
	}

	QString boreDescFile = wellBoreDescFile();

	QList<IData*> datas = m_workingSetManager->folders().wells->data();
	int i = 0;
	while (m_loadedData==nullptr && i<datas.size()) {
		WellHead* wellHead = dynamic_cast<WellHead*>(datas[i]);
		if (wellHead!=nullptr) {
			QList<WellBore*> wellBores = wellHead->wellBores();

			int j = 0;
			while (m_loadedData==nullptr && j<wellBores.size()) {
				WellBore* wellBore = wellBores[j];
				if (wellBore->isIdPathIdentical(boreDescFile)) {
					m_loadedData = wellBore;
				} else {
					j++;
				}
			}
		}
		if (m_loadedData==nullptr) {
			i++;
		}
	}
	if (m_loadedData) {
		m_loadedData->setAllDisplayPreference(true);
		if (m_loadedData->wellHead()) {
			m_loadedData->wellHead()->setAllDisplayPreference(true);
		}
		QList<WellPick*> picks = m_loadedData->picks();
		for (long i=0; i<picks.size(); i++) {
			picks[i]->setAllDisplayPreference(true);
		}
	}
}

bool WellInformation::commentsEditable() const {
	return false;
}

QString WellInformation::comments() const {
	return "";
}

void WellInformation::setComments(const QString& txt) {
	// comments are disabled
}

bool WellInformation::hasIcon() const {
	return false;
}

QIcon WellInformation::icon(int preferedSizeX, int preferedSizeY) const {
	return QIcon();
}

QString WellInformation::mainOwner() const {
	QString txt;
	if (m_cacheOwners.size()>0) {
		txt = m_cacheOwners[0];
	} else {
		QStringList ownerList = owners();
		if (ownerList.size()>0) {
			txt = ownerList[0];
		}
	}
	return txt;
}

QStringList WellInformation::owners() const {
	if (m_cacheOwners.size()==0) {
		searchFileCache();
	}
	return m_cacheOwners;
}

QDateTime WellInformation::mainCreationDate() const {
	QDateTime date;
	if (m_cacheCreationDates.size()>0) {
		date = m_cacheCreationDates[0];
	} else {
		QList<QDateTime> dateList = creationDates();
		if (dateList.size()>0) {
			date = dateList[0];
		}
	}
	return date;
}

QList<QDateTime> WellInformation::creationDates() const {
	if (m_cacheCreationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheCreationDates;
}

QDateTime WellInformation::mainModificationDate() const {
	QDateTime date;
	if (m_cacheModificationDates.size()>0) {
		date = m_cacheModificationDates[0];
	} else {
		QList<QDateTime> dateList = modificationDates();
		if (dateList.size()>0) {
			date = dateList[0];
		}
	}
	return date;
}

QList<QDateTime> WellInformation::modificationDates() const {
	if (m_cacheModificationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheModificationDates;
}

QString WellInformation::name() const {
	return wellHeadName() + " " + wellBoreName();
}

QString WellInformation::wellBoreName() const {
	QString txt = m_cacheWellBore;
	if (txt.isNull() || txt.isEmpty()) {
		if (!m_loadedData.isNull()) {
			txt = m_loadedData->name();
		} else {
			txt = getWellTinyName(m_wellBoreDir);
		}
		m_cacheWellBore = txt;
	}
	return txt;
}

QString WellInformation::wellHeadName() const {
	QString txt = m_cacheWellHead;
	if (txt.isNull() || txt.isEmpty()) {
		if (!m_loadedData.isNull()) {
			txt = m_loadedData->wellHead()->name();
		} else {
			QString wellHeadPath = QFileInfo(m_wellBoreDir).dir().absolutePath();
			txt = getWellTinyName(wellHeadPath);
		}
		m_cacheWellHead = txt;
	}
	return txt;
}

QStringList WellInformation::wellKinds() const {
	if (!logRetrieved) {
		searchLogsTfpsPicks();
	}

	QStringList kinds;
	for (int i=0; i<m_cacheWellLogPaths.size(); i++) {
		QString path = m_cacheWellLogPaths[i];

		QString kind;
		auto kindIt = m_cacheWellKinds.find(path);
		if (kindIt!=m_cacheWellKinds.end()) {
			kind = kindIt->second;
		} else {
			kind = WellBore::getKindFromLogFile(path);
			m_cacheWellKinds[path] = kind;
		}
		if (!kind.isNull() && !kind.isEmpty()) {
			kinds.append(kind);
		}
	}
	return kinds;
}

QStringList WellInformation::wellLogs() const {
	if (!logRetrieved) {
		searchLogsTfpsPicks();
	}
	return m_cacheWellLogNames;
}

QStringList WellInformation::wellPicks() const {
	if (!logRetrieved) {
		searchLogsTfpsPicks();
	}
	return m_cacheWellPickNames;
}

QStringList WellInformation::wellTfps() const {
	if (!logRetrieved) {
		searchLogsTfpsPicks();
	}
	return m_cacheWellTfpNames;
}

QStringList WellInformation::wellTfpPaths() const {
	if (!logRetrieved) {
		searchLogsTfpsPicks();
	}
	return m_cacheWellTfpPaths;
}

information::StorageType WellInformation::storage() const {
	return information::StorageType::SISMAGE;
}

QString WellInformation::getWellTinyName(const QString& wellPath) {
	QFileInfo fileInfo(wellPath);
	QString tinyname = fileInfo.fileName();
	QString fullname = fileInfo.absoluteFilePath();
	QDir boreDir(fullname);
	QString boreDescName = tinyname + ".desc";
	if (boreDir.exists(boreDescName)) {
		QString descFile = boreDir.absoluteFilePath(boreDescName);
		QString name = ProjectManagerNames::getKeyTabFromFilename(descFile, "Name");
		if (!name.isNull() && !name.isEmpty()) {
			tinyname = name;
		}
	}
	return tinyname;
}

void WellInformation::searchFileCache() const {
	m_cacheCreationDates.clear();
	m_cacheModificationDates.clear();
	m_cacheOwners.clear();

	QFileInfo boreFileInfo(m_wellBoreDir);
	QDateTime mainCreationTime = boreFileInfo.birthTime();
	QDateTime mainModificationTime = boreFileInfo.lastModified();
	QString mainOwner = boreFileInfo.owner();
	if (mainCreationTime.isValid()) {
		m_cacheCreationDates.append(mainCreationTime);
	}
	if (mainModificationTime.isValid()) {
		m_cacheModificationDates.append(mainModificationTime);
	}
	if (!mainOwner.isNull() && !mainOwner.isEmpty()) {
		m_cacheOwners.append(mainOwner);
	}
	QDir boreDir = boreFileInfo.dir();
	QFileInfoList infoList = boreDir.entryInfoList(QStringList() << "*", QDir::NoDotAndDotDot);
	for (int i=0; i<infoList.size(); i++) {
		QFileInfo& fileInfo = infoList[i];
		QDateTime creationTime = fileInfo.birthTime();
		QDateTime modificationTime = fileInfo.lastModified();
		QString owner = fileInfo.owner();

		if (creationTime.isValid() && !m_cacheCreationDates.contains(creationTime)) {
			m_cacheCreationDates.append(creationTime);
		}
		if (modificationTime.isValid() && !m_cacheModificationDates.contains(modificationTime)) {
			m_cacheModificationDates.append(modificationTime);
		}
		if (!owner.isNull() && !owner.isEmpty() && !m_cacheOwners.contains(owner)) {
			m_cacheOwners.append(owner);
		}
	}
}

bool WellInformation::hasProperty(information::Property property) const {
	return property==information::Property::CREATION_DATE || property==information::Property::MODIFICATION_DATE ||
			property==information::Property::NAME || property==information::Property::OWNER ||
			property==information::Property::STORAGE_TYPE || property==information::Property::WELL_BORE ||
			property==information::Property::WELL_HEAD || property==information::Property::WELL_KIND ||
			property==information::Property::WELL_LOG_NAME || property==information::Property::WELL_PICK_NAME ||
			property==information::Property::WELL_TFP_NAME;
}

QVariant WellInformation::property(information::Property property) const {
	QVariant out;

	switch (property) {
	case information::Property::CREATION_DATE:
		out = PropertyFiltersParser::toVariant(creationDates());
		break;
	case information::Property::MODIFICATION_DATE:
		out = PropertyFiltersParser::toVariant(modificationDates());
		break;
	case information::Property::NAME:
		out = name();
		break;
	case information::Property::OWNER:
		out = owners();
		break;
	case information::Property::STORAGE_TYPE:
		out = PropertyFiltersParser::storageToString(storage());
		break;
	case information::Property::WELL_BORE:
		out = wellBoreName();
		break;
	case information::Property::WELL_HEAD:
		out = wellHeadName();
		break;
	case information::Property::WELL_KIND:
		out = wellKinds();
		break;
	case information::Property::WELL_LOG_NAME:
		out = wellLogs();
		break;
	case information::Property::WELL_PICK_NAME:
		out = wellPicks();
		break;
	case information::Property::WELL_TFP_NAME:
		out = wellTfps();
		break;
	}

	return out;
}

bool WellInformation::isCompatible(information::Property property, const QVariant& filter) const {
	if (!hasProperty(property)) {
		return false;
	}

	bool isCompat = false;
	if (property==information::Property::WELL_KIND) {
		if (!logRetrieved) {
			searchLogsTfpsPicks();
		}

		int i = 0;
		while (!isCompat && i<m_cacheWellLogPaths.size()) {
				QString path = m_cacheWellLogPaths[i];

				QString kind;
				auto kindIt = m_cacheWellKinds.find(path);
				if (kindIt!=m_cacheWellKinds.end()) {
					kind = kindIt->second;
				} else {
					kind = WellBore::getKindFromLogFile(path);
					m_cacheWellKinds[path] = kind;
				}
				isCompat = !kind.isNull() && !kind.isEmpty() &&
						PropertyFiltersParser::isCompatible(property, filter, kind);

			if (!isCompat) {
				i++;
			}
		}
	} else {
		QVariant value = this->property(property);
		isCompat = PropertyFiltersParser::isCompatible(property, filter, value);
	}
	return isCompat;
}

IInformationPanelWidget* WellInformation::buildInformationWidget(QWidget* parent) {
	return new WellInformationPanelWidget(this, parent);
}

QWidget* WellInformation::buildMetadataWidget(QWidget* parent) {
	return new NurbInformationMetadataWidget(this, parent);
}

QString WellInformation::folder() const {
	return m_wellBoreDir;
}

QString WellInformation::mainPath() const {
	return m_wellBoreDir;
}

QString WellInformation::currentTfpName() const {
	QString tfpName;

	if (m_loadedData) {
		tfpName = m_loadedData->getTfpName();
	}

	return tfpName;
}

QString WellInformation::currentTfpPath() const {
	QString tfpName;

	if (m_loadedData) {
		tfpName = m_loadedData->getTfpFilePath();
	}

	return tfpName;
}

QString WellInformation::defaultTfp() const {
	QString defaultAbsolutePath = WellBore::getTfpFileFromDescFile(wellBoreDescFile());
	QString name = ProjectManagerNames::getKeyTabFromFilename(defaultAbsolutePath, "Name");
	return name;
}

PMANAGER_BORE_DISPLAY WellInformation::getBoreInList(bool* ok) const {
	PMANAGER_BORE_DISPLAY out;
	if (m_workingSetManager==nullptr || m_workingSetManager->getManagerWidget()==nullptr) {
		if (ok) {
			*ok = false;
		}
		return out;
	}
	std::vector<PMANAGER_WELL_DISPLAY> wellDisplayList = m_workingSetManager->getManagerWidget()->get_display_well_list();

	QString absoluteFilePath = QFileInfo(m_wellBoreDir).absoluteFilePath();
	auto headIt = std::find_if(wellDisplayList.begin(), wellDisplayList.end(), [absoluteFilePath](const PMANAGER_WELL_DISPLAY& wellDisplay) {
		auto it = std::find_if(wellDisplay.bore.begin(), wellDisplay.bore.end(), [absoluteFilePath](const PMANAGER_BORE_DISPLAY& boreDisplay) {
			return absoluteFilePath.compare(QFileInfo(boreDisplay.bore_fullname).absoluteFilePath())==0;
		});
		return it!=wellDisplay.bore.end();
	});
	bool valid = headIt!=wellDisplayList.end();
	if (valid) {
		std::vector<PMANAGER_BORE_DISPLAY>::const_iterator boreIt;
		boreIt = std::find_if(headIt->bore.begin(), headIt->bore.end(), [absoluteFilePath](const PMANAGER_BORE_DISPLAY& boreDisplay) {
			return absoluteFilePath.compare(QFileInfo(boreDisplay.bore_fullname).absoluteFilePath())==0;
		});
		valid = boreIt!=headIt->bore.end();
		if (valid) {
			out = *boreIt;
		}
	}

	if (ok) {
		*ok = valid;
	}

	return out;
}

WellInformation::WellBoreDescParams WellInformation::readDescFile(const QString& descFile)
{
	WellBoreDescParams wellBoreDescParams;

	QFile file(descFile);
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		qDebug() << "WellBore : cannot read desc file in text format " << descFile;
		return wellBoreDescParams;
	}

	QTextStream in(&file);
	while (!in.atEnd()) {
		QString line = in.readLine();
		QStringList lineSplit = line.split("\t");
		if(lineSplit.size()>1 && lineSplit.first().compare("Datum")==0) {
			wellBoreDescParams.datum = lineSplit[1];
		}
		else  if(lineSplit.size()>1 && lineSplit.first().compare("Status")==0) {
			wellBoreDescParams.status = lineSplit[1];
		}
		else if(lineSplit.size()>1 && lineSplit.first().compare("Elev")==0) {
			wellBoreDescParams.elev = lineSplit[1];
		}
		else if(lineSplit.size()>1 && lineSplit.first().compare("UWI")==0) {
			wellBoreDescParams.uwi = lineSplit[1];
		}
		else if(lineSplit.size()>1 && lineSplit.first().compare("Domain")==0) {
			wellBoreDescParams.domain = lineSplit[1];
		}
		else if(lineSplit.size()>1 && lineSplit.first().compare("Velocity")==0) {
			wellBoreDescParams.velocity = lineSplit[1];
		}
		else if(lineSplit.size()>1 && lineSplit.first().compare("IHS")==0) {
			wellBoreDescParams.ihs = lineSplit[1];
		}
	}

	return wellBoreDescParams;
}

void WellInformation::searchLogsTfpsPicks() const {
	if (logRetrieved || m_workingSetManager==nullptr) {
		return;
	}
	m_cacheWellLogNames.clear();
	m_cacheWellLogPaths.clear();
	m_cacheWellTfpNames.clear();
	m_cacheWellTfpPaths.clear();
	m_cacheWellPickNames.clear();

	bool valid;
	PMANAGER_BORE_DISPLAY wellBore = getBoreInList(&valid);

	if (valid) {
		logRetrieved = true;
		for (long i=0; i<wellBore.log_tinyname.size(); i++) {
			m_cacheWellLogNames << wellBore.log_tinyname[i];
		}
		for (long i=0; i<wellBore.log_fullname.size(); i++) {
			m_cacheWellLogPaths << wellBore.log_fullname[i];
		}
		for (long i=0; i<wellBore.picks_tinyname.size(); i++) {
			m_cacheWellPickNames << wellBore.picks_tinyname[i];
		}
		for (long i=0; i<wellBore.tf2p_tinyname.size(); i++) {
			m_cacheWellTfpNames << wellBore.tf2p_tinyname[i];
		}
		for (long i=0; i<wellBore.tf2p_tinyname.size(); i++) {
			m_cacheWellTfpPaths << wellBore.tf2p_fullname[i];
		}
	}
}

WellInformation::WellBoreDescParams WellInformation::wellBoreDescParams() const {
	if (!m_wellBoreDescParamsSet) {
		QString descFile = wellBoreDescFile();
		m_wellBoreDescParams = readDescFile(descFile);
		m_wellBoreDescParamsSet = true;
	}
	return m_wellBoreDescParams;
}

QString WellInformation::wellBoreDescFile() const {
	QFileInfo wellFileInfo(m_wellBoreDir);
	QString descFile = m_wellBoreDir + "/" + wellFileInfo.fileName() + ".desc";
	return descFile;
}

bool WellInformation::selectWell(const QString& wellBoreDir, WorkingSetManager* manager) {
	if (manager==nullptr || manager->getManagerWidget()==nullptr) {
		return false;
	}
	manager->getManagerWidget()->set_wells_filter("");

	manager->getManagerWidget()->clear_well_gui_selection();
	bool valid = manager->getManagerWidget()->select_well_fullpath_name(wellBoreDir);

	manager->getManagerWidget()->trt_well_basket_add();

	if (valid) {
		manager->getManagerWidget()->fill_empty_logs_list();
		std::vector<WELLLIST> wellList = manager->getManagerWidget()->get_well_list();
		std::vector<PMANAGER_WELL_DISPLAY> wellDisplayList = manager->getManagerWidget()->get_display_well_list();
		const std::vector<QString>& picksNames = manager->getManagerWidget()->get_picks_names();
		const std::vector<QBrush>& picksColors = manager->getManagerWidget()->get_picks_colors();
		std::vector<MARKER> picksList = manager->getManagerWidget()->staticGetPicksSortedWells(picksNames, picksColors, wellDisplayList);
		DataSelectorDialog::addWellBore(manager, wellList, picksList, true);
	}
	return valid;
}

bool WellInformation::unselectWell(const QString& wellBoreDir, WorkingSetManager* manager) {
	if (manager==nullptr || manager->getManagerWidget()==nullptr) {
		return false;
	}
	manager->getManagerWidget()->set_wells_filter("");

	manager->getManagerWidget()->clear_well_basket_gui_selection();
	bool valid = manager->getManagerWidget()->select_well_basket_fullpath_name(wellBoreDir);

	manager->getManagerWidget()->trt_well_basket_sub();

	if (valid) {
		std::vector<WELLLIST> wellList = manager->getManagerWidget()->get_well_list();
		std::vector<PMANAGER_WELL_DISPLAY> wellDisplayList = manager->getManagerWidget()->get_display_well_list();
		const std::vector<QString>& picksNames = manager->getManagerWidget()->get_picks_names();
		const std::vector<QBrush>& picksColors = manager->getManagerWidget()->get_picks_colors();
		std::vector<MARKER> picksList = manager->getManagerWidget()->staticGetPicksSortedWells(picksNames, picksColors, wellDisplayList);
		DataSelectorDialog::addWellBore(manager, wellList, picksList, true);
	}
	return valid;
}

bool WellInformation::setCurrentTfp(const QString& tfpPath, const QString& tfpName) {
	bool valid = false;
	QString currentTfpPath = this->currentTfpPath();
	if (tfpPath.compare(currentTfpPath)!=0 && !m_loadedData.isNull()) {
//		std::vector<QString> tfpPaths = m_loadedData->tfpsPaths();
//		std::vector<QString> tfpNames = m_loadedData->tfpsNames();
//		int loadedTfpIdx = 0;
//		while (loadedTfpIdx<tfpPaths.size()) {
//			if (tfpPath.compare(tfpPaths[loadedTfpIdx])!=0) {
//				loadedTfpIdx++;
//			}
//		}
//		if (loadedTfpIdx<tfpPaths.size()) {
//			valid = m_loadedData->selectTFP(loadedTfpIdx);
//		} else {
//			tfpPaths.push_back(tfpPath);
//			tfpNames.push_back(tfpName);
//
//			m_loadedData->SetTfpsPath(tfpPaths);
//			m_loadedData->SetTfpName(tfpNames);
//			valid = m_loadedData->selectTFP(tfpNames.size()-1);
//		}
		if (/*valid &&*/ m_workingSetManager!=nullptr && m_workingSetManager->getManagerWidget()!=nullptr) {
			m_workingSetManager->getManagerWidget()->set_wells_filter("");
			m_workingSetManager->getManagerWidget()->set_tfps_filter("");

			bool selectWellOk = m_workingSetManager->getManagerWidget()->select_well_basket_tinyname("[ " + m_loadedData->wellHead()->name() + " ] " + m_loadedData->name());

			// clear all tfp to force selection of the new one
			if (selectWellOk) {
				m_workingSetManager->getManagerWidget()->select_all_tfp_basket();
				m_workingSetManager->getManagerWidget()->trt_welltf2p_basket_sub();

				valid = m_workingSetManager->getManagerWidget()->select_tfp_tinyname(tfpName);

				m_workingSetManager->getManagerWidget()->trt_welltf2p_basket_add();
			}
		}

		if (valid) {
			std::vector<WELLLIST> wellList = m_workingSetManager->getManagerWidget()->get_well_list();
			std::vector<PMANAGER_WELL_DISPLAY> wellDisplayList = m_workingSetManager->getManagerWidget()->get_display_well_list();
			const std::vector<QString>& picksNames = m_workingSetManager->getManagerWidget()->get_picks_names();
			const std::vector<QBrush>& picksColors = m_workingSetManager->getManagerWidget()->get_picks_colors();
			std::vector<MARKER> picksList = m_workingSetManager->getManagerWidget()->staticGetPicksSortedWells(picksNames, picksColors, wellDisplayList);
			DataSelectorDialog::addWellBore(m_workingSetManager, wellList, picksList, true);
		}

		if (m_workingSetManager && m_workingSetManager->getManagerWidget()) {
			m_workingSetManager->getManagerWidget()->save_to_default_session();
		}

		emit currentTfpChanged(tfpPath);
	}
	return valid;
}
