#include "pickinformation.h"

#include "DataSelectorDialog.h"
#include "folderdata.h"
#include "GeotimeProjectManagerWidget.h"
#include "nurbinformationmetadatawidget.h"
#include "pickinformationpanelwidget.h"
#include "propertyfiltersparser.h"
#include "wellpick.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QPainter>


PickInformation::PickInformation(const QString& name, const QString& path, const QColor& color,
		WorkingSetManager* manager, QObject* parent) : IInformation(parent), m_name(name), m_path(path),
		m_color(color), m_manager(manager) {
	searchLoadedData();
}

PickInformation::~PickInformation() {

}

// for actions
bool PickInformation::isDeletable() const {
	return false;
}

bool PickInformation::deleteStorage(QString* errorMsg) {
	if (errorMsg) {
		*errorMsg = tr("Picks cannot be deleted");
	}
	return false;
}

bool PickInformation::isSelectable() const {
	return true;
}

bool PickInformation::isSelected() const {
	if (isSelectable() && !m_loadedData.isNull()) {
		return true;
	}

	if (!isSelectable() || m_manager.isNull() || m_manager->getManagerWidget()==nullptr) {
		return false;
	}

	std::vector<QString> full = m_manager->getManagerWidget()->get_picks_fullnames();
	if (m_cacheDataIdx>=0 && m_cacheDataIdx<full.size() && m_path.compare(full[m_cacheDataIdx])==0) {
		return true;
	}

	auto it = std::find(full.begin(), full.end(), m_path);
	bool found = it!=full.end();

	if (found) {
		m_cacheDataIdx = std::distance(full.begin(), it);
	} else {
		m_cacheDataIdx = -1;
	}

	return found;
}

void PickInformation::toggleSelection(bool toggle) {
	if (!isSelectable() || toggle==isSelected()) {
		return;
	}

	// this cannot be threaded as is, it would need a synchronization point
	bool valid = true;
	if (toggle) {
		valid = selectPick(m_path, m_manager);
	} else {
		valid = unselectPick(m_path, m_manager);
	}

	if (valid) {
		if (!toggle) {
			m_loadedData = nullptr;
		} else {
			searchLoadedData();
		}

		if (m_manager && m_manager->getManagerWidget()) {
			m_manager->getManagerWidget()->save_to_default_session();
		}
	}
}

// comments
bool PickInformation::commentsEditable() const {
	return false;
}

QString PickInformation::comments() const {
	return "";
}

void PickInformation::setComments(const QString& txt) {
	// no comments
}

bool PickInformation::hasIcon() const {
	return true;
}

QIcon PickInformation::icon(int preferedSizeX, int preferedSizeY) const {
	QImage img(preferedSizeX, preferedSizeY, QImage::Format_RGB32);
	QPainter p(&img);
	p.fillRect(img.rect(), m_color);

	QPixmap pixmap = QPixmap::fromImage(img);
	return QIcon(pixmap);
}

// for sort and filtering
QString PickInformation::mainOwner() const {
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

QStringList PickInformation::owners() const {
	if (m_cacheOwners.size()==0) {
		searchFileCache();
	}
	return m_cacheOwners;
}

QDateTime PickInformation::mainCreationDate() const {
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

QList<QDateTime> PickInformation::creationDates() const {
	if (m_cacheCreationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheCreationDates;
}

QDateTime PickInformation::mainModificationDate() const {
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

QList<QDateTime> PickInformation::modificationDates() const {
	if (m_cacheModificationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheModificationDates;
}

QString PickInformation::name() const {
	return m_name;
}

information::StorageType PickInformation::storage() const {
	return information::StorageType::SISMAGE;
}

QColor PickInformation::color() const {
	return m_color;
}

void PickInformation::searchFileCache() const {
	if (m_cacheSearchDone) {
		return;
	}

	m_cacheCreationDates.clear();
	m_cacheModificationDates.clear();
	m_cacheOwners.clear();

	QFileInfo txtFile(m_path);
	if (txtFile.exists()) {
		QDateTime mainCreationTime = txtFile.birthTime();
		QDateTime mainModificationTime = txtFile.lastModified();
		QString mainOwner = txtFile.owner();
		m_cacheCreationDates.append(mainCreationTime);
		m_cacheModificationDates.append(mainModificationTime);
		m_cacheOwners.append(mainOwner);
	}
}

bool PickInformation::hasProperty(information::Property property) const {
	return property==information::Property::CREATION_DATE || property==information::Property::MODIFICATION_DATE ||
			property==information::Property::NAME || property==information::Property::OWNER ||
			property==information::Property::STORAGE_TYPE;
}

QVariant PickInformation::property(information::Property property) const {
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
	}

	return out;
}

bool PickInformation::isCompatible(information::Property prop, const QVariant& filter) const {
	if (!hasProperty(prop)) {
		return false;
	}

	QVariant value = this->property(prop);
	return PropertyFiltersParser::isCompatible(prop, filter, value);
}

// for gui representation, maybe should be done by another class
IInformationPanelWidget* PickInformation::buildInformationWidget(QWidget* parent) {
	return new PickInformationPanelWidget(this, parent);
}

QWidget* PickInformation::buildMetadataWidget(QWidget* parent) {
	return new NurbInformationMetadataWidget(this, parent);
}

QString PickInformation::folder() const {
	return QFileInfo(m_path).absolutePath();
}

QString PickInformation::mainPath() const {
	return m_path;
}

void PickInformation::searchLoadedData() {
	if (m_manager==nullptr || m_loadedData!=nullptr) {
		return;
	}

	QList<IData*> datas = m_manager->folders().markers->data();
	int i = 0;
	while (m_loadedData==nullptr && i<datas.size()) {
		Marker* marker = dynamic_cast<Marker*>(datas[i]);
		if (marker!=nullptr && m_name.compare(marker->name())==0) {
			m_loadedData = marker;
		}
		if (m_loadedData==nullptr) {
			i++;
		}
	}
	if (m_loadedData) {
		const QList<WellPick*>& picks = m_loadedData->wellPicks();
		for (long i=0; i<picks.size(); i++) {
			picks[i]->setAllDisplayPreference(true);
		}
	}
}

bool PickInformation::selectPick(const QString& pickPath, WorkingSetManager* manager) {
	if (manager==nullptr || manager->getManagerWidget()==nullptr) {
		return false;
	}
	bool valid = manager->getManagerWidget()->add_pick_fullpath_name(pickPath);

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

bool PickInformation::unselectPick(const QString& pickPath, WorkingSetManager* manager) {
	if (manager==nullptr || manager->getManagerWidget()==nullptr) {
		return false;
	}

	bool valid = manager->getManagerWidget()->remove_pick_fullpath_name(pickPath);

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
