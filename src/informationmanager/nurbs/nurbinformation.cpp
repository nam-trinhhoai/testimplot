#include "nurbinformation.h"

#include "folderdata.h"
#include "GeotimeProjectManagerWidget.h"
#include "nurbinformationmetadatawidget.h"
#include "nurbinformationpanelwidget.h"
#include "nurbswidget.h"
#include "propertyfiltersparser.h"

#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QPainter>
#include <QPixmap>
#include <QTextStream>
#include <QWidget>

NurbInformation::NurbInformation(const QString& name, const QString& fullPath, WorkingSetManager* manager,
		QObject* parent) : IInformation(parent), m_name(name), m_fullPath(fullPath), m_manager(manager) {

	// read txt file
	bool valid;
	Manager::NurbsParams params = Manager::read(m_fullPath, &valid);
	m_color = params.color;
	m_precision = params.precision;
	m_nbCurves = params.generatrices.size();
}

NurbInformation::~NurbInformation() {

}

// for actions
bool NurbInformation::isDeletable() const {
	return true;
}

bool NurbInformation::deleteStorage(QString* errorMsg) {
	if (isSelected()) {
		toggleSelection(false);
	}

	QFileInfo txtFileInfo(m_fullPath);
	QDir parentDir = txtFileInfo.dir();
	QFileInfo parentInfo(parentDir.absolutePath());
	if (!parentInfo.exists() || !parentInfo.isWritable()) {
		if (errorMsg!=nullptr) {
			*errorMsg = "Permission issue on : " + parentInfo.absoluteFilePath();
		}
		return false;
	}

	QString objPath = parentDir.absoluteFilePath(txtFileInfo.baseName()+".obj");
	QFileInfo objFileInfo(objPath);
	bool txtSuccess = true;
	if (txtFileInfo.exists()) {
		txtSuccess = QFile::remove(m_fullPath);
	}
	bool objSuccess = true;
	if (objFileInfo.exists()) {
		objSuccess = QFile::remove(objPath);
	}

	if (errorMsg!=nullptr) {
		*errorMsg = "";
	}

	if (!txtSuccess) {
		qDebug() << "Failed to delete file : " << m_fullPath;
		if (errorMsg!=nullptr) {
			*errorMsg = "Failed to delete file : " + m_fullPath;
		}
	}
	if (!objSuccess) {
		qDebug() << "Failed to delete file : " << objPath;
		if (errorMsg!=nullptr) {
			if ((*errorMsg).size()>0) {
				*errorMsg += "\n";
			}
			*errorMsg += "Failed to delete file : " + m_fullPath;
		}
	}

	return txtSuccess && objSuccess;
}

bool NurbInformation::isSelectable() const {
	return !m_manager.isNull();
}

bool NurbInformation::isSelected() const {
	if (!isSelectable()) {
		return false;
	}

	QList<IData*> datas = m_manager->folders().nurbs->data();
	long i=0;
	bool notFound = true;
	while (notFound && i<datas.size()) {
		notFound = datas[i]->name().compare(m_name)!=0;
		i++;
	}
	return !notFound;
}

void NurbInformation::toggleSelection(bool toggle) {
	if (!isSelectable()) {
		return;
	}

	bool currentState = isSelected();
	if (currentState==toggle) {
		return;
	}
	QString nurbsName = m_name;
	nurbsName.replace(".txt", "");
	if (toggle) {
		nurbsName += ".txt";
		// select data
		NurbsWidget::addNurbs(m_fullPath, nurbsName);

		if (m_manager && m_manager->getManagerWidget()) {
			m_manager->getManagerWidget()->set_nurbs_filter("");

			m_manager->getManagerWidget()->clear_nurbs_gui_selection();
			bool valid = m_manager->getManagerWidget()->select_nurbs_tinyname(m_name+".txt"); // tinyname to match end by .txt

			m_manager->getManagerWidget()->trt_nurbs_basket_add();

			m_manager->getManagerWidget()->save_to_default_session();
		}
	} else {
		// unselect data
		NurbsWidget::removeNurbs(m_fullPath, nurbsName);

		if (m_manager && m_manager->getManagerWidget()) {
			m_manager->getManagerWidget()->set_nurbs_filter("");

			m_manager->getManagerWidget()->clear_nurbs_basket_gui_selection();
			bool valid = m_manager->getManagerWidget()->select_nurbs_basket_tinyname(m_name+".txt");

			m_manager->getManagerWidget()->trt_nurbs_basket_sub();

			m_manager->getManagerWidget()->save_to_default_session();
		}
	}
}

// comments
bool NurbInformation::commentsEditable() const {
	return false;
}

QString NurbInformation::comments() const {
	return "";
}

void NurbInformation::setComments(const QString& txt) {
	// no comments
}

bool NurbInformation::hasIcon() const {
	return true;
}

QIcon NurbInformation::icon(int preferedSizeX, int preferedSizeY) const {
	QImage img(preferedSizeX, preferedSizeY, QImage::Format_RGB32);
	QPainter p(&img);
	p.fillRect(img.rect(), m_color);

	QPixmap pixmap = QPixmap::fromImage(img);
	return QIcon(pixmap);
}

// for sort and filtering
QString NurbInformation::mainOwner() const {
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

QStringList NurbInformation::owners() const {
	if (m_cacheOwners.size()==0) {
		searchFileCache();
	}
	return m_cacheOwners;
}

QDateTime NurbInformation::mainCreationDate() const {
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

QList<QDateTime> NurbInformation::creationDates() const {
	if (m_cacheCreationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheCreationDates;
}

QDateTime NurbInformation::mainModificationDate() const {
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

QList<QDateTime> NurbInformation::modificationDates() const {
	if (m_cacheModificationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheModificationDates;
}

QString NurbInformation::name() const {
	return m_name;
}

information::StorageType NurbInformation::storage() const {
	return information::StorageType::NEXTVISION;
}

void NurbInformation::searchFileCache() const {
	if (m_cacheSearchDone) {
		return;
	}

	m_cacheCreationDates.clear();
	m_cacheModificationDates.clear();
	m_cacheOwners.clear();

	QFileInfo txtFile(m_fullPath);
	if (txtFile.exists()) {
		QDateTime mainCreationTime = txtFile.birthTime();
		QDateTime mainModificationTime = txtFile.lastModified();
		QString mainOwner = txtFile.owner();
		m_cacheCreationDates.append(mainCreationTime);
		m_cacheModificationDates.append(mainModificationTime);
		m_cacheOwners.append(mainOwner);
	}

	QFileInfo objFile(txtFile.dir().absoluteFilePath(txtFile.baseName()+".obj"));
	if (objFile.exists()) {
		QDateTime creationTime = objFile.birthTime();
		QDateTime modificationTime = objFile.lastModified();
		QString owner = objFile.owner();
		if (!m_cacheCreationDates.contains(creationTime)) {
			m_cacheCreationDates.append(creationTime);
		}
		if (!m_cacheModificationDates.contains(modificationTime)) {
			m_cacheModificationDates.append(modificationTime);
		}
		if (!m_cacheOwners.contains(owner)) {
			m_cacheOwners.append(owner);
		}
	}
}

bool NurbInformation::hasProperty(information::Property property) const {
	return property==information::Property::CREATION_DATE || property==information::Property::MODIFICATION_DATE ||
			property==information::Property::NAME || property==information::Property::OWNER ||
			property==information::Property::STORAGE_TYPE;
}

QVariant NurbInformation::property(information::Property property) const {
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

bool NurbInformation::isCompatible(information::Property prop, const QVariant& filter) const {
	if (!hasProperty(prop)) {
		return false;
	}

	QVariant value = this->property(prop);
	return PropertyFiltersParser::isCompatible(prop, filter, value);
}

QColor NurbInformation::color() const {
	return m_color;
}

void NurbInformation::setColor(QColor color) {
	//if (m_color!=color) {
		m_color = color;
		if (isSelected()) {
			NurbsWidget::setColor(m_name, m_color);
		}
		qDebug()<<" setColor ==> "<<m_fullPath;
		QFileInfo fileinfo(m_fullPath);
		QDir dir(fileinfo.absoluteDir());
			if(!dir.exists())dir.mkdir(dir.path());
		NurbsWidget::saveColor(m_fullPath, m_color);
		emit colorChanged(m_color);
		emit iconChanged();
	//}
}

int NurbInformation::nbCurves() const {
	return m_nbCurves;
}

int NurbInformation::precision() const {
	return m_precision;
}

// for gui representation, maybe should be done by another class
IInformationPanelWidget* NurbInformation::buildInformationWidget(QWidget* parent) {
	return new NurbInformationPanelWidget(this, parent);
}

QWidget* NurbInformation::buildMetadataWidget(QWidget* parent) {
	return new NurbInformationMetadataWidget(this, parent);
}

QString NurbInformation::folder() const {
	return QFileInfo(m_fullPath).absolutePath();
}

QString NurbInformation::mainPath() const {
	return m_fullPath;
}
