#include "folderdata.h"

#include <cmath>
#include <iostream>
#include "gdal.h"

#include "folderdatagraphicrepfactory.h"

FolderData::FolderData(WorkingSetManager *workingSet, const QString &name,
		QObject *parent) :
		IData(workingSet, parent) {
	m_name = name;
	m_uuid = QUuid::createUuid();

	m_repFactory = new FolderDataGraphicRepFactory(this);
}

IGraphicRepFactory* FolderData::graphicRepFactory() {
	return m_repFactory;
}

QUuid FolderData::dataID() const {
	return m_uuid;
}

FolderData::~FolderData() {
	for (IData* holdedData : m_content) {
		holdedData->deleteLater();
	}
}

// 17082021
bool FolderData::isDataContains(IData *data) {
	bool bIsPresent=false;

	for (IData* pData : m_content) {
		if (data == pData) {
			bIsPresent = true;
			break;
		}
	}

	return bIsPresent;
}

void FolderData::addData(IData *data) {
	if(isDataContains(data)== false){
		m_content.push_back(data);
		emit dataAdded(data);
	}
}

// 17082021
void FolderData::deleteData(IData *data) {
	m_content.removeOne(data);
	emit dataRemoved(data);
}

void FolderData::removeData(IData *data) {
	if(isDataContains(data)== true	){
		m_content.removeOne(data);
		emit dataRemoved(data);
		data->deleteLater();// to keep mechanism from WorkingSetManager
	}
}

QList<IData*> FolderData::data() {
	return m_content;
}

