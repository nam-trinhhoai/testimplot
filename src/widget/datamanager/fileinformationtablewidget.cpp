#include "fileinformationtablewidget.h"

#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>
#include <QDateTime>
#include <QLabel>

FileInformationTableWidget::FileInformationTableWidget(QWidget *parent, Qt::WindowFlags f) :
		QWidget(parent, f) {
	setLocale(QLocale::C); // this may need to be set for the whole application. Do not know why it is not (AS) 01/04/2021 (not a joke ;-) )

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	mainLayout->addWidget(new QLabel("WARNING : These data do not support yet deletion, to delete them : please use Sismage"));

	m_tableWidget = new QTableWidget(0, 3);
	mainLayout->addWidget(m_tableWidget);

	QStringList headers;
	headers << "Name" << "Owner" << "Birth Date";
	m_tableWidget->setHorizontalHeaderLabels(headers);
}

FileInformationTableWidget::~FileInformationTableWidget() {

}

void FileInformationTableWidget::clear() {
	m_tableWidget->clearContents();

	for (long index=m_tableWidget->rowCount()-1; index>=0; index--) {
		m_tableWidget->removeRow(index);
	}
}

void FileInformationTableWidget::addData(const QList<MonoFileBasedData>& newData) {
	long rowCount = m_tableWidget->rowCount();
	for (const MonoFileBasedData& fileData : newData) {
		if (fileData.isValid()) {
			m_tableWidget->insertRow(rowCount);
			QTableWidgetItem* nameItem = new QTableWidgetItem(fileData.name());
			nameItem->setToolTip(fileData.name());
			m_tableWidget->setItem(rowCount, 0, nameItem);
			QTableWidgetItem* ownerItem = new QTableWidgetItem(fileData.owner());
			m_tableWidget->setItem(rowCount, 1, ownerItem);
			QTableWidgetItem* birthItem = new QTableWidgetItem(locale().toString(fileData.birthDate(),  "dd.MM.yyyy hh:mm"));
			m_tableWidget->setItem(rowCount, 2, birthItem);
			rowCount++;
		}
	}

	m_data.append(newData);
}

MonoFileBasedData::MonoFileBasedData(const QString& name, const QString& path) {
	m_name = name;
	m_path = path;
	m_fileInfo = QFileInfo(m_path);
}

MonoFileBasedData::~MonoFileBasedData() {

}

QString MonoFileBasedData::name() const {
	return m_name;
}

QString MonoFileBasedData::path() const {
	return m_path;
}

bool MonoFileBasedData::isValid() const {
	return m_fileInfo.exists();
}

QString MonoFileBasedData::owner() const {
	return m_fileInfo.owner();
}

QDateTime MonoFileBasedData::birthDate() const {
	return m_fileInfo.birthTime();
}
