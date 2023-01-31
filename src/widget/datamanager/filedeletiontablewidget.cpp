#include "filedeletiontablewidget.h"
#include "sismagedbmanager.h"
#include "SeismicManager.h"
#include "deleteableleaf.h"
#include "leafcontainer.h"

#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>
#include <QDateTime>
#include <QLabel>
#include <QMessageBox>
#include <QDir>
#include <QFileInfo>
#include <QFile>
#include <QDateTime>
#include <QStringList>
#include <QMenu>
#include <QAction>
#include <QDebug>


FileDeletionTableWidget::FileDeletionTableWidget(LeafContainer* container, QWidget *parent, Qt::WindowFlags f) :
		QWidget(parent, f), m_container(container) {
	setLocale(QLocale::C); // this may need to be set for the whole application. Do not know why it is not (AS) 01/04/2021 (not a joke ;-) )

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	m_tableWidget = new QTableWidget(0, 3);
	m_tableWidget->setContextMenuPolicy(Qt::CustomContextMenu);
	m_tableWidget->setDragDropMode(QAbstractItemView::DragOnly);
	mainLayout->addWidget(m_tableWidget);

	QStringList headers;
	headers << "Name" << "Owner" << "Birth Date";
	m_tableWidget->setHorizontalHeaderLabels(headers);

	connect(m_tableWidget, &QTableWidget::customContextMenuRequested, this, &FileDeletionTableWidget::customContextMenuFromTable);

	connect(m_container, &LeafContainer::dataAdded, this, &FileDeletionTableWidget::addData);
	connect(m_container, &LeafContainer::dataRemoved, this, &FileDeletionTableWidget::removeData);
	connect(m_container, &LeafContainer::dataCleared, this, &FileDeletionTableWidget::clear);
}

FileDeletionTableWidget::~FileDeletionTableWidget() {

}

LeafContainer* FileDeletionTableWidget::container() {
	return m_container;
}

void FileDeletionTableWidget::clear() {
	m_tableWidget->clearContents();

	for (long index=m_tableWidget->rowCount()-1; index>=0; index--) {
		m_tableWidget->removeRow(index);
	}
}

void FileDeletionTableWidget::addData(const QList<std::size_t>& newData) {
	long rowCount = m_tableWidget->rowCount();
	for (const std::size_t& fileDataId : newData) {
		if (m_container->containId(fileDataId)) {
			const DeletableLeaf& fileData = m_container->at(fileDataId);
			if (fileData.isValid()) {
				m_tableWidget->insertRow(rowCount);
				QTableWidgetItem* nameItem = new QTableWidgetItem(fileData.name());
				nameItem->setData(Qt::UserRole, QVariant((qulonglong) fileDataId));
				nameItem->setToolTip(fileData.name());
				m_tableWidget->setItem(rowCount, 0, nameItem);

				QTableWidgetItem* ownerItem = new QTableWidgetItem(fileData.owner());
				ownerItem->setData(Qt::UserRole, QVariant((qulonglong) fileDataId));
				m_tableWidget->setItem(rowCount, 1, ownerItem);

				QTableWidgetItem* birthItem = new QTableWidgetItem(locale().toString(fileData.birthDate(),  "dd.MM.yyyy hh:mm"));
				birthItem->setData(Qt::UserRole, QVariant((qulonglong) fileDataId));
				m_tableWidget->setItem(rowCount, 2, birthItem);
				rowCount++;
			}
		}
	}
}

void FileDeletionTableWidget::removeData(std::size_t id, DeletableLeaf leaf) {
	long index = 0;
	bool notFound = true;
	while (index<m_tableWidget->rowCount() && notFound) {
		QTableWidgetItem* item = m_tableWidget->item(index, 0);
		QVariant variant = item->data(Qt::UserRole);
		std::size_t currentId = variant.toULongLong(&notFound);
		notFound = (!notFound) || currentId!=id;
		if (notFound) {
			index++;
		}
	}
	if (!notFound) {
		m_tableWidget->removeRow(index);
	}
}

void FileDeletionTableWidget::customContextMenuFromTable(const QPoint& pos) {
	QTableWidgetItem* item = m_tableWidget->itemAt(pos);
	if (item) {
		qDebug() << item->text();
		QVariant var = item->data(Qt::UserRole);
		bool ok = false;
		std::size_t dataKey;
		if (!var.isNull() && var.isValid()) {
			dataKey = var.toULongLong(&ok);
		}

		if (ok) {
			QMenu menu;
			QAction* actionDelete = new QAction("Move to trash");
			menu.addAction(actionDelete);
			connect(actionDelete, &QAction::triggered, [this, dataKey, item]() {
				emit requestDataDeletion(dataKey);
//				bool success = m_container->at(dataKey).moveToTrash();
//				if (success && m_container->containId(dataKey)) {// security to avoid multiple deletion of row
//					m_tableWidget->removeRow(item->row());
//				} // else message provided by the delete function already
			});
			menu.exec(m_tableWidget->mapToGlobal(pos));
		}
	}
}

