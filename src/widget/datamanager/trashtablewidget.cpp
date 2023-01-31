#include "trashtablewidget.h"
#include "deleteableleaf.h"

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

TrashTableWidget::TrashTableWidget(LeafContainerAggregator* container, QWidget *parent, Qt::WindowFlags f) :
		QWidget(parent, f), m_container(container) {
	setLocale(QLocale::C); // this may need to be set for the whole application. Do not know why it is not (AS) 01/04/2021 (not a joke ;-) )

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	m_tableWidget = new QTableWidget(0, 3);
	m_tableWidget->setContextMenuPolicy(Qt::CustomContextMenu);
	mainLayout->addWidget(m_tableWidget);

	QStringList headers;
	headers << "Name" << "Owner" << "Birth Date";
	m_tableWidget->setHorizontalHeaderLabels(headers);

	connect(m_tableWidget, &QTableWidget::customContextMenuRequested, this, &TrashTableWidget::customContextMenuFromTable);

	connect(m_container, &LeafContainerAggregator::dataAdded, this, &TrashTableWidget::addData);
	connect(m_container, &LeafContainerAggregator::dataRemoved, this, &TrashTableWidget::removeData);
	connect(m_container, &LeafContainerAggregator::dataCleared, this, &TrashTableWidget::clear);
}

TrashTableWidget::~TrashTableWidget() {

}

LeafContainerAggregator* TrashTableWidget::container() {
	return m_container;
}

void TrashTableWidget::clear() {
	m_tableWidget->clearContents();

	for (long index=m_tableWidget->rowCount()-1; index>=0; index--) {
		m_tableWidget->removeRow(index);
	}
}

void TrashTableWidget::addData(const QList<LeafContainerAggregator::AggregatorKey>& newData) {
	long rowCount = m_tableWidget->rowCount();
	for (const LeafContainerAggregator::AggregatorKey& fileDataId : newData) {
		if (m_container->containId(fileDataId)) {
			const DeletableLeaf& fileData = m_container->at(fileDataId);
			if (fileData.isValid()) {
				QVariant var = QVariant::fromValue(fileDataId);

				m_tableWidget->insertRow(rowCount);
				QTableWidgetItem* nameItem = new QTableWidgetItem(fileData.name());
				nameItem->setData(Qt::UserRole, var);
				nameItem->setToolTip(fileData.name());
				m_tableWidget->setItem(rowCount, 0, nameItem);

				QTableWidgetItem* ownerItem = new QTableWidgetItem(fileData.owner());
				ownerItem->setData(Qt::UserRole, var);
				m_tableWidget->setItem(rowCount, 1, ownerItem);

				QTableWidgetItem* birthItem = new QTableWidgetItem(locale().toString(fileData.birthDate(),  "dd.MM.yyyy hh:mm"));
				birthItem->setData(Qt::UserRole, var);
				m_tableWidget->setItem(rowCount, 2, birthItem);
				rowCount++;
			}
		}
	}
}

void TrashTableWidget::removeData(LeafContainerAggregator::AggregatorKey id, DeletableLeaf leaf) {
	long index = 0;
	bool notFound = true;
	while (index<m_tableWidget->rowCount() && notFound) {
		QTableWidgetItem* item = m_tableWidget->item(index, 0);
		QVariant variant = item->data(Qt::UserRole);
		LeafContainerAggregator::AggregatorKey currentId;
		notFound = variant.canConvert<LeafContainerAggregator::AggregatorKey>();
		if (notFound) {
			currentId = variant.value<LeafContainerAggregator::AggregatorKey>();
		}
		notFound = (!notFound) || currentId.containerId!=id.containerId || currentId.leafId!=id.leafId;
		if (notFound) {
			index++;
		}
	}
	if (!notFound) {
		m_tableWidget->removeRow(index);
	}
}

void TrashTableWidget::customContextMenuFromTable(const QPoint& pos) {
	QTableWidgetItem* item = m_tableWidget->itemAt(pos);
	if (item) {
		qDebug() << item->text();
		QVariant var = item->data(Qt::UserRole);
		bool ok = false;
		LeafContainerAggregator::AggregatorKey dataKey;
		if (!var.isNull() && var.isValid() && var.canConvert<LeafContainerAggregator::AggregatorKey>()) {
			dataKey = var.value<LeafContainerAggregator::AggregatorKey>();
			ok = true;
		}

		if (ok) {
			QMenu menu;
			QAction* actionRestore = new QAction("Restore");
			menu.addAction(actionRestore);
			connect(actionRestore, &QAction::triggered, [this, dataKey, item]() {
				emit requestDataRestoration(dataKey);
			});
			QAction* actionDelete = new QAction("Delete");
			menu.addAction(actionDelete);
			connect(actionDelete, &QAction::triggered, [this, dataKey, item]() {
				emit requestDataDeletion(dataKey);
			});
			menu.exec(m_tableWidget->mapToGlobal(pos));
		}
	}
}
