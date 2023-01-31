#ifndef SRC_WIDGET_DATAMANAGER_TRASHTABLEWIDGET_H_
#define SRC_WIDGET_DATAMANAGER_TRASHTABLEWIDGET_H_

#include <QWidget>
#include <QMap>

#include "leafcontaineraggregator.h"

class QTableWidget;
class DeletableLeaf;

class TrashTableWidget : public QWidget {
	Q_OBJECT
public:
	TrashTableWidget(LeafContainerAggregator* container, QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~TrashTableWidget();

	LeafContainerAggregator* container();

signals:
	void requestDataRestoration(LeafContainerAggregator::AggregatorKey leafKey);
	void requestDataDeletion(LeafContainerAggregator::AggregatorKey leafKey);

private slots:
	void customContextMenuFromTable(const QPoint& pos);
	void clear();
	void addData(const QList<LeafContainerAggregator::AggregatorKey>& newData);
	void removeData(LeafContainerAggregator::AggregatorKey id, DeletableLeaf leaf);

private:
	QTableWidget* m_tableWidget;
	LeafContainerAggregator* m_container;
};

#endif
