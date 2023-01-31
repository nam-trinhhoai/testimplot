#ifndef SRC_WIDGET_DATAMANAGER_FILEDELETIONTABLEWIDGET_H_
#define SRC_WIDGET_DATAMANAGER_FILEDELETIONTABLEWIDGET_H_

#include <QWidget>
#include <QMap>

class QTableWidget;
class DeletableLeaf;
class LeafContainer;

class FileDeletionTableWidget : public QWidget {
	Q_OBJECT
public:
	FileDeletionTableWidget(LeafContainer* container, QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~FileDeletionTableWidget();

	LeafContainer* container();

signals:
	void requestDataDeletion(std::size_t leafKey);

private slots:
	void customContextMenuFromTable(const QPoint& pos);
	void clear();
	void addData(const QList<std::size_t>& newData);
	void removeData(std::size_t id, DeletableLeaf leaf);

private:
	QTableWidget* m_tableWidget;
	LeafContainer* m_container;
};

#endif
