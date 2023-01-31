#ifndef MANAGERWIDGET_H
#define MANAGERWIDGET_H

#include "iinformationaggregator.h"
#include "propertyfiltersparser.h"

#include <QModelIndex>
#include <QWidget>

#include <unordered_map>

class GenericInfoWidget;
class IInformation;
class IInformationPanelWidget;

class QComboBox;
class QPushButton;
class QTextEdit;
class QToolButton;
class QTreeWidget;
class QTreeWidgetItem;
class QVBoxLayout;
class QScrollArea;
class QLineEdit;

class TreeWidget :public QWidget
{
	Q_OBJECT
public:
	struct InformationConnections {
		QMetaObject::Connection iconUpdateConn;
		QMetaObject::Connection deleteConn;
	};

	TreeWidget(IInformationAggregator* aggregator, QWidget* parent=nullptr);
	~TreeWidget();
	void setSelectButtonVisible(bool val);
	QTreeWidget* getTreeWidget() { return m_treewidget; }
	void setFilterString(QString data);

signals:
	void currentInformationChanged(IInformation* information);

private slots:
	void addInformationToTree(IInformation* information);
	void changeSort(int index);
	void currentItemChanged(const QTreeWidgetItem* current, const QTreeWidgetItem* previous);
	void dataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QList<int>& roles);
	void managerDeleted();
	void informationDeleted(QObject* deletedObj);
	void parseFilter(const QString& filter);
	void populateTree();
	void removeInformationFromTree(IInformation* information);
	void unselectAll();
	void selectAll();
	void toggleSortOrder();

private:
	void cleanInformationConnections();
	QTreeWidgetItem* createItem(IInformation* information);
	bool filterInformation(const IInformation* information);
	void setTreeItemIcon(IInformation* info, QTreeWidgetItem* item);
	void updateFromInformationIcon(IInformation* info);

	QComboBox* m_sortPropertyComboBox;
	QToolButton* m_sortButton;
	QTreeWidget* m_treewidget;
	QLineEdit* m_filter = nullptr;

	IInformationAggregator* m_aggregator;
	QString m_filterStr;
	PropertyFiltersParser m_parser;
	const QString m_separator = ";";

	bool m_sortActivate = false;
	bool m_sortOrderTopToBottom = true;
	information::Property m_sortProperty;
	QPushButton * m_selectAllButton = nullptr;
	QPushButton * m_unselectAllButton = nullptr;

	std::unordered_map<IInformation*, InformationConnections> m_informationConnections;
};


class InformationWidget :public QWidget
{
	Q_OBJECT
public:
	InformationWidget(QWidget* parent=nullptr);
	~InformationWidget();

	IInformation* currentInformation();

public slots:
	void onSave();
	void setCurrentInformation(IInformation* information);

signals:
	void currentInformationChanged(IInformation* information);

private:
	QComboBox* m_comboStorage;
	QTextEdit* m_edit;
	GenericInfoWidget* m_infoWidget;

	IInformation* m_currentInformation = nullptr;
};


class GenericInfoWidget :public QWidget
{
	Q_OBJECT
public:
	GenericInfoWidget(QWidget* parent=nullptr);
	~GenericInfoWidget();

	IInformation* currentInformation();

public slots:
	void informationChanged(IInformationPanelWidget*);
	void metadataChanged(QWidget*);
	void onSave();
	void setCurrentInformation(IInformation* information);

signals:
	void currentInformationChanged(IInformation* information);

private :
	QVBoxLayout* m_infoGrpLayout;
	QVBoxLayout* m_dataGrpLayout;
	IInformationPanelWidget* m_lastWidgetInfo = nullptr;
	QWidget* m_lastWidgetData = nullptr;
	QScrollArea *m_generalInfoScrollArea;
	QScrollArea *m_metaDataScrollArea;

	IInformation* m_currentInformation = nullptr;
};



class ManagerWidget : public QWidget{
	Q_OBJECT
public:
	// object take ownership of the aggregator
	ManagerWidget(IInformationAggregator* aggregator, QWidget* parent=nullptr);
	~ManagerWidget();

	IInformation* currentInformation();

public slots:
	void onNew();
	void onSave();
	void onDelete();
	void onClose();
	void setCurrentInformation(IInformation* information);

signals:
	void currentInformationChanged(IInformation* information);

private:
	QPushButton* m_buttonNew;
	QPushButton* m_buttonDelete;

	InformationWidget* m_infoWidget;
	TreeWidget* m_treeWidget;

	IInformationAggregator* m_aggregator;
	IInformation* m_currentInformation = nullptr;
};

#endif
