
#ifndef __MANAGERFILESELECTORDIALOG__
#define __MANAGERFILESELECTORDIALOG__

#include "iinformationaggregator.h"
#include "propertyfiltersparser.h"

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

class InformationWidget;
class TreeWidget;


#include <QModelIndex>
#include <QWidget>
#include <QDialog>

#include <utility>      // std::pair, std::make_pair
#include <vector>

#include <unordered_map>

class ManagerFileSelectorWidget : public QDialog { // public QWidget{
	Q_OBJECT
public:
	// object take ownership of the aggregator
	ManagerFileSelectorWidget(IInformationAggregator* aggregator, QWidget* parent=nullptr);
	~ManagerFileSelectorWidget();

	IInformation* currentInformation();
	std::pair<std::vector<QString>, std::vector<QString>> getSelectedNames() const;
	std::vector<int> getSelectedIndexes() const;


public slots:
	// void onNew();
	// void onSave();
	// void onDelete();
	// void onClose();
	// void setCurrentInformation(IInformation* information);
	void setFilterString(QString data);

signals:
	void currentInformationChanged(IInformation* information);

private:
	QPushButton* m_buttonNew = nullptr;
	QPushButton* m_buttonDelete = nullptr;

	InformationWidget* m_infoWidget = nullptr;
	TreeWidget* m_treeWidget = nullptr;

	IInformationAggregator* m_aggregator = nullptr;
	IInformation* m_currentInformation = nullptr;
	std::pair<std::vector<QString>, std::vector<QString>> seismicGetSelectedNames() const;
	std::pair<std::vector<QString>, std::vector<QString>> nextvisionHorizonGetSelectedNames() const;
	std::pair<std::vector<QString>, std::vector<QString>> logGetSelectedNames() const;
	std::vector<int> logGetSelectedIndexes() const;


};




#endif
