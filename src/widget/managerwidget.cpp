#include "managerwidget.h"
#include "iinformation.h"
#include "iinformationpanelwidget.h"
#include "propertyfiltersparser.h"

#include <QSplitter>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPushButton>
#include <QGroupBox>
#include <QTextEdit>
#include <QLineEdit>
#include <QComboBox>
#include <QMessageBox>
#include <QTabWidget>
#include <QToolButton>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QScrollArea>

#include <algorithm>

//class TreeWidget
TreeWidget::TreeWidget(IInformationAggregator* aggregator, QWidget* parent) : QWidget(parent)
{
	m_aggregator = aggregator;

	QVBoxLayout* mainLayout = new QVBoxLayout;

	QHBoxLayout* filterLayout = new QHBoxLayout;
	filterLayout->addWidget(new QLabel("Filter : "));

	m_filter = new QLineEdit();
	m_filter->setMaximumHeight(50);
	QString helpTop = "This is an advanced filter, the key separator is \""+m_separator+"\"\nHere are the keys definition : \n";
	QString filterHelp = PropertyFiltersParser::constructPropertiesParsingHelp(m_aggregator->availableProperties());
	m_filter->setToolTip(helpTop+filterHelp);
	filterLayout->addWidget(m_filter);

	QHBoxLayout* sortLayout = new QHBoxLayout;
	sortLayout->addWidget(new QLabel("Sort by "));
	m_sortPropertyComboBox = new QComboBox;
	m_sortPropertyComboBox->addItem("");
	std::list<information::Property> properties = m_aggregator->availableProperties();
	for (auto it=properties.begin(); it!=properties.end(); it++)
	{
		m_sortPropertyComboBox->addItem(PropertyFiltersParser::propertyToString(*it), QVariant::fromValue(*it));
	}
	sortLayout->addWidget(m_sortPropertyComboBox);
	m_sortButton = new QToolButton;
	m_sortButton->setArrowType(Qt::DownArrow); // top to bottom
	sortLayout->addWidget(m_sortButton);

	// this could be a QListWidget
	m_treewidget = new QTreeWidget();
	m_treewidget->setHeaderHidden(true);
	m_treewidget->setIconSize(QSize(16, 16));
	// https://stackoverflow.com/questions/6625188/qtreeview-horizontal-scrollbar-problems
	m_treewidget->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
	m_treewidget->header()->setStretchLastSection(false);

	mainLayout->addLayout(filterLayout);
	mainLayout->addLayout(sortLayout);
	mainLayout->addWidget(m_treewidget);

	QHBoxLayout* selectLayout = new QHBoxLayout;
	mainLayout->addLayout(selectLayout);
	m_selectAllButton = new QPushButton("Select all");
	selectLayout->addWidget(m_selectAllButton);
	m_unselectAllButton = new QPushButton("Unselect all");
	selectLayout->addWidget(m_unselectAllButton);

	setLayout(mainLayout);

	connect(m_filter, &QLineEdit::textChanged, this, &TreeWidget::parseFilter);
	connect(m_unselectAllButton, &QPushButton::clicked, this, &TreeWidget::unselectAll);
	connect(m_selectAllButton, &QPushButton::clicked, this, &TreeWidget::selectAll);
	connect(m_sortButton, &QToolButton::clicked, this, &TreeWidget::toggleSortOrder);
	connect(m_sortPropertyComboBox, &QComboBox::currentIndexChanged, this, &TreeWidget::changeSort);
	connect(m_treewidget, &QTreeWidget::currentItemChanged, this, &TreeWidget::currentItemChanged);
	connect(m_treewidget->model(), &QAbstractItemModel::dataChanged, this, &TreeWidget::dataChanged);

	connect(m_aggregator, &IInformationAggregator::destroyed, this, &TreeWidget::managerDeleted);
	connect(m_aggregator, &IInformationAggregator::informationAdded, this, &TreeWidget::addInformationToTree);
	connect(m_aggregator, &IInformationAggregator::informationRemoved, this, &TreeWidget::removeInformationFromTree);

	populateTree();
}

TreeWidget::~TreeWidget()
{
	cleanInformationConnections();
}

void TreeWidget::setFilterString(QString data)
{
	m_filter->setText(data);
	parseFilter(data);
}

void TreeWidget::setSelectButtonVisible(bool val)
{
	if ( m_selectAllButton ) m_selectAllButton->setVisible(val);
	if ( m_unselectAllButton ) m_unselectAllButton->setVisible(val);
}

void TreeWidget::addInformationToTree(IInformation* information)
{
	if (m_aggregator==nullptr) {
		return;
	}

	bool validWithFilter = filterInformation(information);

	int insertionIndex = m_treewidget->topLevelItemCount();
	if (validWithFilter && m_sortActivate)
	{
		std::function<bool(const IInformation*, const IInformation*)> pred = InformationPredicate::createComparator(m_sortProperty);

		insertionIndex = 0;
		bool found = false;
		while (!found && insertionIndex<m_treewidget->topLevelItemCount())
		{
			QTreeWidgetItem* item = m_treewidget->topLevelItem(insertionIndex);
			QVariant var = item->data(0, Qt::UserRole);
			bool valid = var.isValid() && var.canConvert<IInformation*>();
			IInformation* itemInfo = nullptr;
			if (valid)
			{
				// need an item with valid information
				itemInfo = var.value<IInformation*>();
				valid = itemInfo!=nullptr;
			}
			if (valid && m_sortOrderTopToBottom)
			{
				// is search info is lower than current top level info, we found the insert location since we scan from index 0 (and scanned the lowest value)
				found = pred(information, itemInfo);
			}
			else if (valid && !m_sortOrderTopToBottom)
			{
				// is search info is greater than current top level info, we found the insert location since we scan from index 0 (and scanned the greatest value)
				found = pred(itemInfo, information);
			}
			if (!found)
			{
				insertionIndex++;
			}
		}
	}
	else if (validWithFilter)
	{
		std::shared_ptr<IInformationIterator> it = m_aggregator->begin();
		insertionIndex = 0;
		while (it->isValid() && information!=it->cobject())
		{
			insertionIndex++;
			it->next();
		}
	}
	if (validWithFilter)
	{
		// safety check in case of strange index
		if (insertionIndex<0)
		{
			insertionIndex = 0;
		}
		else if (insertionIndex>m_treewidget->topLevelItemCount())
		{
			insertionIndex = m_treewidget->topLevelItemCount();
		}

		QTreeWidgetItem* item = createItem(information);
		m_treewidget->insertTopLevelItem(insertionIndex, item);
	}
}

void TreeWidget::changeSort(int index)
{
	if (index<0 || index>=m_sortPropertyComboBox->count())
	{
		return;
	}

	QVariant var = m_sortPropertyComboBox->itemData(index);
	bool isValid = var.isValid() && var.canConvert<information::Property>();
	bool sortChanged = false;
	if (isValid)
	{
		information::Property newSortProperty = var.value<information::Property>();
		sortChanged = newSortProperty!=m_sortProperty;
		m_sortProperty = newSortProperty;
	}
	sortChanged = sortChanged || m_sortActivate!=isValid;
	m_sortActivate = isValid;

	if (sortChanged)
	{
		populateTree();
	}
}

void TreeWidget::cleanInformationConnections()
{
	for (auto it=m_informationConnections.begin(); it!=m_informationConnections.end(); it++)
	{
		disconnect(it->second.iconUpdateConn);
		disconnect(it->second.deleteConn);
	}
}

QTreeWidgetItem* TreeWidget::createItem(IInformation* information)
{
	if (information==nullptr)
	{
		return nullptr;
	}

	QTreeWidgetItem *item = new QTreeWidgetItem;
	item->setText(0, information->name());
	item->setToolTip(0, information->name());
	QVariant var;
	var.setValue(information);
	item->setData(0, Qt::UserRole, var);
	item->setFlags(item->flags() & ~Qt::ItemIsEditable);
	if (information->hasIcon())
	{
		item->setIcon(0, information->icon(16, 16));
	}
	if (information->isSelectable())
	{
		item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
		item->setData(0, Qt::CheckStateRole, (information->isSelected()) ? Qt::Checked : Qt::Unchecked);
	}
	else
	{
		item->setFlags(item->flags() & ~Qt::ItemIsUserCheckable);
	}

	InformationConnections conns;
	conns.iconUpdateConn = connect(information, &IInformation::iconChanged, this, [this, information]()
	{
		updateFromInformationIcon(information);
	});
	conns.deleteConn = connect(information, &IInformation::destroyed, this, &TreeWidget::informationDeleted);
	m_informationConnections[information] = conns;

	return item;
}

void TreeWidget::currentItemChanged(const QTreeWidgetItem* current, const QTreeWidgetItem* previous)
{
	IInformation* information = nullptr;

	if (current)
	{
		QVariant var = current->data(0, Qt::UserRole);
		if(var.canConvert<IInformation*>())
		{
			information = qvariant_cast<IInformation*>(var);
		}
	}

	emit currentInformationChanged(information);
}

void TreeWidget::dataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QList<int>& roles)
{
	if ((roles.size()!=0 && !roles.contains(Qt::CheckStateRole)) || topLeft.column()>0 || bottomRight.column()<0)
	{
		return;
	}

	for (int i=topLeft.row(); i<=bottomRight.row(); i++)
	{
		QModelIndex modelIndex = topLeft.sibling(i, 0);
		QVariant var = modelIndex.data(Qt::UserRole);
		IInformation* information = nullptr;
		if(var.canConvert<IInformation*>())
		{
			information = qvariant_cast<IInformation*>(var);
		}
		bool checked = topLeft.data(Qt::CheckStateRole).toBool();
		if (information && information->isSelectable() && information->isSelected()!=checked)
		{
			information->toggleSelection(checked);
		}
	}
}

void TreeWidget::managerDeleted() {
	m_aggregator = nullptr;
	populateTree();
}

bool TreeWidget::filterInformation(const IInformation* information)
{
	bool validForFilter = false;
	if (m_filterStr.isNull() || m_filterStr.isEmpty())
	{
		validForFilter = true;
	}
	else
	{
		validForFilter = m_parser.isValid(information);
	}
	return validForFilter;
}

void TreeWidget::informationDeleted(QObject* deletedObj)
{
	auto it = std::find_if(m_informationConnections.begin(), m_informationConnections.end(), [deletedObj](
			const std::pair<IInformation*, InformationConnections>& pair)
			{
		return pair.first==deletedObj;
	});

	if (it!=m_informationConnections.end())
	{
		m_informationConnections.erase(it);

		// erase item in tree widget
		// if the connections are not defined, the item is not searched because it is already or about to be deleted
		long n = m_treewidget->topLevelItemCount();
		long i = 0;
		QTreeWidgetItem* foundItem = nullptr;
		while (foundItem==nullptr && i<n)
		{
			QTreeWidgetItem* item = m_treewidget->topLevelItem(i);
			QVariant var = item->data(0, Qt::UserRole);
			bool valid = var.isValid() && var.canConvert<IInformation*>();

			if (valid)
			{
				IInformation* info = var.value<IInformation*>();
				valid = info==deletedObj;
			}
			if (valid)
			{
				foundItem = item;
			}
			if (foundItem==nullptr)
			{
				i++;
			}
		}

		if (foundItem)
		{
			delete foundItem;
		}
	}
}

void TreeWidget::parseFilter(const QString& filter)
{
	m_filterStr = m_filter->text();
	m_parser = PropertyFiltersParser(m_filterStr, m_separator);
	populateTree();
}

void TreeWidget::populateTree()
{
	cleanInformationConnections();
	if (m_aggregator==nullptr) {
		m_treewidget->clear();
		return;
	}

	std::list<IInformation*> orderedInformations;

	std::shared_ptr<IInformationIterator> it = m_aggregator->begin();
	while (it->isValid())
	{
		IInformation* info = it->object();
		bool validWithFilter = filterInformation(info);
		if (validWithFilter && m_sortActivate)
		{
			// ordered insertion
			// https://stackoverflow.com/questions/15843525/how-do-you-insert-the-value-in-a-sorted-vector
			std::function<bool(const IInformation*, const IInformation*)> pred = InformationPredicate::createComparator(m_sortProperty);
			std::list<IInformation*>::const_iterator orderedIt;
			orderedIt = std::upper_bound(orderedInformations.begin(), orderedInformations.end(), info, pred);
			orderedInformations.insert(orderedIt, info);
		}
		else if (validWithFilter)
		{
			orderedInformations.push_back(info);
		}

		it->next();
	}

	if (!m_sortOrderTopToBottom)
	{
		orderedInformations.reverse();
	}

	bool currentChanged = true;
	IInformation* currentInformation = nullptr;
	if (m_treewidget->currentItem())
	{
		QVariant var = m_treewidget->currentItem()->data(0, Qt::UserRole);
		if(var.canConvert<IInformation*>())
		{
			currentInformation = qvariant_cast<IInformation*>(var);
		}
	}
	{
		QSignalBlocker b(m_treewidget);
		m_treewidget->clear();
		for (auto infoIt=orderedInformations.begin(); infoIt!=orderedInformations.end(); infoIt++)
		{
			IInformation* info = *infoIt;
			QTreeWidgetItem *item = createItem(info);

			m_treewidget->addTopLevelItem(item);

			if (info==currentInformation)
			{
				currentChanged = false;
				m_treewidget->setCurrentItem(item);
			}
		}
	}
	if (currentChanged) {
		currentItemChanged(nullptr, nullptr);
	}
}

void TreeWidget::removeInformationFromTree(IInformation* information)
{
	// TODO
	int i = 0;
	int N = m_treewidget->topLevelItemCount();
	bool found = false;
	while (!found && i<N)
	{
		QTreeWidgetItem* item = m_treewidget->topLevelItem(i);
		QVariant var = item->data(0, Qt::UserRole);
		bool valid = var.isValid() && var.canConvert<IInformation*>();
		IInformation* itemInfo = nullptr;
		if (valid)
		{
			// need an item with valid information
			itemInfo = var.value<IInformation*>();
			valid = itemInfo!=nullptr;
		}
		found = valid && itemInfo==information;
		if (!found)
		{
			i++;
		}
	}

	if (found)
	{
		QTreeWidgetItem* item = m_treewidget->topLevelItem(i);
		delete item;
	}
}

void TreeWidget::toggleSortOrder()
{
	m_sortOrderTopToBottom = !m_sortOrderTopToBottom;
	if (m_sortOrderTopToBottom)
	{
		m_sortButton->setArrowType(Qt::DownArrow); // top to bottom
	}
	else
	{
		m_sortButton->setArrowType(Qt::UpArrow); // bottom to top
	}
	if (m_sortActivate)
	{
		populateTree(); // could be faster by reversing tree
	}
}

void TreeWidget::unselectAll()
{
	for (int i=0; i<m_treewidget->topLevelItemCount(); i++)
	{
		QTreeWidgetItem* item = m_treewidget->topLevelItem(i);

		if (item->flags() & Qt::ItemIsUserCheckable)
		{
			item->setData(0, Qt::CheckStateRole, Qt::Unchecked);
		}
	}
}

void TreeWidget::selectAll()
{
	for (int i=0; i<m_treewidget->topLevelItemCount(); i++)
	{
		QTreeWidgetItem* item = m_treewidget->topLevelItem(i);

		if (item->flags() & Qt::ItemIsUserCheckable)
		{
			item->setData(0, Qt::CheckStateRole, Qt::Checked);
		}
	}
}

void TreeWidget::setTreeItemIcon(IInformation* info, QTreeWidgetItem* item)
{
	if (info->hasIcon())
	{
		item->setIcon(0, info->icon(16, 16));
	}
	else
	{
		item->setData(0, Qt::DecorationRole, QVariant());
	}
}

void TreeWidget::updateFromInformationIcon(IInformation* searchInfo)
{
	QTreeWidgetItem* foundItem = nullptr;
	QTreeWidgetItem* currentItem = m_treewidget->currentItem();
	if (currentItem)
	{
		QVariant var = currentItem->data(0, Qt::UserRole);
		bool valid = var.isValid() && var.canConvert<IInformation*>();

		if (valid)
		{
			IInformation* info = var.value<IInformation*>();
			valid = info==searchInfo;
		}
		if (valid)
		{
			foundItem = currentItem;
		}
	}

	long n = m_treewidget->topLevelItemCount();
	long i = 0;
	while (foundItem==nullptr && i<n)
	{
		QTreeWidgetItem* item = m_treewidget->topLevelItem(i);
		QVariant var = item->data(0, Qt::UserRole);
		bool valid = var.isValid() && var.canConvert<IInformation*>();

		if (valid)
		{
			IInformation* info = var.value<IInformation*>();
			valid = info==searchInfo;
		}
		if (valid)
		{
			foundItem = item;
		}
		if (foundItem==nullptr)
		{
			i++;
		}
	}

	if (foundItem)
	{
		setTreeItemIcon(searchInfo, foundItem);
	}
}

//class GenericInfoWidget
GenericInfoWidget::GenericInfoWidget(QWidget* parent) : QWidget(parent)
{
	QVBoxLayout* mainLayout = new QVBoxLayout();
	QTabWidget* tabinfos = new QTabWidget();

	QWidget* widgetInfo = new QWidget();
	QVBoxLayout* infoLayout = new QVBoxLayout();
	QGroupBox* groupGeneral = new QGroupBox("General");

	m_infoGrpLayout = new QVBoxLayout();
	groupGeneral->setLayout(m_infoGrpLayout);
	m_generalInfoScrollArea = new QScrollArea;
	m_generalInfoScrollArea->setWidgetResizable(true);
	m_infoGrpLayout->addWidget(m_generalInfoScrollArea);




	infoLayout->addWidget(groupGeneral);
	widgetInfo->setLayout(infoLayout);



	QWidget* widgetData = new QWidget();
	QVBoxLayout* dataLayout = new QVBoxLayout();
	QGroupBox* groupGeneral2 = new QGroupBox("General");
	m_dataGrpLayout = new QVBoxLayout();
	groupGeneral2->setLayout(m_dataGrpLayout);

	m_metaDataScrollArea = new QScrollArea;
	m_metaDataScrollArea->setWidgetResizable(true);
	m_dataGrpLayout->addWidget(m_metaDataScrollArea);

	dataLayout->addWidget(groupGeneral2);
	widgetData->setLayout(dataLayout);

	tabinfos->addTab(widgetInfo,"General Information");
	tabinfos->addTab(widgetData,"Meta Data");

	mainLayout->addWidget(tabinfos);

	setLayout(mainLayout);
}

GenericInfoWidget::~GenericInfoWidget()
{

}

void GenericInfoWidget::informationChanged(IInformationPanelWidget* widget)
{
	if(m_lastWidgetInfo)
	{
		// m_infoGrpLayout->removeWidget(m_lastWidgetInfo);
		m_generalInfoScrollArea->takeWidget();
		m_lastWidgetInfo->deleteLater();
	}

	if (widget)
	{
		// m_infoGrpLayout->addWidget(widget);
		m_generalInfoScrollArea->setWidget(widget);
	}
	m_lastWidgetInfo = widget;
}

void GenericInfoWidget::metadataChanged(QWidget* widget)
{
	if(m_lastWidgetData)
	{
		// m_dataGrpLayout->removeWidget(m_lastWidgetData);
		m_metaDataScrollArea->takeWidget();
		m_lastWidgetData->deleteLater();
	}
	if (widget)
	{
		// m_dataGrpLayout->addWidget(widget);
		m_metaDataScrollArea->setWidget(widget);
	}
	m_lastWidgetData = widget;
}

void GenericInfoWidget::onSave()
{
	if (m_lastWidgetInfo && m_currentInformation)
	{
		m_lastWidgetInfo->saveChanges();
	}
}

IInformation* GenericInfoWidget::currentInformation()
{
	return m_currentInformation;
}

void GenericInfoWidget::setCurrentInformation(IInformation* information)
{
	if (m_currentInformation!=information)
	{
		IInformationPanelWidget* informationWidget = nullptr;
		QWidget* metadataWidget = nullptr;
		if (information)
		{
			informationWidget = information->buildInformationWidget();
			metadataWidget = information->buildMetadataWidget();
		}

		informationChanged(informationWidget);
		metadataChanged(metadataWidget);
		m_currentInformation = information;

		emit currentInformationChanged(m_currentInformation);
	}
}



//============================================================================
//class InformationWidget
InformationWidget::InformationWidget(QWidget* parent) : QWidget(parent)
{
	QVBoxLayout* mainLayout = new QVBoxLayout;


	m_infoWidget = new GenericInfoWidget();


	QWidget* widgetcomm = new QWidget();
	widgetcomm->setMinimumHeight(150);
	widgetcomm->setMaximumHeight(150);
	QHBoxLayout* lay2 = new QHBoxLayout();

	QGroupBox* groupStorage = new QGroupBox("Storage");
	QVBoxLayout* laystore = new QVBoxLayout();
	m_comboStorage = new QComboBox();
	m_comboStorage->addItem("NextVision");
	m_comboStorage->addItem("Sismage");
	m_comboStorage->setEnabled(false);
	laystore->addWidget(m_comboStorage);
	groupStorage->setLayout(laystore);

	QGroupBox* groupComm = new QGroupBox("Comments");
	QVBoxLayout* laycomm = new QVBoxLayout();
	m_edit = new QTextEdit();
	laycomm->addWidget(m_edit);
	groupComm->setLayout(laycomm);


	lay2->addWidget(groupStorage);
	lay2->addWidget(groupComm);

	widgetcomm->setLayout(lay2);


	mainLayout->addWidget(m_infoWidget);
	mainLayout->addWidget(widgetcomm);
	setLayout(mainLayout);

	connect(this, &InformationWidget::currentInformationChanged, m_infoWidget, &GenericInfoWidget::setCurrentInformation);
}

InformationWidget::~InformationWidget()
{

}

IInformation* InformationWidget::currentInformation()
{
	return m_currentInformation;
}

void InformationWidget::setCurrentInformation(IInformation* information)
{
	if (m_currentInformation!=information)
	{
		// comments
		if (information && information->commentsEditable())
		{
			m_edit->setPlainText(information->comments());
		}
		else
		{
			m_edit->setPlainText("");
		}
		m_edit->setEnabled(information!=nullptr);

		// storage
		if (information && information->storage()==information::StorageType::SISMAGE)
		{
			m_comboStorage->setCurrentText("Sismage");
		}
		else
		{
			m_comboStorage->setCurrentText("NextVision");
		}

		m_currentInformation = information;

		emit currentInformationChanged(m_currentInformation);
	}
}

void InformationWidget::onSave()
{
	if (m_currentInformation)
	{
		m_infoWidget->onSave();
		if (m_currentInformation->commentsEditable())
		{
			m_currentInformation->setComments(m_edit->toPlainText());
		}
	}
}


//===================================================================================================
//class ManagerWidget
ManagerWidget::ManagerWidget(IInformationAggregator* aggregator, QWidget* parent) : QWidget(parent)
{
	m_aggregator = aggregator;
	m_aggregator->setParent(this);

	setMinimumSize(1024,600);
	setAttribute(Qt::WA_DeleteOnClose);
	setWindowTitle(" Managers nextvision");

	QVBoxLayout* mainLayout = new QVBoxLayout;

	m_treeWidget = new TreeWidget(m_aggregator);
	m_infoWidget = new InformationWidget();


	QSplitter *splitter = new QSplitter(Qt::Horizontal);
	splitter->setStretchFactor(1,3);

	splitter->addWidget(m_treeWidget);
	splitter->addWidget(m_infoWidget);


	mainLayout->addWidget(splitter);


	QWidget* buttonWidget = new QWidget();
	buttonWidget->setMinimumHeight(50);
	buttonWidget->setMaximumHeight(50);
	QHBoxLayout* layButton = new QHBoxLayout();

	m_buttonNew = new QPushButton("New");
	m_buttonNew->setEnabled(m_aggregator->isCreatable());
	QPushButton* buttonSave = new QPushButton("Save");
	m_buttonDelete = new QPushButton("Delete");
	QPushButton* buttonClose = new QPushButton("Close");

	connect(m_buttonNew,SIGNAL(clicked()),this,SLOT(onNew()));
	connect(buttonSave,SIGNAL(clicked()),this,SLOT(onSave()));
	connect(m_buttonDelete,SIGNAL(clicked()),this,SLOT(onDelete()));
	connect(buttonClose,SIGNAL(clicked()),this,SLOT(onClose()));

	layButton->addWidget(m_buttonNew);
	layButton->addWidget(buttonSave);
	layButton->addWidget(m_buttonDelete);
	layButton->addWidget(buttonClose);

	buttonWidget->setLayout(layButton);

	mainLayout->addWidget(buttonWidget);
	setLayout(mainLayout);

	connect(this, &ManagerWidget::currentInformationChanged, m_infoWidget, &InformationWidget::setCurrentInformation);
	connect(m_treeWidget, &TreeWidget::currentInformationChanged, this, &ManagerWidget::setCurrentInformation);
}



ManagerWidget::~ManagerWidget()
{
	disconnect(m_treeWidget, &TreeWidget::currentInformationChanged, this, &ManagerWidget::setCurrentInformation);
}

void ManagerWidget::onNew()
{
	if (m_aggregator!=nullptr && m_aggregator->isCreatable())
	{
		bool createOutput = m_aggregator->createStorage();
		if (!createOutput)
		{
			QMessageBox::warning(this, tr("Create data"), tr("Failed to create new data"));
		}
	}
}
void ManagerWidget::onSave()
{
	if (m_currentInformation!=nullptr)
	{
		m_infoWidget->onSave();
	}
}

void ManagerWidget::onDelete()
{
	if (m_currentInformation!=nullptr && m_currentInformation->isDeletable())
	{
		QString errorMsg;
		bool deleteOutput = m_aggregator->deleteInformation(m_currentInformation, &errorMsg);
		// <!> warning m_currentInformation could have change because of signals
		if (!deleteOutput)
		{
			QMessageBox::warning(this, tr("Delete data"), tr("Failed to delete data : ") + errorMsg);
		}
		else
			QMessageBox::information(this, tr("Delete data"), tr("Delete success"));
	}
}

void ManagerWidget::onClose()
{
	close();
}

IInformation* ManagerWidget::currentInformation()
{
	return m_currentInformation;
}

void ManagerWidget::setCurrentInformation(IInformation* information)
{
	if (m_currentInformation!=information)
	{
		if (information)
		{
			m_buttonDelete->setEnabled(information->isDeletable());
		}
		else
		{
			m_buttonDelete->setEnabled(false);
		}
		m_currentInformation = information;

		emit currentInformationChanged(m_currentInformation);
	}
}





