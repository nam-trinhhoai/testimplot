
#include "fileSelectorDialog.h"

#include <QListWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QString>
#include <QPushButton>
#include <QListWidget>
#include <QLineEdit>
#include <QComboBox>
#include <QLabel>
#include <QIcon>
#include <freeHorizonQManager.h>
#include <QDialogButtonBox>

FileSelectorDialog::FileSelectorDialog(const std::vector<QString>* pList, QString const& title)
{
	m_list0 = pList;
	this->setWindowTitle(title);
	m_listWidget = new QListWidget();

	QHBoxLayout* pMainLayout = new QHBoxLayout();
	this->setLayout(pMainLayout);

	QVBoxLayout* pListLayout = new QVBoxLayout();
	pMainLayout->addLayout(pListLayout);

	QHBoxLayout* mainSearchLayout = new QHBoxLayout();
	QLabel *mainSearchLabel = new QLabel("main filter");
	m_mainSearch = new QComboBox();
	for (int i=0; i<MAIN_SEARCH_PREFIX.size(); i++)
		m_mainSearch->addItem(MAIN_SEARCH_PREFIX[i]);
	mainSearchLayout->addWidget(mainSearchLabel);
	mainSearchLayout->addWidget(m_mainSearch);
	m_searchString = new QLineEdit();
	pListLayout->addLayout(mainSearchLayout);
	pListLayout->addWidget(m_searchString);
	pListLayout->addWidget(m_listWidget);
	// m_listWidget->setDragEnabled(true);
	// m_listWidget->setDropIndicatorShown(true);
	// m_listWidget->setDragDropMode(QAbstractItemView::InternalMove);
	// m_listWidget->setSelectionMode(QAbstractItemView::MultiSelection);
	setMultipleSelection(false);
	for(int n=0; n<m_list0->size(); n++) {
		m_listWidget->addItem((*m_list0)[n]);
	}
	m_listWidget->setMinimumWidth(m_listWidget->sizeHintForColumn(0)+10);
	m_listWidget->adjustSize();
	m_listWidget->setIconSize(QSize(16, 16));

	// m_stringList->clear();

	connect(m_listWidget, 	&QListWidget::currentItemChanged, this, &FileSelectorDialog::slotSelectItem);

	QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	pListLayout->addWidget(buttonBox);

	connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
	connect(m_listWidget, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(listFileDoubleClick(QListWidgetItem*)));
	connect(m_searchString, SIGNAL(textChanged(QString)), this, SLOT(trt_SearchChange(QString)));
	connect(m_mainSearch, SIGNAL(currentIndexChanged(int)), this, SLOT(trt_mainChangeDisplay(int)));
	displayNames();

	this->adjustSize();
}


FileSelectorDialog::~FileSelectorDialog() {
	/*
	int count = m_listWidget->count();
	for(int index = 0; index < count; index++){
		QListWidgetItem * item = m_listWidget->item(index);
		m_stringList->push_back(item->text());
	}
	*/
}

void FileSelectorDialog::setDataPath(std::vector<QString>* pData)
{
	m_path = pData;
}

void FileSelectorDialog::setMainSearchType(int val)
{
	m_mainSearchType = val;
	m_mainSearch->setCurrentIndex(val);
	displayNames();
}

void FileSelectorDialog::setMultipleSelection(bool val)
{
	if ( val ) m_listWidget->setSelectionMode(QAbstractItemView::MultiSelection);
	else m_listWidget->setSelectionMode(QAbstractItemView::SingleSelection);
}

/**
 * get the selected index
 * Return -1 if no item seleced
 */
int FileSelectorDialog::getSelectedIndex() const{
	// return m_selectedItem;
	QString selectedString = getSelectedString();
	if ( selectedString.isEmpty() ) return -1;
	int idx = -1;
	for (int i=0; i<m_list0->size(); i++)
	{
		if ( selectedString.compare((*m_list0)[i]) == 0 ) return i;
	}
	return -1;
}

std::vector<int> FileSelectorDialog::getMultipleSelectedIndex() const
{
	std::vector<int> idx;
	QList<QListWidgetItem*> list0 = m_listWidget->selectedItems();
	for (int n=0; n<list0.size(); n++)
	{
		QString selectedString = list0[n]->text();
		for (int i=0; i<m_list0->size(); i++)
		{
			if ( selectedString.compare((*m_list0)[i]) == 0 )
			{
				idx.push_back(i);
				break;
			}
		}
	}
	return idx;
}


/**
 * get the selected string, use getSelectedIndex to know if there is no selected.
 */
QString FileSelectorDialog::getSelectedString() const{
	if (m_listWidget->currentIndex().row() < 0)
		return "";
	return m_listWidget->currentItem()->text();
}

void FileSelectorDialog::slotSelectItem( QListWidgetItem * current, QListWidgetItem * previous){
	/*
	int count = m_listWidget->count();
	for(int index = 0; index < count; index++){
		QListWidgetItem * item = m_listWidget->item(index);
		QFont font = item->font();
		font.setBold(false);
		item->setFont(font);
	}

	m_selectedItem = m_listWidget->currentRow();
	qDebug() << m_selectedItem;
	QListWidgetItem* pTemp = m_listWidget->currentItem();
	QFont font = pTemp->font();
	font.setBold(true);
	pTemp->setFont(font);
	*/
}

void FileSelectorDialog::listFileDoubleClick(QListWidgetItem* item)
{
	QDialog::accept();
}




// ====================================================================
std::vector<QString> FileSelectorDialog::getMainSearchSeismicFiles()
{
	std::vector<QString> ret;
	for (int n=0; n<m_list0->size(); n++)
	{
		QString str = (*m_list0)[n];
		bool ok = true;
		for (int i=0; i<FILE_SORT_PREFIX.size(); i++)
			if ( str.contains(FILE_SORT_PREFIX[i],  Qt::CaseInsensitive) )
			{
				ok = false;
				break;
			}
		if ( ok )
		{
			ret.push_back(str);
		}
	}
	return ret;
}

std::vector<QString> FileSelectorDialog::getMainSearchSpecificFiles(std::vector<QString> key)
{
	std::vector<QString> ret;
	for (int n=0; n<m_list0->size(); n++)
	{
		QString str = (*m_list0)[n];
		bool ok = false;
		for (int i=0; i<key.size(); i++)
			if ( isMultiKeyInside(str, key[i] ) )
			{
				ok = true;
				break;
			}
		if ( ok )
			ret.push_back(str);
	}
	return ret;
}


std::vector<QString> FileSelectorDialog::getMainSearchAllFiles()
{
	return *m_list0;
}

std::vector<QString> FileSelectorDialog::getMainSearchFiles()
{
	std::vector<QString> ret;
	switch ( m_mainSearch->currentIndex() )
	{
	case MAIN_SEARCH_LABEL::all: return getMainSearchAllFiles(); break;
	case MAIN_SEARCH_LABEL::horizon: return getMainSearchAllFiles(); break;
	case MAIN_SEARCH_LABEL::seismic: return getMainSearchSeismicFiles(); break;
	case MAIN_SEARCH_LABEL::dip: return getMainSearchSpecificFiles(DIP_SORT); break;
	case MAIN_SEARCH_LABEL::dipxy: return getMainSearchSpecificFiles(DIPXY_SORT); break;
	case MAIN_SEARCH_LABEL::dipxz: return getMainSearchSpecificFiles(DIPXZ_SORT); break;
	case MAIN_SEARCH_LABEL::rgt: return getMainSearchSpecificFiles(RGT_SORT); break;
	case MAIN_SEARCH_LABEL::Avi: return getMainSearchSpecificFiles(AVI_SORT); break;
	default: return getMainSearchAllFiles(); break;
	}
	return ret;
}



bool FileSelectorDialog::isMultiKeyInside(QString str, QString key)
{
	if ( key.isEmpty() ) return true;
	QStringList list1 = key.split(" ", Qt::SkipEmptyParts);
    int nbsearch = list1.size();

    int val = 0;
    for (int s=0; s<nbsearch; s++)
    {
    	int idx = str.indexOf(list1[s], 0, Qt::CaseInsensitive);
    	if ( idx >=0 ) val++;
    }
    if ( val == nbsearch || nbsearch == 0)
    {
    	return true;
    }
    else
    {
    	return false;
    }
}




QString FileSelectorDialog::getPathFromName(QString name)
{
	if ( m_path == nullptr ) return "";
	if ( m_list0->size() != m_path->size() ) return "";
	for (int i=0; i<m_list0->size(); i++)
	{
		if ( (*m_list0)[i] == name )
		{
			return (*m_path)[i];
		}
	}
	return "";
}


void FileSelectorDialog::displayNames()
{
	m_listWidget->clear();
	std::vector<QString> list0 = getMainSearchFiles();
	QString prefix = m_searchString->text();
	for (int n=0; n<list0.size(); n++)
	{
		QString str = list0[n];
		if ( isMultiKeyInside(str, prefix ) )
		{
			QListWidgetItem *item;
			if ( m_mainSearchType == MAIN_SEARCH_LABEL::horizon )
			{
				QString path = getPathFromName(str);
				QIcon icon = FreeHorizonQManager::getHorizonIcon(path, 16);
				item = new QListWidgetItem(icon, str);
			}
			else if ( m_mainSearchType == MAIN_SEARCH_LABEL::seismic )
			{
				QString path = getPathFromName(str);
				QIcon icon = FreeHorizonQManager::getDataSetIcon(path);
				item = new QListWidgetItem(icon, str);
			}
			else
			{
				item = new QListWidgetItem(str);
			}
			// this->m_listWidget->addItem(str);
			this->m_listWidget->addItem(item);
		}
	}
}



void FileSelectorDialog::trt_mainChangeDisplay(int idx)
{
	displayNames();
}




void FileSelectorDialog::trt_SearchChange(QString txt)
{
	displayNames();
}
