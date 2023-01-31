/*
 * GraphicLayerSelectorDialog.cpp
 *
 */

#include "abstract2Dinnerview.h"

#include <QListWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QString>
#include <QPushButton>
#include <QListWidget>
#include <QListWidget>
#include <QDialogButtonBox>
#include "GraphicLayerSelectorDialog.h"

GraphicLayerSelectorDialog::GraphicLayerSelectorDialog(QString Path, Abstract2DInnerView *view):
QDialog(),
m_listWidget( new QListWidget() ),
m_stringList ( new QStringList() ),
m_Path(Path),
m_view(view){

	QString title = tr("Load/Manage Graphic Layer");
	this->setWindowTitle(title);

	QDir culturalDirectory(Path);
	if (view->viewType() == InlineView)
	{
		*m_stringList = QStringList(culturalDirectory.entryList(QStringList()<<"*.section"));
		m_stringList->replaceInStrings(".section", "");
	}
	else
	{
		*m_stringList = QStringList(culturalDirectory.entryList(QStringList()<<"*.map"));
		m_stringList->replaceInStrings(".map", "");
	}
	QHBoxLayout* pMainLayout = new QHBoxLayout();
	this->setLayout(pMainLayout);

	QVBoxLayout* pListLayout = new QVBoxLayout();
	pMainLayout->addLayout(pListLayout);

	pListLayout->addWidget(m_listWidget);
	m_listWidget->setDragEnabled(true);
	m_listWidget->setDropIndicatorShown(true);
	m_listWidget->setDragDropMode(QAbstractItemView::InternalMove);

	foreach(QString const& string, *m_stringList) {
		m_listWidget->addItem(string);
	}
	m_listWidget->setMinimumWidth(m_listWidget->sizeHintForColumn(0)+10);
	m_listWidget->adjustSize();

	m_stringList->clear();

	connect(m_listWidget, 	&QListWidget::currentItemChanged,
			this, &GraphicLayerSelectorDialog::slotSelectItem);

	QDialogButtonBox *buttonBox = new QDialogButtonBox(
			QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	QPushButton *deleteButton = buttonBox->addButton("delete",QDialogButtonBox::ActionRole);
	pListLayout->addWidget(buttonBox);

	connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
	connect(deleteButton, &QPushButton::clicked, this, &GraphicLayerSelectorDialog::deleteFile);

	this->adjustSize();
}

GraphicLayerSelectorDialog::~GraphicLayerSelectorDialog() {
	int count = m_listWidget->count();
	for(int index = 0; index < count; index++){
		QListWidgetItem * item = m_listWidget->item(index);
		m_stringList->push_back(item->text());
	}
}

void GraphicLayerSelectorDialog::deleteFile()
{
	if (m_listWidget->count()>0)
	{
		QFile::remove(m_Path+getSelectedString() +((m_view->viewType()==InlineView)?".section" :".map"));
		qDeleteAll(m_listWidget->selectedItems());
	}
}

/**
 * get the selected index
 * Return -1 if no item seleced
 */
int GraphicLayerSelectorDialog::getSelectedIndex() const{
	return m_selectedItem;
}

/**
 * get the selected string, use getSelectedIndex to know if there is no selected.
 */
QString GraphicLayerSelectorDialog::getSelectedString() const{
	if (m_listWidget->currentIndex().row() < 0)
		return "";
	return m_listWidget->currentItem()->text();
}

void GraphicLayerSelectorDialog::slotSelectItem( QListWidgetItem * current, QListWidgetItem * previous){
	int count = m_listWidget->count();
	for(int index = 0; index < count; index++){
		QListWidgetItem * item = m_listWidget->item(index);
		QFont font = item->font();
		font.setBold(false);
		item->setFont(font);
	}

	m_selectedItem = m_listWidget->currentRow();
	if (m_selectedItem != -1)
	{
		QListWidgetItem* pTemp = m_listWidget->currentItem();
		QFont font = pTemp->font();
		font.setBold(true);
		pTemp->setFont(font);
	}
}

