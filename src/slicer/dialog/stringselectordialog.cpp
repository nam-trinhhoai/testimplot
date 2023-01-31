/*
 * StringSelectorDialog.cpp
 *
 *  Created on: Apr 6, 2020
 *      Author: l0222891
 */

#include "stringselectordialog.h"

#include <QListWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QString>
#include <QPushButton>
#include <QListWidget>
#include <QListWidget>
#include <QDialogButtonBox>

StringSelectorDialog::StringSelectorDialog(QStringList* pList, QString const& title):
  QDialog(),
  m_stringList( pList ),
  m_listWidget( new QListWidget() ) {
	this->setWindowTitle(title);

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
			this, &StringSelectorDialog::slotSelectItem);

	QDialogButtonBox *buttonBox = new QDialogButtonBox(
			QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	pListLayout->addWidget(buttonBox);

	connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

	this->adjustSize();
}

StringSelectorDialog::~StringSelectorDialog() {
	  int count = m_listWidget->count();
	  for(int index = 0; index < count; index++){
	    QListWidgetItem * item = m_listWidget->item(index);
	    m_stringList->push_back(item->text());
	  }
}

/**
 * get the selected index
 * Return -1 if no item seleced
 */
int StringSelectorDialog::getSelectedIndex() const{
	return m_selectedItem;
}

/**
 * get the selected string, use getSelectedIndex to know if there is no selected.
 */
QString StringSelectorDialog::getSelectedString() const{
	if (m_listWidget->currentIndex().row() < 0)
		return "";
  return m_listWidget->currentItem()->text();
}

void StringSelectorDialog::slotSelectItem( QListWidgetItem * current, QListWidgetItem * previous){
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
}

