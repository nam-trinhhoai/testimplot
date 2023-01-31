#include "exportlayerdialog.h"

#include <QListWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QString>
#include <QPushButton>
#include <QListWidget>
#include <QListWidget>
#include <QDialogButtonBox>
#include <QLineEdit>
#include <QLabel>
#include <QRegularExpression>
#include <QRegularExpressionValidator>

#include "sismagedbmanager.h"

ExportLayerDialog::ExportLayerDialog(const QStringList& pList, QString const& title):
  QDialog(),
  m_newName( "" ),
  m_stringList( pList ),
  m_listWidget( new QListWidget() ),
  m_lineEdit( new QLineEdit() ) {
	this->setWindowTitle(title);

	QHBoxLayout* pMainLayout = new QHBoxLayout();
	this->setLayout(pMainLayout);

	QVBoxLayout* pListLayout = new QVBoxLayout();
	pMainLayout->addLayout(pListLayout);

	QHBoxLayout* newLayout = new QHBoxLayout;
	pListLayout->addLayout(newLayout);

	QRegularExpression regExp(QString::fromStdString(SismageDBManager::getCulturalRegex()));
	m_validator = new QRegularExpressionValidator(regExp, this);
	m_lineEdit->setValidator(m_validator);
	newLayout->addWidget(new QLabel("New name"));
	newLayout->addWidget(m_lineEdit);

	pListLayout->addWidget(m_listWidget);
	m_listWidget->setDragEnabled(true);
	m_listWidget->setDropIndicatorShown(true);
	m_listWidget->setDragDropMode(QAbstractItemView::InternalMove);

	foreach(QString const& string, m_stringList) {
		m_listWidget->addItem(string);
	}
	m_listWidget->setMinimumWidth(m_listWidget->sizeHintForColumn(0)+10);
	m_listWidget->adjustSize();

	//m_stringList->clear();

	connect(m_listWidget, 	&QListWidget::currentItemChanged,
			this, &ExportLayerDialog::slotSelectItem);

	connect(m_lineEdit, &QLineEdit::editingFinished, this, &ExportLayerDialog::newNameChanged);

	QDialogButtonBox *buttonBox = new QDialogButtonBox(
			QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	pListLayout->addWidget(buttonBox);

	connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

	this->adjustSize();
}

ExportLayerDialog::~ExportLayerDialog() {
	  /*int count = m_listWidget->count();
	  for(int index = 0; index < count; index++){
	    QListWidgetItem * item = m_listWidget->item(index);
	    m_stringList->push_back(item->text());
	  }*/
}

/**
 * get the selected index
 * Return -1 if no item seleced
 */
int ExportLayerDialog::getSelectedIndex() const{
	return m_selectedItem;
}

/**
 * get the selected string, use getSelectedIndex to know if there is no selected.
 */
QString ExportLayerDialog::getSelectedString() const{
	if (m_listWidget->currentIndex().row() < 0)
		return "";
  return m_listWidget->currentItem()->text();
}

void ExportLayerDialog::slotSelectItem( QListWidgetItem * current, QListWidgetItem * previous){
  int count = m_listWidget->count();
  for(int index = 0; index < count; index++){
    QListWidgetItem * item = m_listWidget->item(index);
    QFont font = item->font();
    font.setBold(false);
    item->setFont(font);
  }

  m_selectedItem = m_listWidget->currentRow();
  qDebug() << m_selectedItem;
  if (current!=nullptr) {
	  QListWidgetItem* pTemp = m_listWidget->currentItem();
	  QFont font = pTemp->font();
	  font.setBold(true);
	  pTemp->setFont(font);

	  m_lineEdit->setText("");
	  m_newName = "";
  } else {
	  m_selectedItem = -1;
  }
  qDebug() << m_selectedItem << m_newName;
}

void ExportLayerDialog::newNameChanged() {
	if (!m_lineEdit->text().isNull() && !m_lineEdit->text().isEmpty()) {
		QList<QListWidgetItem*> selectedItems = m_listWidget->selectedItems();
		for (QListWidgetItem* item : selectedItems) {
			item->setSelected(false);
		}
		m_selectedItem = -1;

		m_newName = m_lineEdit->text();
	}
	qDebug() << m_selectedItem << m_newName;
}

bool ExportLayerDialog::isNewName() const {
	return m_selectedItem==-1 && !m_newName.isNull() && !m_newName.isEmpty();
}

QString ExportLayerDialog::newName() const {
	return m_newName;
}

