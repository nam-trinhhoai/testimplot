#include "multitextselectordialog.h"

#include <QDialogButtonBox>
#include <QListWidget>
#include <QSizeGrip>
#include <QVBoxLayout>

MultiTextSelectorDialog::MultiTextSelectorDialog(QStringList texts, QWidget* parent) : QDialog(parent) {
	QVBoxLayout* outerLayout = new QVBoxLayout;
	setLayout(outerLayout);
	outerLayout->setContentsMargins(0, 0, 0, 0);
	outerLayout->setSpacing(0);

	QVBoxLayout* mainLayout = new QVBoxLayout;
	outerLayout->addLayout(mainLayout);

	m_listWidget = new QListWidget;
	for (const QString& text : texts) {
		m_listWidget->addItem(text);
	}
	mainLayout->addWidget(m_listWidget);

	QDialogButtonBox *buttonBox = new QDialogButtonBox(
			QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	mainLayout->addWidget(buttonBox, 0, Qt::AlignRight);

	QSizeGrip* sizegrip = new QSizeGrip(this);
	sizegrip->setContentsMargins(0, 0, 0, 0);
	outerLayout->addWidget(sizegrip, 0, Qt::AlignRight);

	connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

MultiTextSelectorDialog::~MultiTextSelectorDialog() {

}

QStringList MultiTextSelectorDialog::selectedTexts() {
	QStringList out;
	QList<QListWidgetItem*> selection = m_listWidget->selectedItems();

	for (QListWidgetItem* item : selection) {
		out.append(item->text());
	}
	return out;
}
