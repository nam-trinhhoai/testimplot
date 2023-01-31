#include "gettextwithoverwritedialog.h"

#include "nvlineedit.h"

#include <QPushButton>
#include <QHBoxLayout>
#include <QLabel>
#include <QSizeGrip>
#include <QVBoxLayout>


GetTextWithOverWriteDialog::GetTextWithOverWriteDialog(const QString& prefix, QWidget* parent) : QDialog(parent) {
	m_overwrite = false;
	QVBoxLayout* mainLayout = new QVBoxLayout;
	mainLayout->setContentsMargins(0, 0, 0, 0);
	mainLayout->setSpacing(0);
	setLayout(mainLayout);

	QWidget* holder = new QWidget;
	QVBoxLayout* contentLayout = new QVBoxLayout;
	holder->setLayout(contentLayout);
	mainLayout->addWidget(holder, 1);

	QHBoxLayout* textLayout = new QHBoxLayout;
	contentLayout->addLayout(textLayout);

	textLayout->addWidget(new QLabel(prefix), 0);
	m_textEdit = new NvLineEdit;
	textLayout->addWidget(m_textEdit, 1);

	QHBoxLayout *buttonLayout = new QHBoxLayout;
	contentLayout->addLayout(buttonLayout);
	buttonLayout->addStretch(1);
	QPushButton* cancelButton = new QPushButton("Cancel");
	buttonLayout->addWidget(cancelButton, 0);
	QPushButton* okButton = new QPushButton("Ok");
	buttonLayout->addWidget(okButton, 0);
	QPushButton* overwriteButton = new QPushButton("Overwrite");
	buttonLayout->addWidget(overwriteButton, 0);

	mainLayout->addWidget(new QSizeGrip(this), 0, Qt::AlignRight);

	connect(cancelButton, &QPushButton::clicked, this, &QDialog::reject);
	connect(okButton, &QPushButton::clicked, this, &QDialog::accept);
	connect(overwriteButton, &QPushButton::clicked, this, &GetTextWithOverWriteDialog::overwrite);
}

GetTextWithOverWriteDialog::~GetTextWithOverWriteDialog() {

}

QString GetTextWithOverWriteDialog::text() const {
	return m_textEdit->text();
}

void GetTextWithOverWriteDialog::setText(const QString& text) {
	m_textEdit->setText(text);
}

bool GetTextWithOverWriteDialog::isOverwritten() const {
	return m_overwrite;
}

void GetTextWithOverWriteDialog::overwrite() {
	m_overwrite = true;
	accept();
}
