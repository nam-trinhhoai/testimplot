
#include <QHBoxLayout>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>

#include <labelLineEditWidget.h>


LabelLineEditWidget::LabelLineEditWidget(QWidget* parent)
{
	QHBoxLayout * mainLayout = new QHBoxLayout(this);
	m_label = new QLabel("name");
	m_lineEdit = new QLineEdit("");
	mainLayout->addWidget(m_label);
	mainLayout->addWidget(m_lineEdit);
}


// TODO
LabelLineEditWidget:: LabelLineEditWidget(QString label, QString lineEditLabel,  QWidget* parent)
{
	setLabelText(label);
	setLineEditText(lineEditLabel);
}

LabelLineEditWidget::~LabelLineEditWidget()
{

}


void LabelLineEditWidget::setLabelText(QString txt)
{
	if ( m_label == nullptr ) return;
	m_label->setText(txt);
}


void LabelLineEditWidget::setLineEditText(QString txt)
{
	if ( m_lineEdit == nullptr ) return;
	m_lineEdit->setText(txt);
}

QString LabelLineEditWidget::getLineEditText()
{
	return m_lineEdit->text();
}

