
#include "SaveGraphicLayerDialog.h"

#include <QVBoxLayout>
#include <QLineEdit>
#include <QDialogButtonBox>
#include <QLabel>
#include <QFontMetrics>

SaveGraphicLayerDialog::SaveGraphicLayerDialog(QWidget *parent) :
		QDialog(parent)
{
	QString title="Save Graphic Layer";
	setWindowTitle(title);
	//setWindowFlags(Qt::WindowStaysOnTopHint);

	QFontMetrics metrics(font(), this);
	setMinimumWidth( metrics.horizontalAdvance(title));

	QWidget * internal=new QWidget(this);
	QVBoxLayout *v = new QVBoxLayout(internal);
	m_GraphicLayerFileLineEdit = new QLineEdit(internal);

	v->addWidget(new QLabel("Cultural file name:",internal));
	v->addWidget(m_GraphicLayerFileLineEdit);

	QVBoxLayout * mainLayout=new QVBoxLayout(this);
	mainLayout->addWidget(internal);
	QDialogButtonBox *buttonBox = new QDialogButtonBox(
			QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
	mainLayout->addWidget(buttonBox);
}

QString SaveGraphicLayerDialog::fileName()
{
	return m_GraphicLayerFileLineEdit->text();
}

SaveGraphicLayerDialog::~SaveGraphicLayerDialog() {

}

