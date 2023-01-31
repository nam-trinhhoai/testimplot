#include "stratislicechooserdialog.h"
#include <QGridLayout>
#include <QComboBox>
#include <QDialogButtonBox>


#include <QLabel>
#include "seismicsurvey.h"
#include "seismic3dabstractdataset.h"
#include <QStyledItemDelegate>

StratiSliceChooserDialog::StratiSliceChooserDialog(
		SeismicSurvey *survey, QWidget *parent) :
		QDialog(parent) {
	QString title="Strati slice on ";
	title+=survey->name();

	setWindowTitle(title);

	QWidget * internal=new QWidget(this);
	QGridLayout *v = new QGridLayout(internal);
	m_seismicCombo = new QComboBox(internal);
	m_rgtCombo=new QComboBox(internal);

	m_seismicCombo->setItemDelegate(new QStyledItemDelegate());
	m_rgtCombo->setItemDelegate(new QStyledItemDelegate());

	for(Seismic3DAbstractDataset *d:survey->datasets())
	{
		m_seismicCombo->addItem(d->name(),QVariant::fromValue(d));
		m_rgtCombo->addItem(d->name(),QVariant::fromValue(d));
	}

	v->addWidget(new QLabel("Seismic",internal),0,0);
	v->addWidget(m_seismicCombo,0,1);

	v->addWidget(new QLabel("RGT",internal),1,0);
	v->addWidget(m_rgtCombo,1,1);


	QVBoxLayout * mainLayout=new QVBoxLayout(this);
	mainLayout->addWidget(internal);
	QDialogButtonBox *buttonBox = new QDialogButtonBox(
			QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
	mainLayout->addWidget(buttonBox);
}

StratiSliceChooserDialog::~StratiSliceChooserDialog() {

}

Seismic3DAbstractDataset *StratiSliceChooserDialog::seismic()
{
	return m_seismicCombo->currentData(Qt::UserRole).value<Seismic3DAbstractDataset*>();
}
Seismic3DAbstractDataset *StratiSliceChooserDialog::rgt()
{
	return m_rgtCombo->currentData(Qt::UserRole).value<Seismic3DAbstractDataset*>();
}

