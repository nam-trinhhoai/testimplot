
#include <QSizeGrip>
#include <QVBoxLayout>
#include <importsismagehorizonwithworkingsetwidget.h>
#include <importsismagehorizondialog.h>




ImportSismageHorizonDialog::ImportSismageHorizonDialog(WorkingSetManager* workingSet)
{
	setWindowTitle("Horizon Sismage Import");
	setAttribute(Qt::WA_DeleteOnClose);
	m_workingSetManager = workingSet;
	m_importSismageHorizonWithWorkingSetManager = new ImportSismageHorizonWithWorkingSetWidget(workingSet);

	QVBoxLayout *layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);
	layout->addWidget(m_importSismageHorizonWithWorkingSetManager);
	layout->addWidget(new QSizeGrip(this), 0, Qt::AlignRight);

	// QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, Qt::Horizontal);
	// layout->addWidget(buttonBox);

	setMinimumHeight(800);
	setMinimumWidth(400);
}


ImportSismageHorizonDialog::~ImportSismageHorizonDialog()
{
	}
