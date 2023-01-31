
#include <QFormLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QIcon>
#include <QApplication>
#include <QMessageBox>
#include <QFile>

#include <GeotimeProjectManagerWidget.h>
#include <workingsetmanager.h>
#include <DataSelectorDialog.h>
#include <freeHorizonQManager.h>
#include <freeHorizonManager.h>
#include <nextvisionhorizonattributinfo.h>



NextvisionHorizonAttributInfo::NextvisionHorizonAttributInfo(QString path, QString name, WorkingSetManager *workingSetManager,
		QString horizonName, QString horizonPath, QWidget* parent) :
				QWidget(parent) {
	m_path = path;
	m_name = name;
	QFileInfo fi(name);
	m_nameWithoutExt = fi.completeBaseName();
	m_workingSetManager = workingSetManager;
	m_horizonName = horizonName;
	m_horizonPath = horizonPath;

	QFile file (m_path);
	if ( !file.exists() ) return;

	QVBoxLayout* mainLayout0 = new QVBoxLayout(this);
	QHBoxLayout *main0 = new QHBoxLayout;

	QVBoxLayout* mainLayout = new QVBoxLayout;
	mainLayout->setAlignment(Qt::AlignLeft | Qt::AlignTop);

	QFormLayout* layout = new QFormLayout;

	QFrame *line1 = new QFrame;
	line1->setObjectName(QString::fromUtf8("line"));
	line1->setGeometry(QRect(320, 150, 118, 3));
	line1->setFrameShape(QFrame::HLine);

	layout->addRow("Name: ", new QLabel(name));

	m_attributType = FreeHorizonQManager::getPrefixFromFile(m_name);
	// mainLayout->addRow("Type", new QLabel(attributType));
	QString sizeOnDisk = FreeHorizonQManager::getSizeOnDisk(m_path);
	layout->addRow("Size on disk: ", new QLabel(sizeOnDisk));
	if ( m_attributType == "spectrum" )
	{
		QString nFreq = QString::number(FreeHorizonManager::getNbreSpectrumFreq(m_path.toStdString()));
		layout->addRow("Nbre frequencies: ", new QLabel(nFreq));
	}
	if ( m_attributType == "gcc" )
	{
		QString ngcc = QString::number(FreeHorizonManager::getNbreGccScales(m_path.toStdString()));
		layout->addRow("Nbre scales: ", new QLabel(ngcc));
	}

	QFrame *line2 = new QFrame;
	line2->setObjectName(QString::fromUtf8("line"));
	line2->setGeometry(QRect(320, 150, 118, 3));
	line2->setFrameShape(QFrame::HLine);

	QPushButton *bDelete = new QPushButton;
	bDelete->setIcon(QIcon(":slicer/icons/trash_icon.svg"));
	bDelete->setFixedSize(QSize(20, 20));
	bDelete->setToolTip("delete: " + name);
	// if ( m_attributType == "isochrone" ) bDelete->setEnabled(false);

	mainLayout->addLayout(layout);
	main0->addLayout(mainLayout);
	main0->addWidget(bDelete);

	mainLayout0->addWidget(line1);
	mainLayout0->addLayout(main0);
	mainLayout0->addWidget(line2);

	connect(bDelete, SIGNAL(clicked()), this, SLOT(trt_delete()));
}



NextvisionHorizonAttributInfo::~NextvisionHorizonAttributInfo()
{
}

void NextvisionHorizonAttributInfo::deleteAttribut()
{
	QFile file (m_path);
	bool success = false;
	if (file.exists())
		success = file.remove();

	if ( success )
	{
		QString surveyPath = m_workingSetManager->getManagerWidget()->get_survey_fullpath_name();
		QString surveyName = m_workingSetManager->getManagerWidget()->get_survey_name();
		bool bIsNewSurvey = false;
		SeismicSurvey* survey = DataSelectorDialog::dataGetBaseSurvey(m_workingSetManager, surveyName, surveyPath, bIsNewSurvey);
		DataSelectorDialog::removeNVHorizonsAttribut(m_workingSetManager, survey,
				m_horizonPath, m_horizonName, m_path, m_nameWithoutExt); // todo
		this->setVisible(false);
		QMessageBox *msgBox = new QMessageBox(parentWidget());
		msgBox->setText("info");
		msgBox->setInformativeText("Attribut: " + m_name + " deleted");
		msgBox->setStandardButtons(QMessageBox::Ok );
		msgBox->exec();
	}
	else
	{
		QMessageBox *msgBox = new QMessageBox(parentWidget());
		msgBox->setText("info");
		msgBox->setInformativeText("Probleme en deleting attribut: " + m_name + "\nPlease check the file");
		msgBox->setStandardButtons(QMessageBox::Ok );
		msgBox->exec();
	}
}

void NextvisionHorizonAttributInfo::trt_delete()
{
	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	if ( m_attributType == "isochrone" )
		msgBox->setInformativeText("Do you really want to delete the isochrone attribut ?\nIt can be dangerous to keep attributs without this isochrone file");
	else
		msgBox->setInformativeText("Do you really want to delete the attribut:\n" + m_name);
	msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
	int ret = msgBox->exec();
	if ( ret == QMessageBox::Yes )
	{
		deleteAttribut();
	}
}
