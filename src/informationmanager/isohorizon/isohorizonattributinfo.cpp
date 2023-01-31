
#include <QFormLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QIcon>
#include <QApplication>
#include <QMessageBox>
#include <QFile>

#include <DataSelectorDialog.h>

#include <freeHorizonQManager.h>
#include <freeHorizonManager.h>
#include <isoHorizonQManager.h>
#include <isohorizonattributinfo.h>



IsoHorizonAttributInfo::IsoHorizonAttributInfo(QString path, QString name, QString attributDirPath, QWidget* parent) :
				QWidget(parent) {
	m_path = path;
	m_name = name;
	m_attributDirPath = attributDirPath;

	m_fullPath00000 = m_path + "/iso_00000";
	// m_attributName = FreeHorizonQManager::getAttributData(m_fullPath00000);
	// m_attributPath = FreeHorizonQManager::getAttributPath(m_fullPath00000);

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



IsoHorizonAttributInfo::~IsoHorizonAttributInfo()
{
}

void IsoHorizonAttributInfo::deleteAttribut()
{
	std::vector<QString> path = IsoHorizonQManager::getListDir(m_attributDirPath);
	int cpt = 0;
	std::vector<QString> failFilename;
	for (int i=0; i<path.size(); i++)
	{
		QString filename = path[i] + "/" + m_name;
		QFile file (filename);
		bool ret = file.remove();
		if ( ret ) cpt++;
		else
			failFilename.push_back(filename);
	}

	this->setVisible(false);
	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("info");
	QString mess = "Attribut: " + m_name + " deleted in " + QString::number(cpt) + " directories\n";

	if ( failFilename.size() > 0 )
	{
		mess += "the following attribut could not be deleted. Please check the files:\n";
		for (int i=0; i<failFilename.size(); i++)
			mess += failFilename[i] + "\n";
	}

	msgBox->setInformativeText(mess);
	msgBox->setStandardButtons(QMessageBox::Ok );
	msgBox->exec();
}

void IsoHorizonAttributInfo::trt_delete()
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
