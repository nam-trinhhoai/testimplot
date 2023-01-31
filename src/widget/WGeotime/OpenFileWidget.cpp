
#include <QGridLayout>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QLabel>
#include <QSettings>
#include <QStyledItemDelegate>
#include <QFileDialog>
#include <QProcess>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonDocument>

#include "OpenFileWidget.h"



OpenFileWidget::OpenFileWidget(QWidget *parent, std::vector<QString> vTinyNames, std::vector<QString> vFullNames, bool multiSelection) :
								QDialog(parent){
	QString title="Data Selection";
	setWindowTitle(title);

	this->multiSelection = multiSelection;
	QVBoxLayout * mainLayout=new QVBoxLayout(this);
	listFile = new QListWidget;


	QHBoxLayout* sessionLayout = new QHBoxLayout;
	QDialogButtonBox *buttonBox = new QDialogButtonBox(
			QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

	connect(buttonBox, SIGNAL(accepted()), this, SLOT(accepted()));
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

	connect(listFile, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(listFileDoubleClick(QListWidgetItem*)));

	sessionLayout->addWidget(buttonBox);
	mainLayout->addWidget(listFile);
	mainLayout->addLayout(sessionLayout);


	this->vTinyNames = vTinyNames;
	this->vFullNames = vFullNames;
	listFile->clear();
	int idx = 0;
	for (QString name:vTinyNames)
	{
		listFile->addItem(name);
	}
	tinyName = "";
	fullName = "";

	//    QSettings settings;
	//    const QString dirProject = settings.value(RGT_SEISMIC_SLICER_DIR_PROJECT,
	//                                            "").toString();
	//    m_selectorWidget.
	//    const QString project = settings.value(RGT_SEISMIC_SLICER_PROJECT,
	//                                            "").toString();
	//    const QString lastPath = settings.value(RGT_SEISMIC_SLICER_SURVEY_PATH,
	//                                            QDir::homePath()).toString();
	//	QString path(lastPath);
	// connect(loadSessionButton, &QPushButton::clicked, this, &DataSelectorDialog::loadSession);
	// connect(saveSessionButton, &QPushButton::clicked, this, &DataSelectorDialog::saveSession);

	// code to transfert tarum session to next vison session
	// ! warning does not worry about overwriting files !
	//	run();
	this->setMinimumWidth(900);
	this->setMinimumHeight(400);
}

OpenFileWidget::~OpenFileWidget() {

}

int OpenFileWidget::getIndexFromVectorString(std::vector<QString> list, QString txt)
{
    for (int i=0; i<list.size(); i++)
    {
        if ( list[i].compare(txt) == 0 )
            return i;
    }
    return -1;
}

QString OpenFileWidget::getFullNameFromTinyName(QString tinyName)
{
	int idx = getIndexFromVectorString(vTinyNames, tinyName);
	if ( idx < 0 ) return QString("");
	return vFullNames[idx];
}


QString OpenFileWidget::getSelectedTinyName()
{
	return tinyName;
}

QString OpenFileWidget::getSelectedFullName()
{
	return fullName;
}

void OpenFileWidget::listFileDoubleClick(QListWidgetItem* item)
{
	if ( item == nullptr ) return;
	tinyName = item[0].text();
	fullName = getFullNameFromTinyName(tinyName);
	qDebug() << tinyName;
	qDebug() << fullName;
	accept();
}


void OpenFileWidget::accepted() {
	if ( listFile->selectedItems().empty() ) return;
	tinyName = listFile->currentItem()->text();
	fullName = getFullNameFromTinyName(tinyName);
	qDebug() << tinyName;
	qDebug() << fullName;
	accept();
}

