
#include <QHBoxLayout>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QPoint>
#include <QSize>

#include "globalconfig.h"
#include <seismicinformationaggregator.h>
// #include <SurveyManager.h>
#include <seismicsurvey.h>
#include <managerwidget.h>
#include <DataSelectorDialog.h>
#include<ProjectManagerNames.h>
#include <fileInformationWidget.h>
#include <ProjectManagerWidget.h>
#include <fileSelectorDialog.h>
#include <managerFileSelectorWidget.h>
#include <QMessageBox>
#include <fileSelectWidget.h>

#include <Xt.h>


FileSelectWidget::FileSelectWidget(QWidget* parent)
{
	ihmCreate();
}


// TODO
FileSelectWidget:: FileSelectWidget(QString label, QString buttonLabel, QString lineEditLabel,  QWidget* parent)
{
	ihmCreate();
	setLabelText(label);
	setButtonText(buttonLabel);
	setLineEditText(lineEditLabel);
}

FileSelectWidget::~FileSelectWidget()
{

}

void FileSelectWidget::setDims(int dimx, int dimy, int dimz)
{
	dimx0 = dimx;
	dimy0 = dimy;
	dimz0 = dimz;
}

void FileSelectWidget::ihmCreate()
{
	m_labelDimensions = new QLabel("...");
	QVBoxLayout *main0 = new QVBoxLayout(this);
	QHBoxLayout * mainLayout = new QHBoxLayout;
	m_label = new QLabel("filename");
	m_button = new QPushButton("");
	m_button->setIcon(QIcon(":/slicer/icons/openfile.png"));
	m_button->setIconSize(QSize(20, 20));
	const QSize size = QSize(20, 20);
	m_button->resize(size);
	// m_button->setFixedWidth(20);
	// m_button->setFixedHeight(20);

	m_lineEdit = new QLineEdit("");
	m_lineEdit->setReadOnly(true);
	mainLayout->addWidget(m_label);
	mainLayout->addWidget(m_lineEdit);
	mainLayout->addWidget(m_button);
	main0->addWidget(m_labelDimensions);
	main0->addLayout(mainLayout);
	m_lineEdit->setContextMenuPolicy(Qt::CustomContextMenu);
	connect(m_button, SIGNAL(clicked()), this, SLOT(buttonClick()));
	connect(m_lineEdit, SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(ProvideContextMenuList(const QPoint &)));
	setLabelDimensionVisible(false);
}

void FileSelectWidget::setProjectManager(ProjectManagerWidget *manager)
{
	m_projectManager = manager;
}

void FileSelectWidget::setLabelText(QString txt)
{
	if ( m_label == nullptr ) return;
	m_label->setText(txt);
}

void FileSelectWidget::setButtonText(QString txt)
{
	if ( m_button == nullptr ) return;
	m_button->setText(txt);
}

void FileSelectWidget::setLineEditText(QString txt)
{
	if ( m_lineEdit == nullptr ) return;
	m_textOriginal = txt;
	m_lineEdit->setText(txt);
}

void FileSelectWidget::clear()
{
	filename = "";
	path = "";
	if ( m_lineEdit )
		m_lineEdit->setText(m_textOriginal);
}

QString FileSelectWidget::getLineEditText()
{
	if ( m_lineEdit ) return m_lineEdit->text();
	return "";
}

void FileSelectWidget::setFileType(FILE_TYPE type)
{
	fileType = type;
}

void FileSelectWidget::setFileSortType(int val)
{
	fileSortType = val;
}

void FileSelectWidget::setReadOnly(bool val)
{
	m_lineEdit->setReadOnly(val);
}


void FileSelectWidget::setLabelDimensionVisible(bool val)
{
	m_labelDimensions->setVisible(val);
}

QString FileSelectWidget::getFilename()
{
	return filename;
}

QString FileSelectWidget::getPath()
{
	return path;
}

void FileSelectWidget::buttonClick()
{
	if ( fileType == FILE_TYPE::seismic )
	{
		seismicFileOpen();
	}

	/*
	if ( m_projectManager == nullptr ) return;

	std::vector<QString> seismicNames;
	std::vector<QString> seismicPath;

	if ( fileType == FILE_TYPE::seismic )
	{
		seismicNames = m_projectManager->getSeismicAllNames();
		seismicPath = m_projectManager->getSeismicAllPath();
	}
	else if ( fileType == FILE_TYPE::rgtCubeToAttribut )
	{
		seismicNames = m_projectManager->getRgbRawDirectoryNames();
		seismicPath = m_projectManager->getRgbRawDirectoryPath();
	}
	else if ( fileType == FILE_TYPE::avi )
	{
		seismicNames = m_projectManager->getAviNames();
		seismicPath = m_projectManager->getAviPath();
	}

	FileSelectorDialog dialog(&seismicNames, "Select file name");
	if ( fileSortType == FILE_SORT_TYPE::All ) dialog.setMainSearchType(FileSelectorDialog::MAIN_SEARCH_LABEL::all);
	else if ( fileSortType == FILE_SORT_TYPE::Seismic ) dialog.setMainSearchType(FileSelectorDialog::MAIN_SEARCH_LABEL::seismic);
	else if ( fileSortType == FILE_SORT_TYPE::Rgt ) dialog.setMainSearchType(FileSelectorDialog::MAIN_SEARCH_LABEL::rgt);
	else if ( fileSortType == FILE_SORT_TYPE::dip ) dialog.setMainSearchType(FileSelectorDialog::MAIN_SEARCH_LABEL::dip);
	else if ( fileSortType == FILE_SORT_TYPE::dipxy ) dialog.setMainSearchType(FileSelectorDialog::MAIN_SEARCH_LABEL::dipxy);
	else if ( fileSortType == FILE_SORT_TYPE::dipxz ) dialog.setMainSearchType(FileSelectorDialog::MAIN_SEARCH_LABEL::dipxz);
	else if ( fileSortType == FILE_SORT_TYPE::patch ) dialog.setMainSearchType(FileSelectorDialog::MAIN_SEARCH_LABEL::patch);
	else if ( fileSortType == FILE_SORT_TYPE::Avi ) dialog.setMainSearchType(FileSelectorDialog::MAIN_SEARCH_LABEL::Avi);
	int code = dialog.exec();
	if (code==QDialog::Accepted)
	{
		int selectedIdx = dialog.getSelectedIndex();
		if (selectedIdx>=0 && selectedIdx<seismicNames.size())
		{
			filename = seismicNames[selectedIdx];
			path = seismicPath[selectedIdx];
			setLineEditText(filename);
			updateLabelDimensions(path);
			// fprintf(stderr, "%d %s\n", selectedIdx, dialog.getSelectedString().toStdString().c_str());
		}
	}
	*/
}


void FileSelectWidget::seismicFileOpen()
{
	// m_projectManager->getSurveyPath();
	// m_projectManager->
	// SeismicSurvey* survey = DataSelectorDialog::dataGetBaseSurvey(m_manager, surveyName, surveyPath, bIsNewSurvey);
    SeismicInformationAggregator* aggregator = new SeismicInformationAggregator(m_workingSetManager, dimx0, dimy0, dimz0);
    // ManagerFileSelectorWidget* widget = new ManagerFileSelectorWidget(aggregator);
    // widget->show();
    ManagerFileSelectorWidget *widget = new ManagerFileSelectorWidget(aggregator);

    if ( fileSortType == FILE_SORT_TYPE::Seismic ) ;
    else if ( fileSortType == FILE_SORT_TYPE::Rgt ) widget->setFilterString("rgt");
    else if ( fileSortType == FILE_SORT_TYPE::dipxy ) widget->setFilterString("dipxy");
    else if ( fileSortType == FILE_SORT_TYPE::dipxz ) widget->setFilterString("dipxz");
    else if ( fileSortType == FILE_SORT_TYPE::patch ) widget->setFilterString("__nextvisionpatch");


    int code = widget->exec();
    if (code==QDialog::Accepted)
    {
    	std::pair<std::vector<QString>, std::vector<QString>> names = widget->getSelectedNames();
    	if ( names.first.size() == 1 )
    	{
    		filename = names.first[0];
    		path = names.second[0];
    		if ( fileFormatCheck(path) )
    		{
    			setLineEditText(filename);
    			updateLabelDimensions(path);
    			emit filenameChanged();
    		}
    	}
    	else
    	{
    		QMessageBox::information(this, tr("Warning"), tr("You have to choose one data and only one data"));
    	}
    }
    delete widget;
}

void FileSelectWidget::updateLabelDimensions(QString filename)
{
	if ( !m_labelDimensions->isVisible() ) return;
	m_labelDimensions->setText("...");
	if ( !QFileInfo::exists(filename) ) return;
	inri::Xt xt((char*)filename.toStdString().c_str());
	if ( !xt.is_valid() ) return;
	int dimx = xt.nSamples();
	int dimy = xt.nRecords();
	int dimz = xt.nSlices();
	QString str = "size: " + QString::number(dimx) + " - " + QString::number(dimy) + " - " + QString::number(dimz);
	m_labelDimensions->setText(str);
}

void FileSelectWidget::ProvideContextMenu(QLineEdit *lineEdit, const QPoint &pos)
{
	QPoint item = lineEdit->mapToGlobal(pos);
	QMenu submenu;
	submenu.addAction("info");
	submenu.addAction("folder");
	QAction* rightClickItem = submenu.exec(item);

	QString path0 = ProjectManagerNames::getAbsolutePath(path);
	if (rightClickItem && rightClickItem->text().contains("info") )
	{

		FileInformationWidget dialog(path);
		int code = dialog.exec();
	}
	else if ( rightClickItem && rightClickItem->text().contains("folder") )
	{
		GlobalConfig& config = GlobalConfig::getConfig();
		QString cmd = config.fileExplorerProgram() + " " + path0;
		system(cmd.toStdString().c_str());
	}
}


void FileSelectWidget::ProvideContextMenuList(const QPoint &pos)
{
	ProvideContextMenu(m_lineEdit, pos);
}

QString FileSelectWidget::fileFormatString(FILE_FORMAT format)
{
	if ( format == FILE_FORMAT::INT16 ) return "short int";
	if ( format == FILE_FORMAT::UINT32 ) return "unsigned int";
	if ( format == FILE_FORMAT::FLOAT32 ) return "float 32";
	return "unknown";
}

bool FileSelectWidget::fileFormatCheck(QString path)
{
	if ( m_fileFormat == FileSelectWidget::FILE_FORMAT::ALL ) return true;
	QFileInfo fi(path);
	QString ext = fi.suffix();
	if ( ext != "xt" ) return true;
	inri::Xt xt(path.toStdString().c_str() );
	if ( !xt.is_valid() ) return false;
	inri::Xt::Type xtType = xt.type();
	QString typeStr = QString::fromStdString(xt.type2str(xtType));
	QString mess = "";
	if ( typeStr == "Float" )
	{
		if ( m_fileFormat != FileSelectWidget::FILE_FORMAT::FLOAT32 ) mess = "warning";
	}
	else if ( typeStr == "Unsigned_32" )
	{
		if ( m_fileFormat != FileSelectWidget::FILE_FORMAT::UINT32 ) mess = "warning";
	}
	else if ( typeStr == "Signed_16" )
	{
		if ( m_fileFormat != FileSelectWidget::FILE_FORMAT::INT16 ) mess = "warning";
	}
	if ( mess.isEmpty() ) return true;
	QString mess0 = "The data is in the wrong format.\nYou choose a " + typeStr + " file format and you have to choose a " + fileFormatString(m_fileFormat) + " format\n";
	mess0 += "Choose another data or convert it with the file converter.";
	QMessageBox::information(this, mess, mess0);
	return false;
}
