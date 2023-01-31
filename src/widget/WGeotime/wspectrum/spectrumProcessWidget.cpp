
#include <QScrollArea>
#include <attributToXtWidget.h>
#include <spectrumProcessWidget.h>


SpectrumProcessWidget::SpectrumProcessWidget()
{
	QHBoxLayout * mainLayout00 = new QHBoxLayout(this);

	QVBoxLayout * selectorWidgetLayout = new QVBoxLayout(this);

	QGroupBox *qgbProgramManager = new QGroupBox;
	QVBoxLayout * mainLayout02 = new QVBoxLayout(qgbProgramManager);
	// m_selectorWidget = new GeotimeProjectManagerWidget(this);
	m_selectorWidget = new ProjectManagerWidget();
	mainLayout02->addWidget(m_selectorWidget);

/*
	m_selectorWidget = new ProjectManagerWidget(this);
	std::vector<ProjectManagerWidget::ManagerTabName> tabNames;
	tabNames.push_back(ProjectManagerWidget::ManagerTabName::SEISMIC);
	tabNames.push_back(ProjectManagerWidget::ManagerTabName::HORIZON);
	tabNames.push_back(ProjectManagerWidget::ManagerTabName::RGBRAW);
	m_selectorWidget->onlyShow(tabNames);
	selectorWidgetLayout->addWidget(m_selectorWidget);
	*/

	QGroupBox *qgbMainLayout01 = new QGroupBox;
	QVBoxLayout *processWidgetLayout = new QVBoxLayout(qgbMainLayout01);

	m_seismicFileSelectWidget = new FileSelectWidget(this);
	m_seismicFileSelectWidget->setProjectManager(m_selectorWidget);
	m_seismicFileSelectWidget->setLabelText("seismic filename");
	m_seismicFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Seismic);
	m_seismicFileSelectWidget->setFileFormat(FileSelectWidget::FILE_FORMAT::INT16);

	m_rgtToRgb16bitsWidget = new RgtToRgb16bitsWidget(m_selectorWidget, this);
	m_rgtToRgb16bitsWidget->setSpectrumProcessWidget(this);

	m_rgb16ToRgb8Widget = new Rgb16ToRgb8Widget(m_selectorWidget, this);
	m_rgb16ToRgb8Widget->setSpectrumProcessWidget(this);

	m_rgtToSeismicMeanWidget = new RgtToSeismicMeanWidget(m_selectorWidget, this);
	m_rgtToSeismicMeanWidget->setSpectrumProcessWidget(this);

	m_rgtToGccWidget = new RgtToGccWidget(m_selectorWidget, this);
	m_rgtToGccWidget->setSpectrumProcessWidget(this);

	m_attributToXtWidget = new AttributToXtWidget(m_selectorWidget, this);
	m_attributToXtWidget->setSpectrumProcessWidget(this);

	m_rawToAviWidget = new RawToAviWidget(m_selectorWidget, this);
	m_rawToAviWidget->setSpectrumProcessWidget(this);

	m_aviViewWidget = new AviViewWidget(m_selectorWidget, this);

	int idx = 0;
	QTabWidget *tabwidget_table1 = new QTabWidget();
	tabwidget_table1->insertTab(idx++, m_rgtToRgb16bitsWidget, QIcon(QString("")), "RGT --> RGB 16 bits");
	// tabwidget_table1->insertTab(idx++, m_rgb16ToRgb8Widget, QIcon(QString("")), "RGB 16 --> RGB 8");
	tabwidget_table1->insertTab(idx++, m_rgtToSeismicMeanWidget, QIcon(QString("")), "RGT --> Mean");
	tabwidget_table1->insertTab(idx++, m_rgtToGccWidget, QIcon(QString("")), "RGT --> Gcc");
	tabwidget_table1->insertTab(idx++, m_attributToXtWidget, QIcon(QString("")), "attribut --> xt");
	tabwidget_table1->insertTab(idx++, m_rawToAviWidget, QIcon(QString("")), "raw  --> avi");
	tabwidget_table1->insertTab(idx++, m_aviViewWidget, QIcon(QString("")), "avi view");

	processWidgetLayout->addWidget(m_seismicFileSelectWidget);
	processWidgetLayout->addWidget(tabwidget_table1);

	/*
	m_systemInfo = new GeotimeSystemInfo(this);
	m_systemInfo->setVisible(true);
	m_systemInfo->setMinimumWidth(350);
	*/
	QGroupBox *qgbSystem = new QGroupBox;
	m_systemInfo = new GeotimeSystemInfo(this);
	m_systemInfo->setVisible(true);
	m_systemInfo->setMinimumWidth(350);
	QVBoxLayout *layout2s = new QVBoxLayout(qgbSystem);
	layout2s->addWidget(m_systemInfo);

	// mainLayout00->addLayout(selectorWidgetLayout);
	// mainLayout00->addLayout(processWidgetLayout);
	// mainLayout00->addWidget(m_systemInfo);
	QTabWidget *tabWidgetMain = new QTabWidget();
	tabWidgetMain->insertTab(0, qgbProgramManager, QIcon(QString("")), "Project Manager");
	tabWidgetMain->insertTab(1, qgbMainLayout01, QIcon(QString("")), "Compute");
	tabWidgetMain->insertTab(2, qgbSystem, QIcon(QString("")), "System");

	QScrollArea *scrollArea = new QScrollArea;
	scrollArea->setWidget(tabWidgetMain);
	scrollArea->setWidgetResizable(true);

	mainLayout00->addWidget(scrollArea);
	resize(1500*2/3, 900);
}


SpectrumProcessWidget::~SpectrumProcessWidget()
{

}


QString SpectrumProcessWidget::getSeismicName()
{
	if ( m_seismicFileSelectWidget )
		return m_seismicFileSelectWidget->getFilename();
	return "";
}

QString SpectrumProcessWidget::getSeismicPath()
{
	if ( m_seismicFileSelectWidget )
		return m_seismicFileSelectWidget->getPath();
	return "";
}

int SpectrumProcessWidget::getDataOutFormat()
{
	return dataOutFormat;
}
