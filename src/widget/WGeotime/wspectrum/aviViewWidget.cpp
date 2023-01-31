



#include <QThread>
#include <QWidget>
#include <QString>
#include <QLineEdit>
#include <QLabel>
#include <QCheckBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QComboBox>
#include <QGroupBox>
#include <QDialog>
#include <QString>
#include <QVBoxLayout>


#include <string>
#include <vector>
#include <string>
#include <math.h>

#include <aviViewWidget.h>

AviViewWidget::AviViewWidget(ProjectManagerWidget *selectorWidget, QWidget* parent)
{
	m_selectorWidget = selectorWidget;
	QVBoxLayout * mainLayout = new QVBoxLayout(this);

	m_aviFileSelectWidget = new FileSelectWidget();
	m_aviFileSelectWidget->setProjectManager(m_selectorWidget);
	m_aviFileSelectWidget->setLabelText("avi filename");
	m_aviFileSelectWidget->setFileType(FileSelectWidget::FILE_TYPE::avi);
	m_aviFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Avi);
	m_aviFileSelectWidget->setLabelDimensionVisible(false);
	m_aviFileSelectWidget->setFileFormat(FileSelectWidget::FILE_FORMAT::ALL);


	QPushButton* pb_start = new QPushButton("start");

	mainLayout->addWidget(m_aviFileSelectWidget);
	mainLayout->addWidget(pb_start);
	mainLayout->setAlignment(Qt::AlignTop);

	connect(pb_start, SIGNAL(clicked()), this, SLOT(trt_run()));
}

AviViewWidget::~AviViewWidget()
{

}

void AviViewWidget::trt_run()
{
	QString avi_2_FullName = m_aviFileSelectWidget->getPath();
	QString cmd = "vlc " + avi_2_FullName;
	int returnVal = system(cmd.toStdString().c_str());
	if (returnVal!=0) {
		cmd = "totem " + avi_2_FullName;
		system(cmd.toStdString().c_str());
	}
}
