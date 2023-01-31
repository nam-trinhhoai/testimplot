
#include <QHBoxLayout>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QListWidget>

#include "nextvisionhorizoninformationaggregator.h"
#include <managerwidget.h>
#include <managerFileSelectorWidget.h>
#include <ProjectManagerWidget.h>
#include <workingsetmanager.h>
#include <fileSelectorDialog.h>
#include <horizonSelectWidget.h>


HorizonSelectWidget::HorizonSelectWidget(QWidget* parent)
{
	ihmCreate();
}


HorizonSelectWidget::~HorizonSelectWidget()
{

}

void HorizonSelectWidget::ihmCreate()
{
	QHBoxLayout *mainLayout = new QHBoxLayout(this);
	m_label = new QLabel("horizons");
	m_listWidget = new QListWidget();
	QVBoxLayout *qvbButtonlayout = new QVBoxLayout;
	m_addButton = new QPushButton("add");
	m_supprButton = new QPushButton("suppr");
	qvbButtonlayout->addWidget(m_addButton);
	qvbButtonlayout->addWidget(m_supprButton);

	mainLayout->addWidget(m_label);
	mainLayout->addWidget(m_listWidget);
	mainLayout->addLayout(qvbButtonlayout);

	m_listWidget->setMaximumHeight(100);

	// setMinimumHeight(150/2);
	// setMaximumHeight(150*2);
	// setMaximumWidth(600);

	connect(m_addButton, SIGNAL(clicked()), this, SLOT(trt_horizonAdd()));
	connect(m_supprButton, SIGNAL(clicked()), this, SLOT(trt_horizonClear()));
}

/*
void FileSelectWidget::setProjectManager(ProjectManagerWidget *manager)
{
	m_projectManager = manager;
}
*/

void HorizonSelectWidget::setLabelText(QString txt)
{
	if ( m_label == nullptr ) return;
	m_label->setText(txt);
}

void HorizonSelectWidget::setAddButtonLabel(QString txt)
{
	if ( m_addButton == nullptr ) return;
	m_addButton->setText(txt);
}

void HorizonSelectWidget::setSupprButtonLabel(QString txt)
{
	if ( m_addButton == nullptr ) return;
	m_supprButton->setText(txt);
}

void HorizonSelectWidget::setProjectManager(ProjectManagerWidget *manager)
{
	m_projectManager = manager;
}

void HorizonSelectWidget::setListMultiSelection(bool type)
{
	m_multiSelection = type;
	if ( type )
	{
		m_listWidget->setSelectionMode(QAbstractItemView::MultiSelection);
	}
	else
	{
		m_listWidget->setSelectionMode(QAbstractItemView::SingleSelection);
	}
}

void HorizonSelectWidget::addData(QString name, QString path)
{
	m_names.push_back(name);
	m_path.push_back(path);
	display();
}

void HorizonSelectWidget::clearData()
{
	m_names.clear();
	m_path.clear();
	display();
}


std::vector<QString> HorizonSelectWidget::getNames()
{
	return m_names;
}

std::vector<QString> HorizonSelectWidget::getPaths()
{
	return m_path;
}


void HorizonSelectWidget::display()
{
	m_listWidget->clear();
	for (long i=0; i<m_names.size(); i++)
		m_listWidget->addItem(m_names[i]);

}


void HorizonSelectWidget::trt_horizonAdd()
{
	NextvisionHorizonInformationAggregator* aggregator = new NextvisionHorizonInformationAggregator(m_workingSetManager, false);
    // ManagerFileSelectorWidget* widget = new ManagerFileSelectorWidget(aggregator);
    // widget->show();
    ManagerFileSelectorWidget *widget = new ManagerFileSelectorWidget(aggregator);
    int code = widget->exec();
    if (code==QDialog::Accepted)
    {
    	std::pair<std::vector<QString>, std::vector<QString>> names = widget->getSelectedNames();

    	for (int i=0; i<names.first.size(); i++)
    	{
    		QString filename = names.first[i];
    		QString path = names.second[i];
    		if ( !nameExist(m_names, filename) )
    		{
    			m_names.push_back(filename);
    			m_path.push_back(path);
    		}
    	}
    }
    delete widget;
	display();
}

void HorizonSelectWidget::trt_horizonClear()
{
	m_names.clear();
	m_path.clear();
	display();
}


/*
void FileSelectWidget::buttonClick()
{

}
*/

bool HorizonSelectWidget::nameExist(std::vector<QString> names, QString name)
{
	for (QString inames : names)
	{
		if ( name.compare(inames) == 0 ) return true;
	}
	return false;
}
