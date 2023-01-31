/*
 *
 *
 *  Created on:
 *      Author: l1000501
 */


#include <QTableView>
#include <QHeaderView>
#include <QStandardItemModel>
#include <QPushButton>
#include <QRadioButton>
#include <QGroupBox>
#include <QLabel>
#include <QPainter>
#include <QChart>
#include <QLineEdit>
#include <QToolButton>
#include <QLineSeries>
#include <QScatterSeries>
#include <QtCharts>
#include <QRandomGenerator>
#include <QTimer>

#include <QVBoxLayout>
#include <QHBoxLayout>

#include <dialog/validator/OutlinedQLineEdit.h>
#include <dialog/validator/SimpleDoubleValidator.h>

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <sys/sysinfo.h>


#include <vector>
#include <math.h>
#include <cmath>
#include <iostream>
// #include "FileConvertionXTCWT.h"
#include <ProjectManagerWidget.h>
#include <fileSelectWidget.h>
#include <orientationWidget.h>
#include "collapsablescrollarea.h"
#include <GeotimeConfiguratorWidget.h>
#include <orientationParametersWidget.h>

#include "Xt.h"




OrientationWidget::OrientationWidget(ProjectManagerWidget *projectManager, bool enableParam, WorkingSetManager *workingSetManager, QWidget* parent) :
				QWidget(parent) {

	m_projectManager = projectManager;
	m_workingSetManager = workingSetManager;
	// QVBoxLayout *vBoxMain0 = new QVBoxLayout(this);

	// groupBox = new QGroupBox(this);
	// groupBox->setTitle("Orientation test");
	QVBoxLayout *vBoxMain = new QVBoxLayout(this);

	QHBoxLayout *hBoxCompute = new QHBoxLayout;
	checkBoxCompute = new QCheckBox("compute");
	comboProcessingType = new QComboBox();
	comboProcessingType->addItem("CPU");
	comboProcessingType->addItem("GPU");
	comboProcessingType->setCurrentIndex(1);
	comboProcessingType->setStyleSheet("QComboBox::item{height: 20px}");
	hBoxCompute->addWidget(checkBoxCompute);
	hBoxCompute->addWidget(comboProcessingType);

	m_dipxyFileSelectWidget = new FileSelectWidget();
	m_dipxyFileSelectWidget->setProjectManager(projectManager);
	m_dipxyFileSelectWidget->setWorkingSetManager(m_workingSetManager);
	m_dipxyFileSelectWidget->setLabelText("dipxy filename");
	m_dipxyFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::dipxy);
	m_dipxyFileSelectWidget->setReadOnly(false);
	m_dipxyFileSelectWidget->setLineEditText("dipxy");
	m_dipxyFileSelectWidget->setFileFormat(FileSelectWidget::FILE_FORMAT::INT16);

	m_dipxzFileSelectWidget = new FileSelectWidget();
	m_dipxzFileSelectWidget->setProjectManager(projectManager);
	m_dipxzFileSelectWidget->setWorkingSetManager(m_workingSetManager);
	m_dipxzFileSelectWidget->setLabelText("dipxz filename");
	m_dipxzFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::dipxz);
	m_dipxzFileSelectWidget->setReadOnly(false);
	m_dipxzFileSelectWidget->setLineEditText("dipxz");
	m_dipxzFileSelectWidget->setFileFormat(FileSelectWidget::FILE_FORMAT::INT16);

	CollapsableScrollArea* optionsAreaParam = new CollapsableScrollArea("Parameters");
	m_parameters = new OrientationParametersWidget();
	QVBoxLayout *qvbParam = new QVBoxLayout;
	qvbParam->addWidget(m_parameters);
	optionsAreaParam->setContentLayout(*qvbParam);
	// QString style = QString("background-color: rgb(%1,%2,%3);").arg(40).arg(56).arg(72);
	// QString style = QString("background-color: rgb(40,56,72);");
	optionsAreaParam->setStyleSheet(GeotimeConfigurationWidget::paramColorStyle);

	vBoxMain->setAlignment(Qt::AlignTop);
	// vBoxMain->addStretch(1);

	// vBoxMain->addWidget(groupBox);
	if ( enableParam )  vBoxMain->addLayout(hBoxCompute);
	vBoxMain->addWidget(m_dipxyFileSelectWidget);
	vBoxMain->addWidget(m_dipxzFileSelectWidget);
	if ( enableParam ) vBoxMain->addWidget(optionsAreaParam);
	vBoxMain->setAlignment(Qt::AlignTop);

	// vBoxMain->setSpacing(0);
	// vBoxMain->setContentsMargins(0,0,0,0);


	// vBoxMain0->addLayout(vBoxMain);
	trt_setEnabled(true);
	connect(checkBoxCompute, SIGNAL(clicked(bool)), this, SLOT(trt_setEnabled(bool)));
}

OrientationWidget::~OrientationWidget() {

}

void OrientationWidget::setConstraintsDims(int dimx, int dimy, int dimz)
{
	m_dimx0 = dimx;
	m_dimy0 = dimy;
	m_dimz0 = dimz;
	m_dipxyFileSelectWidget->setDims(m_dimx0, m_dimy0, m_dimz0);
	m_dipxzFileSelectWidget->setDims(m_dimx0, m_dimy0, m_dimz0);
}


QString OrientationWidget::getDipxyFilename()
{
	// return m_dipxyFileSelectWidget->getFilename();
	return m_dipxyFileSelectWidget->getLineEditText();

}

QString OrientationWidget::getDipxyPath()
{
	return m_dipxyFileSelectWidget->getPath();
}

QString OrientationWidget::getDipxzFilename()
{
	// return m_dipxzFileSelectWidget->getFilename();
	return m_dipxzFileSelectWidget->getLineEditText();
}

QString OrientationWidget::getDipxzPath()
{
	return m_dipxzFileSelectWidget->getPath();
}

int OrientationWidget::getProcessingTypeIndex()
{
	return comboProcessingType->currentIndex();
}

bool OrientationWidget::getComputationChecked()
{
	return checkBoxCompute->isChecked();
}

void OrientationWidget::trt_setEnabled(bool val)
{
	if ( checkBoxCompute->isChecked() )
	{
		m_dipxyFileSelectWidget->setReadOnly(false);
		m_dipxzFileSelectWidget->setReadOnly(false);
	}
	else
	{
		m_dipxyFileSelectWidget->setReadOnly(true);
		m_dipxzFileSelectWidget->setReadOnly(true);
	}
}


void OrientationWidget::setGradient(double val)
{
	m_parameters->setGradient(val);
}

void OrientationWidget::setTensor(double val)
{
	m_parameters->setTensor(val);
}

double OrientationWidget::getGradient()
{
	return m_parameters->getGradient();
}

double OrientationWidget::getTensor()
{
	return m_parameters->getTensor();
}
