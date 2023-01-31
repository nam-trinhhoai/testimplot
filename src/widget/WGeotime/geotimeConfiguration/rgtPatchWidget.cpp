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
#include <workingsetmanager.h>
// #include "FileConvertionXTCWT.h"
#include <rgtPatchParametersWidget.h>
#include "collapsablescrollarea.h"
#include <ProjectManagerWidget.h>
#include <fileSelectWidget.h>
#include <rgtPatchWidget.h>
#include <GeotimeConfiguratorWidget.h>
#include "Xt.h"


RgtPatchWidget::RgtPatchWidget(WorkingSetManager *workingSetManager, QWidget* parent)
{
	m_workingSetManager = workingSetManager;
	ProjectManagerWidget *projectManager = m_workingSetManager->getManagerWidgetV2();
	// patch
	QVBoxLayout *layoutMain = new QVBoxLayout(this);

	QGroupBox *qgbRgtPatchRgt = new QGroupBox("RGT");
	QVBoxLayout *qvbRgtPatchRgt = new QVBoxLayout(qgbRgtPatchRgt);
	m_patchRgtCompute = new QCheckBox("compute");
	// qcb_patchRgtSeismicWeight = new QCheckBox("seismic weight");

	QHBoxLayout *qhbRgtVolumicRgt0 = new QHBoxLayout;
	m_rgtVolumicRgt0 = new QCheckBox("RGT Init");
	m_rgtInitSelectWidget = new FileSelectWidget();
	m_rgtInitSelectWidget->setProjectManager(projectManager);
	m_rgtInitSelectWidget->setWorkingSetManager(m_workingSetManager);
	m_rgtInitSelectWidget->setLabelText("");
	m_rgtInitSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Rgt);
	m_rgtInitSelectWidget->setFileFormat(FileSelectWidget::FILE_FORMAT::INT16);


	qhbRgtVolumicRgt0->addWidget(m_rgtVolumicRgt0);
	qhbRgtVolumicRgt0->addWidget(m_rgtInitSelectWidget);

	QHBoxLayout* qhb_patchRgtRgtName = new QHBoxLayout;//(qgb_orientation);
	QLabel *label_patchRgtRgtName = new QLabel("name");
	m_patchRgtRgtName = new QLineEdit("patch_rgt");
	qhb_patchRgtRgtName->addWidget(label_patchRgtRgtName);
	qhb_patchRgtRgtName->addWidget(m_patchRgtRgtName);

	CollapsableScrollArea* optionsAreaParam = new CollapsableScrollArea("Parameters");
	m_patchParameters = new RgtPatchParametersWidget(projectManager);
	QVBoxLayout *qvbPatchParam = new QVBoxLayout;
	qvbPatchParam->addWidget(m_patchParameters);
	optionsAreaParam->setContentLayout(*qvbPatchParam);

	// QString style = QString("background-color: rgb(%1,%2,%3);").arg(90).arg(125).arg(160);
	optionsAreaParam->setStyleSheet(GeotimeConfigurationWidget::paramColorStyle);


	qvbRgtPatchRgt->addWidget(m_patchRgtCompute);
	qvbRgtPatchRgt->addLayout(qhbRgtVolumicRgt0);
	qvbRgtPatchRgt->addLayout(qhb_patchRgtRgtName);
	qvbRgtPatchRgt->addWidget(optionsAreaParam);
	qgbRgtPatchRgt->setAlignment(Qt::AlignTop);


	layoutMain->addWidget(qgbRgtPatchRgt);
	layoutMain->setAlignment(Qt::AlignTop);
}

RgtPatchWidget::~RgtPatchWidget()
{

}

void RgtPatchWidget::setDecim(int val)
{
	m_patchParameters->setDecim(val);
}

bool RgtPatchWidget::getCompute()
{
	return m_patchRgtCompute->isChecked();
}

QString RgtPatchWidget::getRgtSuffix()
{
	return m_patchRgtRgtName->text();
}


int RgtPatchWidget::getScaleInitIter()
{
	return m_patchParameters->getScaleInitIter();
}

double RgtPatchWidget::getScaleInitEpsilon()
{
	return m_patchParameters->getScaleInitEpsilon();
}

int RgtPatchWidget::getScaleInitDecim()
{
	return m_patchParameters->getScaleInitDecim();
}

int RgtPatchWidget::getIter()
{
	return m_patchParameters->getIter();
}

double RgtPatchWidget::getEpsilon()
{
	return m_patchParameters->getEpsilon();
}

int RgtPatchWidget::getDecim()
{
	return m_patchParameters->getDecim();
}

bool RgtPatchWidget::getScaleInitValid()
{
	return m_patchParameters->getScaleInitValid();
}


bool RgtPatchWidget::getIsRgtInit()
{
	return m_rgtVolumicRgt0->isChecked();
}

QString RgtPatchWidget::getRgtInit()
{
	return m_rgtInitSelectWidget->getPath();
}


void RgtPatchWidget::setConstraintsDims(int dimx, int dimy, int dimz)
{
	m_dimx = dimx;
	m_dimy = dimy;
	m_dimz = dimz;
	m_rgtInitSelectWidget->setDims(m_dimx, m_dimy, m_dimz);
}
