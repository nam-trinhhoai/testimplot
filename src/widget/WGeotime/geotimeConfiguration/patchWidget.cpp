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
#include <QString>

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
#include <patchParametersWidget.h>
#include "collapsablescrollarea.h"
#include <ProjectManagerWidget.h>
#include <fileSelectWidget.h>
#include <patchWidget.h>
#include <workingsetmanager.h>
#include <GeotimeConfiguratorWidget.h>
#include "Xt.h"


PatchWidget::PatchWidget(WorkingSetManager *workingSetManager, QWidget* parent)
{
	m_workingSetManager = workingSetManager;
	m_projectManager = m_workingSetManager->getManagerWidgetV2();
	// patch
	QVBoxLayout *layoutMain = new QVBoxLayout(this);

	QGroupBox *qgbRgtPatch = new QGroupBox("Patch");
	QVBoxLayout *qvbRgtPatch = new QVBoxLayout(qgbRgtPatch);
	checkBoxCompute = new QCheckBox("compute");

	QHBoxLayout *qhbPatchEnable = new QHBoxLayout;
	m_enablePatch = new QCheckBox("enable in rgt compute");
	m_enablePatch->setChecked(true);
	m_patchFileSelectWidget = new FileSelectWidget();
	m_patchFileSelectWidget->setProjectManager(m_projectManager);
	m_patchFileSelectWidget->setWorkingSetManager(m_workingSetManager);
	m_patchFileSelectWidget->setLabelText("name");
	m_patchFileSelectWidget->setLineEditText("patch");
	m_patchFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::patch);
	m_patchFileSelectWidget->setReadOnly(false);
	m_patchFileSelectWidget->setFileFormat(FileSelectWidget::FILE_FORMAT::UINT32);

	qhbPatchEnable->addWidget(m_enablePatch);
	qhbPatchEnable->addWidget(m_patchFileSelectWidget);

	QHBoxLayout *qhbRgtPatchMask = new QHBoxLayout;
	faultInput = new QCheckBox("Fault input");
	m_faultFileSelectWidget = new FileSelectWidget();
	m_faultFileSelectWidget->setProjectManager(m_projectManager);
	m_faultFileSelectWidget->setWorkingSetManager(m_workingSetManager);
	m_faultFileSelectWidget->setLabelText("");
	m_faultFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Seismic);
	m_faultFileSelectWidget->setFileFormat(FileSelectWidget::FILE_FORMAT::INT16);

	qhbRgtPatchMask->addWidget(faultInput);
	qhbRgtPatchMask->addWidget(m_faultFileSelectWidget);

	CollapsableScrollArea* optionsAreaPatchParam = new CollapsableScrollArea("Parameters");
	m_patchParameters = new PatchParametersWidget(m_workingSetManager);
	QVBoxLayout *qvbPatchParam = new QVBoxLayout;
	qvbPatchParam->addWidget(m_patchParameters);
	optionsAreaPatchParam->setContentLayout(*qvbPatchParam);

	// QString style = QString("background-color: rgb(%1,%2,%3);").arg(90).arg(125).arg(160);
	optionsAreaPatchParam->setStyleSheet(GeotimeConfigurationWidget::paramColorStyle);

	qvbRgtPatch->addWidget(checkBoxCompute);
	// qvbRgtPatch->addWidget(m_patchFileSelectWidget);
	qvbRgtPatch->addLayout(qhbPatchEnable);
	// qvbRgtPatch->addLayout(qhbRgtPatchMask);
	qvbRgtPatch->addWidget(optionsAreaPatchParam);
	qgbRgtPatch->setAlignment(Qt::AlignTop);

	layoutMain->addWidget(qgbRgtPatch);
	layoutMain->setAlignment(Qt::AlignTop);
}

PatchWidget::~PatchWidget()
{

}

void PatchWidget::trt_setEnabled(bool val)
{

}

bool PatchWidget::getCompute()
{
	return checkBoxCompute->isChecked();
}

QString PatchWidget::getPatchName()
{
	return m_patchFileSelectWidget->getLineEditText();
}

QString PatchWidget::getPatchPath()
{
	return m_patchFileSelectWidget->getPath();
}


int PatchWidget::getPatchSize()
{
	return m_patchParameters->getPatchSize();
}

bool PatchWidget::getFaultMaskInput()
{
	return m_patchParameters->getFaultMaskInput();
}


QString PatchWidget::getFaultMaskPath()
{
	return m_patchParameters->getFaultMaskPath();
}

int PatchWidget::getFaultThreshold()
{
	return m_patchParameters->getFaultThreshold();
}

QString PatchWidget::getPatchPolarity()
{
	return m_patchParameters->getPatchPolarity();
}

double PatchWidget::getDeltaVOverV()
{
	return m_patchParameters->getDeltaVOverV();
}


int PatchWidget::getGradMax()
{
	return m_patchParameters->getGradMax();
}


std::vector<QString> PatchWidget::getHorizonPaths()
{
	return m_patchParameters->getHorizonPaths();
}

bool PatchWidget::getPatchEnable()
{
	return m_enablePatch->isChecked();
}

void PatchWidget::setConstraintsDims(int dimx, int dimy, int dimz)
{
	m_dimx = dimx;
	m_dimy = dimy;
	m_dimz = dimz;
	m_patchFileSelectWidget->setDims(m_dimx, m_dimy, m_dimz);
	m_faultFileSelectWidget->setDims(m_dimx, m_dimy, m_dimz);
}
