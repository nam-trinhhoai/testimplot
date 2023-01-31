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
#include <algorithm>
// #include "FileConvertionXTCWT.h"

#include <rgtPatchExpertWidget.h>




RgtPatchExpertWidget::RgtPatchExpertWidget(GeotimeConfigurationWidget *pconfig, QWidget* parent) :
		QWidget(parent) {

    this->pconf = pconfig;

    // new
    setWindowTitle("RGT Patch Advanced parametres");
    QVBoxLayout * mainLayout=new QVBoxLayout(this);


    QGroupBox *qgbPatch = new QGroupBox(this);
    qgbPatch->setTitle("patch");
    QVBoxLayout* qhbPatch = new QVBoxLayout(qgbPatch);

    QHBoxLayout *qhb_patchSize = new QHBoxLayout;
    QLabel *labelPatchSize = new QLabel("Patch size:");
    lineEditPatchSize = new QLineEdit("8");
    qhb_patchSize->addWidget(labelPatchSize);
    qhb_patchSize->addWidget(lineEditPatchSize);

    QHBoxLayout *qhb_patchPolarity = new QHBoxLayout;
    QLabel *labelPatchPolarity = new QLabel("Patch polarity:");
    cbPatchPolarity = new QComboBox;
    cbPatchPolarity->addItem("both");
    cbPatchPolarity->addItem("positive");
    cbPatchPolarity->addItem("negative");
    cbPatchPolarity->setStyleSheet("QComboBox::item{height: 20px}");
    qhb_patchPolarity->addWidget(labelPatchPolarity);
    qhb_patchPolarity->addWidget(cbPatchPolarity);

    QHBoxLayout *qhb_patchGradMax = new QHBoxLayout;
    QLabel *labelPatchGradMax = new QLabel("Gradien max:");
    lineeditPatchGradMax = new QLineEdit("3");
    qhb_patchGradMax->addWidget(labelPatchGradMax);
    qhb_patchGradMax->addWidget(lineeditPatchGradMax);

    QHBoxLayout *qhb_patchDeltaVoverV = new QHBoxLayout;
    QLabel *labelDeltaVoverV = new QLabel("Delta_V / V");
    lineeditPatchDeltaVoverV = new QLineEdit("0.7");
    qhb_patchDeltaVoverV->addWidget(labelDeltaVoverV);
    qhb_patchDeltaVoverV->addWidget(lineeditPatchDeltaVoverV);

    QHBoxLayout *qhb_patchFitThreshold = new QHBoxLayout;
    QLabel *labelPatchFitThreshold = new QLabel("Min neighboring patch ratio (%):");
    lineEditPatchFitThreshold = new QLineEdit("90");
    qhb_patchFitThreshold->addWidget(labelPatchFitThreshold);
    qhb_patchFitThreshold->addWidget(lineEditPatchFitThreshold);

    QHBoxLayout *qhb_patchFaultMaskThreshold = new QHBoxLayout;
    QLabel *labelPatchFaultMaskThreshold = new QLabel("Fault mask threshold:");
    lineEditPatchFaultMaskThreshold = new QLineEdit("10000");
    qhb_patchFaultMaskThreshold->addWidget(labelPatchFaultMaskThreshold);
    qhb_patchFaultMaskThreshold->addWidget(lineEditPatchFaultMaskThreshold);


    qhbPatch->addLayout(qhb_patchSize);
    qhbPatch->addLayout(qhb_patchPolarity);
    qhbPatch->addLayout(qhb_patchGradMax);
    qhbPatch->addLayout(qhb_patchDeltaVoverV);
    qhbPatch->addLayout(qhb_patchFitThreshold);
    qhbPatch->addLayout(qhb_patchFaultMaskThreshold);


    QGroupBox *qgbRgt = new QGroupBox(this);
    qgbRgt->setTitle("rgt");
    QVBoxLayout* qhbRgt = new QVBoxLayout(qgbRgt);

    QHBoxLayout* qhbRgtIdleDipMax = new QHBoxLayout;
    QLabel *labelIdleDipMax = new QLabel("Dip mask");
    lineEditRgtIdleDipMax = new QLineEdit();
    qhbRgtIdleDipMax->addWidget(labelIdleDipMax);
    qhbRgtIdleDipMax->addWidget(lineEditRgtIdleDipMax);

    QGroupBox *qgbRgtInitScale = new QGroupBox();
    qgbRgtInitScale->setTitle("scale init");
    QVBoxLayout* qhbScaleInit = new QVBoxLayout(qgbRgtInitScale);

    QGroupBox *qgbRgtScale = new QGroupBox();
    qgbRgtScale->setTitle("scale process");
    QVBoxLayout* qhbScale = new QVBoxLayout(qgbRgtScale);

    // ==================================================
    qcbScaleInit = new QCheckBox("init scale");
    QHBoxLayout *qhb_rgtNbIterScaleInit = new QHBoxLayout;
    QLabel *labelRgtNbIterScaleInit = new QLabel("Iterations:");
    lineEditRgtIterScaleInit = new QLineEdit("250");
    qhb_rgtNbIterScaleInit->addWidget(labelRgtNbIterScaleInit);
    qhb_rgtNbIterScaleInit->addWidget(lineEditRgtIterScaleInit);

    QHBoxLayout *qhb_rgtEpsilonScaleInit = new QHBoxLayout;
    QLabel *labelRgtEpsilonScaleInit = new QLabel("Time smooth factor (epsilon):");
    lineeditTimeSmoothparameterScaleInit = new QLineEdit("0.01");
    qhb_rgtEpsilonScaleInit->addWidget(labelRgtEpsilonScaleInit);
    qhb_rgtEpsilonScaleInit->addWidget(lineeditTimeSmoothparameterScaleInit);

    QHBoxLayout *qhb_rgtDecimScaleInit = new QHBoxLayout;
    QLabel *labelRgtDecimYScaleInit = new QLabel("Decimation factor:");
    lineeditRgtDecimYScaleInit = new QLineEdit("1");
    qhb_rgtDecimScaleInit->addWidget(labelRgtDecimYScaleInit);
    qhb_rgtDecimScaleInit->addWidget(lineeditRgtDecimYScaleInit);
    // ==================================================

    QHBoxLayout *qhb_rgtNbIter = new QHBoxLayout;
    QLabel *labelRgtNbIter = new QLabel("Iterations:");
    lineEditRgtIter = new QLineEdit("250");
    qhb_rgtNbIter->addWidget(labelRgtNbIter);
    qhb_rgtNbIter->addWidget(lineEditRgtIter);

    QHBoxLayout *qhb_rgtEpsilon = new QHBoxLayout;
    QLabel *labelRgtEpsilon = new QLabel("Time smooth factor (epsilon):");
    lineeditTimeSmoothparameter = new QLineEdit("0.01");
    qhb_rgtEpsilon->addWidget(labelRgtEpsilon);
    qhb_rgtEpsilon->addWidget(lineeditTimeSmoothparameter);

    QHBoxLayout *qhb_rgtDecim = new QHBoxLayout;
    QLabel *labelRgtDecimY = new QLabel("Decimation factor:");
    lineeditRgtDecimY = new QLineEdit("1");
    QLabel *labelRgtDecimZ = new QLabel("decimZ:");
    lineeditRgtDecimZ = new QLineEdit("1");
    qhb_rgtDecim->addWidget(labelRgtDecimY);
    qhb_rgtDecim->addWidget(lineeditRgtDecimY);
    // qhb_rgtDecim->addWidget(labelRgtDecimZ);
    // qhb_rgtDecim->addWidget(lineeditRgtDecimZ);

    QHBoxLayout *qhbox_buttons = new  QHBoxLayout;
    QPushButton *pushbutton_ok = new QPushButton("OK");
    QPushButton *pushbutton_cancel = new QPushButton("cancel");
    qhbox_buttons->addWidget(pushbutton_ok);
    qhbox_buttons->addWidget(pushbutton_cancel);

    qhbRgt->addLayout(qhbRgtIdleDipMax);
    qhbRgt->addWidget(qgbRgtInitScale);
    qhbRgt->addWidget(qgbRgtScale);

    qhbScaleInit->addWidget(qcbScaleInit);
    qhbScaleInit->addLayout(qhb_rgtNbIterScaleInit);
    qhbScaleInit->addLayout(qhb_rgtEpsilonScaleInit);
    qhbScaleInit->addLayout(qhb_rgtDecimScaleInit);

    qhbScale->addLayout(qhb_rgtNbIter);
    qhbScale->addLayout(qhb_rgtEpsilon);
    qhbScale->addLayout(qhb_rgtDecim);

    mainLayout->addWidget(qgbPatch);
    mainLayout->addWidget(qgbRgt);
    mainLayout->addLayout(qhbox_buttons);
    // mainLayout->addLayout(qhb_patchFitThreshold);

    fill_fields();

    connect(pushbutton_ok, SIGNAL(clicked()), this, SLOT(trt_ok()));
    connect(pushbutton_cancel, SIGNAL(clicked()), this, SLOT(trt_cancel()));
}

RgtPatchExpertWidget::~RgtPatchExpertWidget() {
}

// ******************************************************************
//
// ******************************************************************
void RgtPatchExpertWidget::fill_fields()
{
	lineEditPatchSize->setText(QString::number(this->pconf->patchParam.patchSize));
	lineEditPatchFitThreshold->setText(QString::number(this->pconf->patchParam.patchFitThreshold));
	lineEditPatchFaultMaskThreshold->setText(QString::number(this->pconf->patchParam.patchFaultMaskThreshold));
	if ( pconf->patchParam.patchPolarity.compare("both") == 0 ) cbPatchPolarity->setCurrentIndex(0);
	if ( pconf->patchParam.patchPolarity.compare("positive") == 0 ) cbPatchPolarity->setCurrentIndex(1);
	if ( pconf->patchParam.patchPolarity.compare("negative") == 0 ) cbPatchPolarity->setCurrentIndex(2);
	lineeditPatchGradMax->setText(QString::number(this->pconf->patchParam.patchGradMax));
	lineeditPatchDeltaVoverV->setText(QString::number((double)this->pconf->patchParam.deltaVoverV, 'f', 4));

	lineEditRgtIdleDipMax->setText(QString::number((double)this->pconf->rgtVolumicParam.idleDipMax/1000.0, 'f', 4));
	lineEditRgtIter->setText(QString::number(this->pconf->rgtVolumicParam.scaleParam.nbIter));
	lineeditTimeSmoothparameter->setText(QString::number(this->pconf->rgtVolumicParam.scaleParam.epsilon, 'f', 4));
	lineeditRgtDecimY->setText(QString::number(this->pconf->rgtVolumicParam.scaleParam.decimY));

	lineEditRgtIterScaleInit->setText(QString::number(this->pconf->rgtVolumicParam.initScaleParam.nbIter));
	lineeditTimeSmoothparameterScaleInit->setText(QString::number(this->pconf->rgtVolumicParam.initScaleParam.epsilon, 'f', 4));
	lineeditRgtDecimYScaleInit->setText(QString::number(this->pconf->rgtVolumicParam.initScaleParam.decimY));
	qcbScaleInit->setChecked(this->pconf->rgtVolumicParam.initScaleEnable);

	// lineeditRgtDecimZ->setText(QString::number(this->pconf->rgtVolumicParam.decimZ));
}



void RgtPatchExpertWidget::trt_ok()
{
	this->pconf->patchParam.patchSize = this->lineEditPatchSize->text().toInt();
	this->pconf->patchParam.patchFitThreshold = this->lineEditPatchFitThreshold->text().toInt();
	this->pconf->patchParam.patchFaultMaskThreshold = this->lineEditPatchFaultMaskThreshold->text().toInt();
	this->pconf->rgtVolumicParam.scaleParam.nbIter = this->lineEditRgtIter->text().toInt();
	this->pconf->rgtVolumicParam.scaleParam.epsilon = this->lineeditTimeSmoothparameter->text().toDouble();
	this->pconf->rgtVolumicParam.scaleParam.decimY = lineeditRgtDecimY->text().toInt();
	this->pconf->patchParam.patchPolarity = cbPatchPolarity->currentText().toStdString();
	this->pconf->patchParam.patchGradMax = lineeditPatchGradMax->text().toInt();
	this->pconf->patchParam.deltaVoverV = this->lineeditPatchDeltaVoverV->text().toDouble();

	if ( pconf->patchParam.patchPolarity.compare("positive") == 0 ) cbPatchPolarity->setCurrentIndex(1);
	if ( pconf->patchParam.patchPolarity.compare("negative") == 0 ) cbPatchPolarity->setCurrentIndex(2);

	this->pconf->rgtVolumicParam.initScaleParam.nbIter = this->lineEditRgtIterScaleInit->text().toInt();
	this->pconf->rgtVolumicParam.initScaleParam.epsilon = this->lineeditTimeSmoothparameterScaleInit->text().toDouble();
	this->pconf->rgtVolumicParam.initScaleParam.decimY = lineeditRgtDecimYScaleInit->text().toInt();

	this->pconf->rgtVolumicParam.initScaleEnable = qcbScaleInit->isChecked();

	int val = (int)(lineEditRgtIdleDipMax->text().toDouble()*1000.0);
	val = std::min(val, 32000);
	this->pconf->rgtVolumicParam.idleDipMax = (short)val;
	// this->pconf->rgtVolumicParam.decimZ = lineeditRgtDecimZ->text().toInt();
    QWidget::close();
}

void RgtPatchExpertWidget::trt_cancel()
{
    QWidget::close();
}



