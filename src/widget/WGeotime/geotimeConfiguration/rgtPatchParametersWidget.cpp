
#include <QGroupBox>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QComboBox>
#include <QLineEdit>
#include <QCheckBox>
#include <QLabel>

#include <ProjectManagerWidget.h>
#include <rgtPatchParametersWidget.h>

RgtPatchParametersWidget::RgtPatchParametersWidget(ProjectManagerWidget *projectManager)
{
	QVBoxLayout* mainLayout = new QVBoxLayout(this);

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
	qcbScaleInit->setChecked(true);
	QHBoxLayout *qhb_rgtNbIterScaleInit = new QHBoxLayout;
	QLabel *labelRgtNbIterScaleInit = new QLabel("Iterations:");
	lineEditRgtIterScaleInit = new QLineEdit("250");
	qhb_rgtNbIterScaleInit->addWidget(labelRgtNbIterScaleInit);
	qhb_rgtNbIterScaleInit->addWidget(lineEditRgtIterScaleInit);

	QHBoxLayout *qhb_rgtEpsilonScaleInit = new QHBoxLayout;
	QLabel *labelRgtEpsilonScaleInit = new QLabel("Smooth factor:");
	lineeditTimeSmoothparameterScaleInit = new QLineEdit("0.01");
	qhb_rgtEpsilonScaleInit->addWidget(labelRgtEpsilonScaleInit);
	qhb_rgtEpsilonScaleInit->addWidget(lineeditTimeSmoothparameterScaleInit);

	QHBoxLayout *qhb_rgtDecimScaleInit = new QHBoxLayout;
	QLabel *labelRgtDecimYScaleInit = new QLabel("Decimation:");
	lineeditRgtDecimYScaleInit = new QLineEdit("10");
	qhb_rgtDecimScaleInit->addWidget(labelRgtDecimYScaleInit);
	qhb_rgtDecimScaleInit->addWidget(lineeditRgtDecimYScaleInit);
	// ==================================================

	QHBoxLayout *qhb_rgtNbIter = new QHBoxLayout;
	QLabel *labelRgtNbIter = new QLabel("Iterations:");
	lineEditRgtIter = new QLineEdit("20");
	qhb_rgtNbIter->addWidget(labelRgtNbIter);
	qhb_rgtNbIter->addWidget(lineEditRgtIter);

	QHBoxLayout *qhb_rgtEpsilon = new QHBoxLayout;
	QLabel *labelRgtEpsilon = new QLabel("Smooth factor:");
	lineeditTimeSmoothparameter = new QLineEdit("0.01");
	qhb_rgtEpsilon->addWidget(labelRgtEpsilon);
	qhb_rgtEpsilon->addWidget(lineeditTimeSmoothparameter);

	QHBoxLayout *qhb_rgtDecim = new QHBoxLayout;
	QLabel *labelRgtDecimY = new QLabel("Decimation:");
	lineeditRgtDecimY = new QLineEdit("2");
	// QLabel *labelRgtDecimZ = new QLabel("decimZ:");
	// lineeditRgtDecimZ = new QLineEdit("1");
	qhb_rgtDecim->addWidget(labelRgtDecimY);
	qhb_rgtDecim->addWidget(lineeditRgtDecimY);
	// qhb_rgtDecim->addWidget(labelRgtDecimZ);
	// qhb_rgtDecim->addWidget(lineeditRgtDecimZ);

	qhbScaleInit->addWidget(qcbScaleInit);
	qhbScaleInit->addLayout(qhb_rgtNbIterScaleInit);
	qhbScaleInit->addLayout(qhb_rgtEpsilonScaleInit);
	qhbScaleInit->addLayout(qhb_rgtDecimScaleInit);

	qhbScale->addLayout(qhb_rgtNbIter);
	qhbScale->addLayout(qhb_rgtEpsilon);
	qhbScale->addLayout(qhb_rgtDecim);

	QHBoxLayout *qhbScales = new QHBoxLayout;
	qhbScales->addWidget(qgbRgtInitScale);
	qhbScales->addWidget(qgbRgtScale);

	// mainLayout->addLayout(qhbRgtIdleDipMax);
	mainLayout->addLayout(qhbScales);


	// mainLayout->addWidget(qgbRgtInitScale);
	// mainLayout->addWidget(qgbRgtScale);
}




RgtPatchParametersWidget::~RgtPatchParametersWidget()
{

}

void RgtPatchParametersWidget::setScaleInitIter(int val)
{
	lineEditRgtIterScaleInit->setText(QString::number(val));
}

void RgtPatchParametersWidget::setScaleInitEpsilon(double val)
{
	lineeditTimeSmoothparameterScaleInit->setText(QString::number(val));
}

void RgtPatchParametersWidget::setScaleInitDecim(int val)
{
	lineeditRgtDecimYScaleInit->setText(QString::number(val));
}


void RgtPatchParametersWidget::setIter(int val)
{
	lineEditRgtIter->setText(QString::number(val));
}

void RgtPatchParametersWidget::setEpsilon(double val)
{
	lineeditTimeSmoothparameter->setText(QString::number(val));
}

void RgtPatchParametersWidget::setDecim(int val)
{
	lineeditRgtDecimY->setText(QString::number(val));
}


bool RgtPatchParametersWidget::getScaleInitValid()
{
	return qcbScaleInit->isChecked();
}

int RgtPatchParametersWidget::getScaleInitIter()
{
	return lineEditRgtIterScaleInit->text().toInt();
}

double RgtPatchParametersWidget::getScaleInitEpsilon()
{
	return lineeditTimeSmoothparameterScaleInit->text().toDouble();
}

int RgtPatchParametersWidget::getScaleInitDecim()
{
	return lineeditRgtDecimYScaleInit->text().toInt();
}

int RgtPatchParametersWidget::getIter()
{
	return lineEditRgtIter->text().toInt();
}

double RgtPatchParametersWidget::getEpsilon()
{
	return lineeditTimeSmoothparameter->text().toDouble();
}

int RgtPatchParametersWidget::getDecim()
{
	return lineeditRgtDecimY->text().toInt();
}
