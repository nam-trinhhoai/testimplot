

#include <QGroupBox>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QComboBox>
#include <QLineEdit>
#include <QLabel>

#include <workingsetmanager.h>
#include <fileSelectWidget.h>
#include <ProjectManagerWidget.h>
#include <horizonSelectWidget.h>
#include <patchParametersWidget.h>


PatchParametersWidget::PatchParametersWidget(WorkingSetManager *workingSetManager)
{
	m_workingSetManager = workingSetManager;
	ProjectManagerWidget *projectManager = m_workingSetManager->getManagerWidgetV2();

	QVBoxLayout* mainLayout = new QVBoxLayout(this);

	QHBoxLayout *qhb_patchSize = new QHBoxLayout;
	QLabel *labelPatchSize = new QLabel("Patch size:");
	lineEditPatchSize = new QLineEdit("16");
	qhb_patchSize->addWidget(labelPatchSize);
	qhb_patchSize->addWidget(lineEditPatchSize);

	QHBoxLayout *qhb_patchPolarity = new QHBoxLayout;
	QLabel *labelPatchPolarity = new QLabel("Patch polarity:");
	cbPatchPolarity = new QComboBox;
	cbPatchPolarity->addItem("both");
	cbPatchPolarity->addItem("positive");
	cbPatchPolarity->addItem("negative");
	cbPatchPolarity->setStyleSheet("QComboBox::item{height: 20px}");
	cbPatchPolarity->setStyleSheet("QComboBox::item{width: 20px}");
	qhb_patchPolarity->addWidget(labelPatchPolarity);
	qhb_patchPolarity->addWidget(cbPatchPolarity);

	QHBoxLayout *qhbSizePolarity = new QHBoxLayout;
	qhbSizePolarity->addLayout(qhb_patchSize);
	qhbSizePolarity->addLayout(qhb_patchPolarity);


	QHBoxLayout *qhb_patchGradMax = new QHBoxLayout;
	QLabel *labelPatchGradMax = new QLabel("Gradien max:");
	lineeditPatchGradMax = new QLineEdit("4");
	qhb_patchGradMax->addWidget(labelPatchGradMax);
	qhb_patchGradMax->addWidget(lineeditPatchGradMax);

	QHBoxLayout *qhb_patchDeltaVoverV = new QHBoxLayout;
	QLabel *labelDeltaVoverV = new QLabel("Delta_V / V");
	lineeditPatchDeltaVoverV = new QLineEdit("0.7");
	qhb_patchDeltaVoverV->addWidget(labelDeltaVoverV);
	qhb_patchDeltaVoverV->addWidget(lineeditPatchDeltaVoverV);

	QHBoxLayout *qhbGradDeltaV = new QHBoxLayout;
	qhbGradDeltaV->addLayout(qhb_patchGradMax);
	qhbGradDeltaV->addLayout(qhb_patchDeltaVoverV);

	QHBoxLayout *qhb_patchFitThreshold = new QHBoxLayout;
	QLabel *labelPatchFitThreshold = new QLabel("Min neighboring patch ratio (%):");
	lineEditPatchFitThreshold = new QLineEdit("90");
	qhb_patchFitThreshold->addWidget(labelPatchFitThreshold);
	qhb_patchFitThreshold->addWidget(lineEditPatchFitThreshold);

	QHBoxLayout *qhbRgtPatchMask = new QHBoxLayout;
	faultInput = new QCheckBox("Fault input");
	m_faultFileSelectWidget = new FileSelectWidget();
	m_faultFileSelectWidget->setProjectManager(projectManager);
	m_faultFileSelectWidget->setWorkingSetManager(m_workingSetManager);
	m_faultFileSelectWidget->setLabelText("");
	m_faultFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Seismic);
	m_faultFileSelectWidget->setFileFormat(FileSelectWidget::FILE_FORMAT::INT16);
	qhbRgtPatchMask->addWidget(faultInput);
	qhbRgtPatchMask->addWidget(m_faultFileSelectWidget);

	QHBoxLayout *qhb_patchFaultMaskThreshold = new QHBoxLayout;
	QLabel *labelPatchFaultMaskThreshold = new QLabel("Fault mask threshold:");
	lineEditPatchFaultMaskThreshold = new QLineEdit("10000");
	qhb_patchFaultMaskThreshold->addWidget(labelPatchFaultMaskThreshold);
	qhb_patchFaultMaskThreshold->addWidget(lineEditPatchFaultMaskThreshold);

	m_horizon = new HorizonSelectWidget();
	m_horizon->setProjectManager(projectManager);
	m_horizon->setWorkingSetManager(m_workingSetManager);

	// mainLayout->addLayout(qhb_patchSize);
	// mainLayout->addLayout(qhb_patchPolarity);
	mainLayout->addLayout(qhbSizePolarity);
	// mainLayout->addLayout(qhb_patchGradMax);
	// mainLayout->addLayout(qhb_patchDeltaVoverV);
	mainLayout->addLayout(qhbGradDeltaV);
	mainLayout->addLayout(qhbRgtPatchMask);
	mainLayout->addLayout(qhb_patchFitThreshold);
	mainLayout->addLayout(qhb_patchFaultMaskThreshold);
	mainLayout->addWidget(m_horizon);

	// const QString style = QString("background-color: rgb(%1, %1, %1)").arg(255).arg(0).arg(0);
	// QString style("background-color: rgb(%1,%2,%3);").arg(255);//.arg(0).arg(0);
	// setStyleSheet(style);
	// setStyleSheet("background-color: yellow;");
}


PatchParametersWidget::~PatchParametersWidget()
{


}

void PatchParametersWidget::setPatchSize(int size)
{
	lineEditPatchSize->setText(QString::number(size));
}

void PatchParametersWidget::setPatchPolarity(int pol)
{
	cbPatchPolarity->setCurrentIndex(pol);
}

void PatchParametersWidget::setGradientMax(int grad)
{
	lineeditPatchGradMax->setText(QString::number(grad));
}

void PatchParametersWidget::setPatchRatio(int ratio)
{
	lineEditPatchFitThreshold->setText(QString::number(ratio));
}

void PatchParametersWidget::setFaultThreshold(int th)
{
	lineEditPatchFaultMaskThreshold->setText(QString::number(th));
}


int PatchParametersWidget::getPatchSize()
{
	return lineEditPatchSize->text().toInt();
}

int PatchParametersWidget::getPatchRatio()
{
	return lineEditPatchFitThreshold->text().toInt();
}

int PatchParametersWidget::getFaultThreshold()
{
	return lineEditPatchFaultMaskThreshold->text().toInt();
}

bool PatchParametersWidget::getFaultMaskInput()
{
	return faultInput->isChecked();
}

QString PatchParametersWidget::getFaultMaskPath()
{
	return m_faultFileSelectWidget->getPath();
}

QString PatchParametersWidget::getPatchPolarity()
{
	return cbPatchPolarity->currentText();
}

double PatchParametersWidget::getDeltaVOverV()
{
	return lineeditPatchDeltaVoverV->text().toDouble();
}

int PatchParametersWidget::getGradMax()
{
	return lineeditPatchGradMax->text().toInt();
}

std::vector<QString> PatchParametersWidget::getHorizonPaths()
{
	return m_horizon->getPaths();
}
