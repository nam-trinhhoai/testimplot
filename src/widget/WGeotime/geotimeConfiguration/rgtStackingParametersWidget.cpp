
#include <QGroupBox>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QComboBox>
#include <QLineEdit>
#include <QCheckBox>
#include <QLabel>

//    	label_sigmagradient->setFixedWidth(label_sigmagradient->fontMetrics().boundingRect(label_sigmagradient->text()).width()*1.5);
#include <workingsetmanager.h>
#include <ProjectManager.h>
#include <horizonSelectWidget.h>
#include <rgtStackingParametersWidget.h>

RgtStackingParametersWidget::RgtStackingParametersWidget(WorkingSetManager *workingSetManager)
{
	m_workingSetManager = workingSetManager;
	ProjectManagerWidget *projectManager = m_workingSetManager->getManagerWidgetV2();
	QVBoxLayout* mainLayout = new QVBoxLayout(this);


	QHBoxLayout *qhbox_iteration = new  QHBoxLayout;
	QLabel *label_iteration = new QLabel("iteration: ");
	label_iteration->setFixedWidth(label_iteration->fontMetrics().boundingRect(label_iteration->text()).width()*1.1);
	lineedit_iteration = new QLineEdit("20");
	qhbox_iteration->addWidget(label_iteration);
	qhbox_iteration->addWidget(lineedit_iteration);
	lineedit_iteration->setFixedWidth(100);
	qhbox_iteration->setAlignment(Qt::AlignLeft);

	QHBoxLayout *qhbox_dipthreshold = new  QHBoxLayout;
	QLabel *label_dipthreshold = new QLabel("dip threshold:");
	label_dipthreshold->setFixedWidth(label_dipthreshold->fontMetrics().boundingRect(label_dipthreshold->text()).width()*1.1);
	lineedit_dipthreshold = new QLineEdit("5.0");
	qhbox_dipthreshold->addWidget(label_dipthreshold);
	qhbox_dipthreshold->addWidget(lineedit_dipthreshold);
	lineedit_dipthreshold->setFixedWidth(100);
	qhbox_dipthreshold->setAlignment(Qt::AlignLeft);

	QHBoxLayout *qhbIterTh = new QHBoxLayout;
	qhbIterTh->addLayout(qhbox_iteration);
	qhbIterTh->addLayout(qhbox_dipthreshold);


	QHBoxLayout *qhbox_decimationfactor = new  QHBoxLayout;
	QLabel *label_decimationfactor = new QLabel("decimation factor:");
	label_decimationfactor->setFixedWidth(label_decimationfactor->fontMetrics().boundingRect(label_decimationfactor->text()).width()*1.1);
	lineedit_decimationfactor = new QLineEdit("1");
	lineedit_decimationfactor->setFixedWidth(100);
	qhbox_decimationfactor->addWidget(label_decimationfactor);
	qhbox_decimationfactor->addWidget(lineedit_decimationfactor);
	qhbox_decimationfactor->setAlignment(Qt::AlignLeft);

   	QHBoxLayout *qhbox_snapping = new  QHBoxLayout;
   	qcb_snapping = new QCheckBox("snapping");
   	qhbox_snapping->addWidget(qcb_snapping);


	QHBoxLayout *qhbox_seedthreshold = new  QHBoxLayout;
	// QLabel *label_seedthreshold = new QLabel("seed threshold:");
	qcb_seedthreshold_valid = new QCheckBox("seed max:");
	qcb_seedthreshold_valid->setChecked(true);
	qcb_seedthreshold_valid->setFixedWidth(qcb_seedthreshold_valid->fontMetrics().boundingRect(qcb_seedthreshold_valid->text()).width()*1.5);

	lineedit_seedthreshold = new QLineEdit("10000");
	lineedit_seedthreshold->setFixedWidth(150);
	// qhbox_seedthreshold->addWidget(label_seedthreshold);
	qhbox_seedthreshold->addWidget(qcb_seedthreshold_valid);
	qhbox_seedthreshold->addWidget(lineedit_seedthreshold);
	qhbox_seedthreshold->setAlignment(Qt::AlignLeft);

	m_horizon = new HorizonSelectWidget();
	m_horizon->setProjectManager(projectManager);
	m_horizon->setWorkingSetManager(m_workingSetManager);

	QVBoxLayout *qvbXlimit = new QVBoxLayout;
	QHBoxLayout *qhbLimitx1 = new QHBoxLayout;
	QLabel *labelXlimit1 = new QLabel("limit x1:");
	lineedit_xlimit1 = new QLineEdit("-1");
	qhbLimitx1->addWidget(labelXlimit1);
	qhbLimitx1->addWidget(lineedit_xlimit1);

	QHBoxLayout *qhbLimitx2 = new QHBoxLayout;
	QLabel *labelXlimit2 = new QLabel("limit x2:");
	lineedit_xlimit2 = new QLineEdit("-1");
	qhbLimitx1->addWidget(labelXlimit2);
	qhbLimitx1->addWidget(lineedit_xlimit2);
	qvbXlimit->addLayout(qhbLimitx1);
	qvbXlimit->addLayout(qhbLimitx2);


	// mainLayout->addLayout(qhbox_iteration);
	// mainLayout->addLayout(qhbox_dipthreshold);
	mainLayout->addLayout(qhbIterTh);
	mainLayout->addLayout(qhbox_decimationfactor);
	mainLayout->addLayout(qhbox_snapping);
	mainLayout->addLayout(qhbox_seedthreshold);
	mainLayout->addWidget(m_horizon);
	mainLayout->addLayout(qvbXlimit);
	mainLayout->setAlignment(Qt::AlignTop);
}




RgtStackingParametersWidget::~RgtStackingParametersWidget()
{

}


void RgtStackingParametersWidget::setNbIter(int val)
{
	lineedit_iteration->setText(QString::number(val));
}

void RgtStackingParametersWidget::setDipThreshold(double val)
{
	lineedit_dipthreshold->setText(QString::number(val));
}

void RgtStackingParametersWidget::setDecimation(int val)
{
	lineedit_decimationfactor->setText(QString::number(val));
}

void RgtStackingParametersWidget::setSnapping(bool val)
{
	if ( val ) qcb_snapping->setCheckState(Qt::Checked); else qcb_snapping->setCheckState(Qt::Unchecked);
}

void RgtStackingParametersWidget::setSeedMaxvalid(bool val)
{
	qcb_seedthreshold_valid->setChecked(val);
}

void RgtStackingParametersWidget::setSeedMax(long val)
{
	lineedit_seedthreshold->setText(QString::number(val));
}

int RgtStackingParametersWidget::getNbIter()
{
	return lineedit_iteration->text().toInt();
}

double RgtStackingParametersWidget::getDipThreshold()
{
	return lineedit_dipthreshold->text().toDouble();
}

int RgtStackingParametersWidget::getDecimation()
{
	return lineedit_decimationfactor->text().toInt();
}

bool RgtStackingParametersWidget::getSnapping()
{
	return qcb_snapping->isChecked();
}

bool RgtStackingParametersWidget::getSeedMaxvalid()
{
	return qcb_seedthreshold_valid->isChecked();
}

long RgtStackingParametersWidget::getSeedMax()
{
	return lineedit_seedthreshold->text().toInt();
}


std::vector<QString> RgtStackingParametersWidget::getHorizonNames()
{
	return m_horizon->getNames();
}

std::vector<QString> RgtStackingParametersWidget::getHorizonPath()
{
	return m_horizon->getPaths();
}


int  RgtStackingParametersWidget::getXlimit1()
{
	return lineedit_xlimit1->text().toInt();
}

int  RgtStackingParametersWidget::getXlimit2()
{
	return lineedit_xlimit2->text().toInt();
}
