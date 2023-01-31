
#include <QGridLayout>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QSpinBox>
#include <QLabel>
#include <QScrollBar>

#include <sampleLimitsChooser.h>

SampleLimitsChooser::SampleLimitsChooser(QWidget *parent, int dimx, int maxSize, float startSample, float stepSample, int *x1, int *x2)
{
	setWindowTitle("trace limits");
	this->dimx = dimx;
	this->maxSize = maxSize;
	this->startSample = startSample;
	this->stepSample = stepSample;
	this->x1 = x1;
	this->x2 = x2;
	this->px1 = *x1;
	this->px2 = *x2;
	this->endSample = (dimx-1)*stepSample + startSample;;

	QVBoxLayout * mainLayout = new QVBoxLayout(this);
	QVBoxLayout * layout1 = new QVBoxLayout(this);
	QString txt = "the dimension x exceeds the maximum acceptable for the GPU memory\n";
	txt += "you have to choose the start time and an end time so that the size is lower than " + QString::number(maxSize) + " pixels";
	QLabel *labelMain = new QLabel(txt);
	QLabel *labelDimx = new QLabel("trace size: " + QString::number(dimx) + " pixels");
	QLabel *labelMaxSize = new QLabel("max size: " + QString::number(maxSize) + " pixels");
	QLabel *labelStartSample = new QLabel("start sample: " + QString::number(startSample) + " ms");
	QLabel *labelStepSample = new QLabel("step sample: " + QString::number(stepSample) + " ms");

	layout1->addWidget(labelMain);
	layout1->addWidget(labelDimx);
	layout1->addWidget(labelMaxSize);
	layout1->addWidget(labelStartSample);
	layout1->addWidget(labelStepSample);
	layout1->setAlignment(Qt::AlignTop);

	labelSizeInfo = new QLabel("");

	QVBoxLayout *vb = new QVBoxLayout;

	QLabel *labelT1 = new QLabel("top offset");
	QHBoxLayout *hbT1 = new QHBoxLayout;
	sbT1 = new QScrollBar();
	sbT1->setOrientation(Qt::Horizontal);
	sbT1->setMinimumWidth(200);
	sbT1->setRange(0, dimx);
	if ( px1 < 0 ) sbT1->setValue(px1); else sbT1->setValue(px1);
	labelT1Value = new QLabel("ms");
	hbT1->addWidget(labelT1);
	hbT1->addWidget(sbT1);
	hbT1->addWidget(labelT1Value);
	hbT1->setAlignment(Qt::AlignLeft);

	QLabel *labelT2 = new QLabel("botton offset");
	QHBoxLayout *hbT2 = new QHBoxLayout;
	sbT2 = new QScrollBar();
	sbT2->setOrientation(Qt::Horizontal);
	sbT2->setMinimumWidth(200);
	sbT2->setRange(0, dimx-1);
	if ( px2 < 0 ) sbT2->setValue(dimx-1); else sbT2->setValue(px2);
	labelT2Value = new QLabel("ms");
	hbT2->addWidget(labelT2);
	hbT2->addWidget(sbT2);
	hbT2->addWidget(labelT2Value);
	hbT2->setAlignment(Qt::AlignLeft);

	vb->addLayout(hbT1);
	vb->addLayout(hbT2);

	QHBoxLayout* sessionLayout = new QHBoxLayout;
	QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	sessionLayout->addWidget(buttonBox);

	mainLayout->addLayout(layout1);
	mainLayout->addWidget(labelSizeInfo);
	mainLayout->addLayout(vb);
	mainLayout->addLayout(sessionLayout);

	connect(sbT1, SIGNAL(valueChanged(int)), this, SLOT(sbT1Change(int)));
	connect(sbT2, SIGNAL(valueChanged(int)), this, SLOT(sbT2Change(int)));
	connect(buttonBox, SIGNAL(accepted()), this, SLOT(accepted()));
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

	this->setMinimumWidth(900);
	this->setMinimumHeight(300);
	labelInfoDisplay();
}


SampleLimitsChooser::~SampleLimitsChooser()
{

}


void SampleLimitsChooser::labelInfoDisplay()
{
	int x1 = sbT1->value();
	int x2 = sbT2->value();

	int startTime = x1 * stepSample + startSample;
	int endTime = x2 * stepSample + startSample;

	int size = x2 - x1 + 1;

	labelT1Value->setText(QString::number(startTime) + " ms - [ " + QString::number(x1) + " pixels ]");
	labelT2Value->setText(QString::number(endTime) + " ms - [ " + QString::number(x2) + " pixels ]");

	QString msg = "size: " + QString::number(size) + " pixels - ";
	msg += "max size: " + QString::number(maxSize) + " pixels\n";
	labelSizeInfo->setText(msg);

	if ( size <= maxSize )
		labelSizeInfo->setStyleSheet("color: #00FF00;");
	else
		labelSizeInfo->setStyleSheet("color: #FF0000;");
}


void SampleLimitsChooser::sbT1Change(int val)
{
	if ( sbT1->value() > sbT2->value()-margin )
		sbT1->setValue(sbT2->value()-margin);
	labelInfoDisplay();
	// fprintf(stderr, "spin: %d\n", val);
}

void SampleLimitsChooser::sbT2Change(int val)
{
	if ( sbT2->value() < sbT1->value()+margin )
		sbT2->setValue(sbT1->value()+margin);
	labelInfoDisplay();
	// fprintf(stderr, "spin: %d\n", val);
}

void SampleLimitsChooser::accepted()
{
	*this->x1 = sbT1->value();
	*this->x2 = sbT2->value();
	accept();
}
