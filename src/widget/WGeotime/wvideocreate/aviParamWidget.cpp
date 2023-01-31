
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QScrollArea>
#include <QColorDialog>

#include <videocreatewidget.h>
#include <aviParamWidget.h>




AviParamWidget::AviParamWidget(VideoCreateWidget *videoParamWidget)
{
	m_videoParamWidget = videoParamWidget;
	QVBoxLayout * mainLayout = new QVBoxLayout(this);

	// QHBoxLayout *layout42 = new QHBoxLayout;
	// QLabel *labelaviPrefix = new QLabel("video prefix:");
	// lineedit_aviPrefix = new QLineEdit("video");
	// layout42->addWidget(labelaviPrefix);
	// layout42->addWidget(lineedit_aviPrefix);

	m_onlyFirstImage = new QCheckBox("only first image");
	m_onlyFirstImage->setChecked(false);

	QHBoxLayout* layout43 = new QHBoxLayout;
	QLabel* labelvideoScale = new QLabel("Scale factor :");
	spinboxvideoScale = new QDoubleSpinBox;
	spinboxvideoScale->setMinimum(std::numeric_limits<double>::min());
	spinboxvideoScale->setMaximum(std::numeric_limits<double>::max());
	spinboxvideoScale->setValue(videoScale);

	layout43->addWidget(labelvideoScale, 0);
	layout43->addWidget(spinboxvideoScale, 1);

	QHBoxLayout* layout44 = new QHBoxLayout();
	QLabel* labelTOrigin = new QLabel("T origin :");
	spinboxRgb1TOrigin = new QSpinBox;
	spinboxRgb1TOrigin->setMinimum(0);
	spinboxRgb1TOrigin->setMaximum(std::numeric_limits<int>::max());
	spinboxRgb1TOrigin->setValue(rgb1TOrigin);

	layout44->addWidget(labelTOrigin, 0);
	layout44->addWidget(spinboxRgb1TOrigin, 1);

	QHBoxLayout* layout45 = new QHBoxLayout();
	QLabel* labelTStep = new QLabel("T Step :");
	spinboxRgb1TStep = new QSpinBox;
	spinboxRgb1TStep->setMinimum(1);
	spinboxRgb1TStep->setMaximum(std::numeric_limits<int>::max());
	spinboxRgb1TStep->setValue(rgb1TStep);

	layout45->addWidget(labelTStep, 0);
	layout45->addWidget(spinboxRgb1TStep, 1);

	QHBoxLayout* layout46 = new QHBoxLayout();
	QLabel* labelIsReversed = new QLabel("Is time reversed :");
	checkboxRgb1IsReversed = new QCheckBox();
	checkboxRgb1IsReversed->setCheckState((rgb1IsReversed) ? Qt::Checked : Qt::Unchecked);

	layout46->addWidget(labelIsReversed, 0);
	layout46->addWidget(checkboxRgb1IsReversed, 1);

	QHBoxLayout* layout47 = new QHBoxLayout();
	QLabel* labelFps = new QLabel("FPS :");
	spinboxFps = new QSpinBox;
	spinboxFps->setMinimum(1);
	spinboxFps->setMaximum(std::numeric_limits<int>::max());
	spinboxFps->setValue(framePerSecond);

	layout47->addWidget(labelFps, 0);
	layout47->addWidget(spinboxFps, 1);

	QHBoxLayout* layout48 = new QHBoxLayout();
	QLabel* labelFirstIso = new QLabel("First Iso :");
	spinboxFirstIso = new QSpinBox;
	spinboxFirstIso->setMinimum(std::numeric_limits<int>::min());
	spinboxFirstIso->setMaximum(std::numeric_limits<int>::max());
	spinboxFirstIso->setValue(firstIso);

	layout48->addWidget(labelFirstIso, 0);
	layout48->addWidget(spinboxFirstIso, 1);

	QHBoxLayout* layout49 = new QHBoxLayout();
	QLabel* labelLastIso = new QLabel("Last Iso :");
	spinboxLastIso = new QSpinBox;
	spinboxLastIso->setMinimum(std::numeric_limits<int>::min());
	spinboxLastIso->setMaximum(std::numeric_limits<int>::max());
	spinboxLastIso->setValue(lastIso);

	layout49->addWidget(labelLastIso, 0);
	layout49->addWidget(spinboxLastIso, 1);

	QHBoxLayout* layout410 = new QHBoxLayout();

	directionComboBox = new QComboBox();
	directionComboBox->addItem("Inline", 0);
	directionComboBox->addItem("Xline", 1);
	sectionPositionSpinBox = new QSpinBox();
	sectionPositionSpinBox->setMinimum(0);
	sectionPositionSpinBox->setMaximum(1);
	sectionPositionSpinBox->setValue(0);
	sectionPositionSlider = new QSlider(Qt::Horizontal);
	sectionPositionSlider->setMinimum(0);
	sectionPositionSlider->setMaximum(1);
	sectionPositionSlider->setValue(0);

	layout410->addWidget(directionComboBox);
	layout410->addWidget(sectionPositionSpinBox);
	layout410->addWidget(sectionPositionSlider);

	QHBoxLayout* layout411 = new QHBoxLayout;
	QSpinBox* textSizeSpinBox = new QSpinBox;
	textSizeSpinBox->setMinimum(1);
	textSizeSpinBox->setMaximum(std::numeric_limits<int>::max());
	textSizeSpinBox->setValue(textSize);
	textSizeSpinBox->setPrefix("Text size : ");
	QLabel* colorLabel = new QLabel("Color :");
	colorHolder = new QPushButton();
	setAviTextColor(textColor);

	layout411->addWidget(textSizeSpinBox);
	layout411->addWidget(colorLabel);
	layout411->addWidget(colorHolder);

	// mainLayout->addLayout(layout42);
	// mainLayout->addWidget(m_onlyFirstImage);
	mainLayout->addLayout(layout43);
	// mainLayout->addLayout(layout44);
	// mainLayout->addLayout(layout45);
	mainLayout->addLayout(layout46);
	mainLayout->addLayout(layout47);
	mainLayout->addLayout(layout48);
	mainLayout->addLayout(layout49);
	mainLayout->addLayout(layout410);
	mainLayout->addLayout(layout411);

//	mainLayout->addWidget(m_progressBar);
//	mainLayout->addWidget(processwatcher_rawtoaviprocess);
//	mainLayout->addWidget(pushbutton_rawtoaviDisplay);
//	mainLayout->addWidget(pushbutton_rawtoaviRun);

//	connect(pushbutton_rawtoaviRun, SIGNAL(clicked()), this, SLOT(trt_rawToAviRun()));
//	connect(pushbutton_rawtoaviDisplay, SIGNAL(clicked()), this, SLOT(trt_rawToAviDisplay()));
//
	connect(spinboxvideoScale, SIGNAL(valueChanged(double)), this, SLOT(trt_videoScaleChanged(double)));
	connect(spinboxRgb1TOrigin, SIGNAL(valueChanged(int)), this, SLOT(trt_rgb1TOriginChanged(int)));
	connect(spinboxRgb1TStep, SIGNAL(valueChanged(int)), this, SLOT(trt_rgb1TStepChanged(int)));
	connect(checkboxRgb1IsReversed, SIGNAL(stateChanged(int)), this, SLOT(trt_rgb1IsReversedChanged(int)));
	connect(spinboxFps, SIGNAL(valueChanged(int)), this, SLOT(trt_fpsChanged(int)));
	connect(spinboxFirstIso, SIGNAL(valueChanged(int)), this, SLOT(trt_firstIsoChanged(int)));
	connect(spinboxLastIso, SIGNAL(valueChanged(int)), this, SLOT(trt_lastIsoChanged(int)));
	connect(directionComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(trt_directionChanged(int)));
	connect(sectionPositionSlider, SIGNAL(valueChanged(int)), this, SLOT(trt_sectionIndexChanged(int)));
	connect(sectionPositionSpinBox, SIGNAL(valueChanged(int)), this, SLOT(trt_sectionIndexChanged(int)));
	connect(textSizeSpinBox, SIGNAL(valueChanged(int)), this, SLOT(trt_textSizeChanged(int)));
	connect(colorHolder, SIGNAL(clicked()), this, SLOT(trt_changeAviTextColor()));
//	connect(processwatcher_rawtoaviprocess, &ProcessWatcherWidget::processEnded, this, &RawToAviWidget::updateRGBD);
//	connect(m_pushButtonAttributDirectory, SIGNAL(clicked()), this, SLOT(trt_attributDirectoryOpen()));

}



AviParamWidget::~AviParamWidget()
{

}

void AviParamWidget::setVideoScale(double val)
{
	spinboxvideoScale->setValue(val);
}

double  AviParamWidget::getVideoScale()
{
	return spinboxvideoScale->value();
}

QVariant AviParamWidget::directionCurrentData()
{
	return directionComboBox->currentData(Qt::UserRole);
}


int AviParamWidget::getSectionPosition()
{
	return sectionPositionSpinBox->value();
}

int AviParamWidget::getSectionPositionMinimum()
{
	return sectionPositionSpinBox->minimum();
}

int AviParamWidget::getSectionPositionSingleStep()
{
	return sectionPositionSpinBox->singleStep();
}

bool AviParamWidget::getOnlyFirstImage()
{
	return m_onlyFirstImage->isChecked();
}

void AviParamWidget::trt_rgb1TOriginChanged(int val) {
	rgb1TOrigin = val;
}

void AviParamWidget::trt_rgb1TStepChanged(int val) {
	rgb1TStep = val;
}

void AviParamWidget::trt_rgb1IsReversedChanged(int state) {
	rgb1IsReversed = state == Qt::Checked;
}

void AviParamWidget::trt_fpsChanged(int val) {
	framePerSecond = val;
}

void AviParamWidget::trt_firstIsoChanged(int val) {
	firstIso = val;
}

void AviParamWidget::trt_lastIsoChanged(int val) {
	lastIso = val;
}

void AviParamWidget::trt_textSizeChanged(int val) {
	textSize = val;
}

int AviParamWidget::getFirstIso()
{
	return firstIso;
}

int AviParamWidget::getLastIso()
{
	return lastIso;
}


int getRgtStep()
{

}

bool getRgtreverse()
{

}


void AviParamWidget::setAviTextColor(const QColor& color) {
	textColor = color;
	colorHolder->setStyleSheet(QString("QPushButton{ color: %1; background-color: %1; }").arg(color.name()));
}

void AviParamWidget::trt_changeAviTextColor() {
	QColorDialog dialog(textColor);
	int errCode = dialog.exec();
	if (errCode==QDialog::Accepted) {
		setAviTextColor(dialog.selectedColor());
	}
}


void AviParamWidget::initSectionSlider()
{
	trt_directionChanged(directionComboBox->currentIndex());
}

void AviParamWidget::trt_directionChanged(int newComboBoxIndex) {
	QString seismicPath = m_videoParamWidget->getSeismicPath();
	if (seismicPath.compare("") == 0) {
		return;
	}

	QVariant dirVariant = directionComboBox->itemData(newComboBoxIndex, Qt::UserRole);
	bool ok;
	int direction = dirVariant.toInt(&ok);
	if (!ok) {
		return;
	}

	int size[3];
	double steps[3];
	double origins[3];
	bool isValid = m_videoParamWidget->getPropertiesFromDatasetPath(seismicPath, size, steps, origins);
	if (!isValid) {
		return;
	}

	// inline
	int axisIndex = 2;
	if (direction == 1) {
		// xline
		axisIndex = 0;
	}
	int axisStart = origins[axisIndex];
	int axisStep = steps[axisIndex];
	int axisEnd = axisStart + (size[axisIndex]-1) * axisStep;
	int pos = (axisStart+axisEnd)/2;

	QSignalBlocker bSlider(sectionPositionSlider);
	QSignalBlocker bSpinBox(sectionPositionSpinBox);
	sectionPositionSlider->setMinimum(axisStart);
	sectionPositionSlider->setMaximum(axisEnd);
	sectionPositionSlider->setSingleStep(axisStep);
	sectionPositionSlider->setValue(pos);

	sectionPositionSpinBox->setMinimum(axisStart);
	sectionPositionSpinBox->setMaximum(axisEnd);
	sectionPositionSpinBox->setSingleStep(axisStep);
	sectionPositionSpinBox->setValue(pos);
}

void AviParamWidget::trt_sectionIndexChanged(int sectionIndex) {
	trt_directionChanged(directionComboBox->currentIndex());
	QSignalBlocker bSlider(sectionPositionSlider);
	QSignalBlocker bSpinBox(sectionPositionSpinBox);

	sectionPositionSlider->setValue(sectionIndex);
	sectionPositionSpinBox->setValue(sectionIndex);
	firstInit = 1;
}

