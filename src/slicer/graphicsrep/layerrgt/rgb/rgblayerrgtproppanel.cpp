#include "rgblayerrgtproppanel.h"

#include <QDebug>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QSlider>
#include <QSpinBox>
#include <QCheckBox>
#include <QSpacerItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QLineEdit>
#include <QLabel>
#include <QFormLayout>
#include <QComboBox>
#include <QStringListModel>
#include <QListView>

#include <iostream>
#include <sstream>

#include "rgbpalettewidget.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "rgblayerrgtrep.h"
#include "rgblayerslice.h"
#include "LayerSlice.h"

RGBLayerRGTPropPanel::RGBLayerRGTPropPanel(RGBLayerRGTRep *rep, QWidget *parent) :
		QWidget(parent) {
	m_rep = rep;
	QVBoxLayout *processLayout = new QVBoxLayout(this);
	//processLayout->setMargin(0);
	processLayout->setContentsMargins(0,0,0,0);

	//Palettes
	m_palette = new RGBPaletteWidget(this);
	processLayout->addWidget(m_palette, 0, Qt::AlignmentFlag::AlignTop);
	m_palette->setPaletteHolders(m_rep->rgbLayerSlice()->image()->holders());
	m_palette->setOpacity(m_rep->rgbLayerSlice()->image()->opacity());

	//Connect the image update
	connect(m_palette, SIGNAL(rangeChanged(unsigned int ,const QVector2D & )),
			m_rep->rgbLayerSlice()->image(),
			SLOT(setRange(unsigned int ,const QVector2D & )));
	connect(m_palette, SIGNAL(opacityChanged(float)),
			m_rep->rgbLayerSlice()->image(), SLOT(setOpacity(float)));


	connect(m_palette, SIGNAL(rangeChanged(unsigned int ,const QVector2D & )),
			this,
			SLOT(setRange(unsigned int ,const QVector2D & )));

	connect(m_rep->rgbLayerSlice()->image(),
			SIGNAL(rangeChanged(unsigned int ,const QVector2D & )), m_palette,
			SLOT(setRange(unsigned int ,const QVector2D & )));
	connect(m_rep->rgbLayerSlice()->image(), SIGNAL(opacityChanged(float)),
			m_palette, SLOT(setOpacity(float)));

	QLabel* useMinValueLabel = new QLabel("Value min");
	m_valueMinCheckBox = new QCheckBox();
	m_valueMinCheckBox->setCheckState((m_rep->rgbLayerSlice()->isMinimumValueActive())? Qt::Checked : Qt::Unchecked);

	connect(m_valueMinCheckBox, &QCheckBox::stateChanged, this, &RGBLayerRGTPropPanel::changeMinActivated);

	QHBoxLayout* useMinLayout = new QHBoxLayout;
	useMinLayout->addWidget(useMinValueLabel);
	useMinLayout->addWidget(m_valueMinCheckBox);
	processLayout->addLayout(useMinLayout);

	m_valueMinSlider = new QSlider(Qt::Horizontal);
	m_valueMinSlider->setMinimum(0);
	m_valueMinSlider->setMaximum(100);
	m_valueMinSlider->setValue(std::floor(m_rep->rgbLayerSlice()->minimumValue()*100));

	connect(m_valueMinSlider, &QSlider::valueChanged, this, &RGBLayerRGTPropPanel::changeMinSlider);

	m_valueMinSpinBox = new QDoubleSpinBox();
	m_valueMinSpinBox->setMinimum(0);
	m_valueMinSpinBox->setMaximum(1);
	m_valueMinSpinBox->setSingleStep(0.01);
	m_valueMinSpinBox->setValue(m_rep->rgbLayerSlice()->minimumValue());

	connect(m_valueMinSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &RGBLayerRGTPropPanel::changeMinSpinBox);

	QHBoxLayout* minValueLayout = new QHBoxLayout;
	minValueLayout->addWidget(m_valueMinSlider);
	minValueLayout->addWidget(m_valueMinSpinBox);
	processLayout->addLayout(minValueLayout);

	if (!m_rep->rgbLayerSlice()->isMinimumValueActive()) {
		m_valueMinSlider->hide();
		m_valueMinSpinBox->hide();
	}

	QHBoxLayout* lockLayout = new QHBoxLayout;
	processLayout->addLayout(lockLayout);

	lockLayout->addWidget(new QLabel("Lock ranges : "));
	m_lockCheckBox = new QCheckBox;
	lockLayout->addWidget(m_lockCheckBox);

	connect(m_lockCheckBox, &QCheckBox::stateChanged, this, &RGBLayerRGTPropPanel::lockStateChange);

	//Colors
	processLayout->addWidget(createFreqChooserWidget());

	updateSpectrum();

	//Listen to data modification
	connect(m_rep->rgbLayerSlice(), SIGNAL(frequencyChanged()), this,
			SLOT(frequencyChanged()));
	connect(m_rep->rgbLayerSlice()->layerSlice(), &LayerSlice::computationFinished, this,
			&RGBLayerRGTPropPanel::updateFromComputation);
	connect(m_rep->rgbLayerSlice(), &RGBLayerSlice::lockChanged, this,
			&RGBLayerRGTPropPanel::updateFromDataLock);

	connect(m_rep->rgbLayerSlice(), &RGBLayerSlice::minimumValueActivated, this,
			&RGBLayerRGTPropPanel::minActivated);
	connect(m_rep->rgbLayerSlice(), &RGBLayerSlice::minimumValueChanged, this,
			&RGBLayerRGTPropPanel::minValueChanged);

	updateFromDataLock();
}

RGBLayerRGTPropPanel::~RGBLayerRGTPropPanel() {
}

void RGBLayerRGTPropPanel::updateFromComputation() {
	if (m_rep->rgbLayerSlice()->redIndex()-2!=m_redSpin->value()) {
		QSignalBlocker b1(m_redSpin);
		QSignalBlocker b2(m_redSlider);
		long redIndex = m_rep->rgbLayerSlice()->redIndex()-2;
		m_redSlider->setValue(redIndex);

		m_redSpin->setValue(redIndex);
		m_redLineEdit->setText(m_rep->rgbLayerSlice()->layerSlice()->getLabelFromPosition(redIndex+2));
		//m_redLineEdit->setText(QString::number(m_rep->rgbLayerSlice()->layerSlice()->getFrequency(redIndex), 'f', 0));
	}
	if (m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()!=m_redSpin->maximum()+3) {
		QSignalBlocker b1(m_redSpin);
		QSignalBlocker b2(m_redSlider);
		m_redSlider->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-3);

		m_redSpin->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-3);
	}
	if (m_rep->rgbLayerSlice()->blueIndex()-2!=m_blueSpin->value()) {
		QSignalBlocker b1(m_blueSpin);
		QSignalBlocker b2(m_blueSlider);
		long blueIndex = m_rep->rgbLayerSlice()->blueIndex()-2;
		m_blueSlider->setValue(blueIndex);

		m_blueSpin->setValue(blueIndex);
		m_blueLineEdit->setText(m_rep->rgbLayerSlice()->layerSlice()->getLabelFromPosition(blueIndex+2));
		//m_blueLineEdit->setText(QString::number(m_rep->rgbLayerSlice()->layerSlice()->getFrequency(blueIndex), 'f', 0));
	}
	if (m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()!=m_blueSpin->maximum()+3) {
		QSignalBlocker b1(m_blueSpin);
		QSignalBlocker b2(m_blueSlider);
		m_blueSlider->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-3);

		m_blueSpin->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-3);
	}
	if (m_rep->rgbLayerSlice()->redIndex()-2!=m_greenSpin->value()) {
		QSignalBlocker b1(m_greenSpin);
		QSignalBlocker b2(m_greenSlider);
		long greenIndex = m_rep->rgbLayerSlice()->greenIndex()-2;
		m_greenSlider->setValue(greenIndex);

		m_greenSpin->setValue(greenIndex);
		m_greenLineEdit->setText(m_rep->rgbLayerSlice()->layerSlice()->getLabelFromPosition(greenIndex+2));
		//m_greenLineEdit->setText(QString::number(m_rep->rgbLayerSlice()->layerSlice()->getFrequency(greenIndex), 'f', 0));
	}
	if (m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()!=m_greenSpin->maximum()+3) {
		QSignalBlocker b1(m_greenSpin);
		QSignalBlocker b2(m_greenSlider);
		m_greenSlider->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-3);

		m_greenSpin->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-3);
	}
}

void RGBLayerRGTPropPanel::frequencyChanged() {
	m_oldDeltaRed = m_rep->rgbLayerSlice()->greenIndex() - m_rep->rgbLayerSlice()->redIndex();
	m_oldDeltaBlue = m_rep->rgbLayerSlice()->blueIndex() - m_rep->rgbLayerSlice()->greenIndex();
	updateSpinValue(m_rep->rgbLayerSlice()->redIndex()-2, m_redSlider, m_redSpin, m_redLineEdit);
	updateSpinValue(m_rep->rgbLayerSlice()->greenIndex()-2, m_greenSlider,
			m_greenSpin, m_greenLineEdit);
	updateSpinValue(m_rep->rgbLayerSlice()->blueIndex()-2, m_blueSlider,
			m_blueSpin, m_blueLineEdit);
}

QWidget* RGBLayerRGTPropPanel::createSlideSpinBox(QString title, QSlider *slider,
		QSpinBox *spin, QLineEdit* lineEdit) {
	QGroupBox *sliderBox = new QGroupBox(title, this);
	sliderBox->setContentsMargins(0, 0, 0, 0);

	createlinkedSliderSpin(sliderBox, slider, spin, lineEdit);

	return sliderBox;
}

QWidget* RGBLayerRGTPropPanel::createFreqChooserWidget() {
	//Frequency chooser
	QWidget *colorWidget = new QWidget(this);
	QFormLayout *colorLayout = new QFormLayout(colorWidget);

	QLabel *r = new QLabel("Red");
	r->setPixmap(QIcon(":/palette/icons/red.png").pixmap(QSize(16, 16)));
	QWidget *redWidget = new QWidget(this);
	m_redSlider = new QSlider(Qt::Orientation::Horizontal, this);
	m_redSpin = new QSpinBox();
	m_redLineEdit = new QLineEdit();
	createlinkedSliderSpin(redWidget, m_redSlider, m_redSpin, m_redLineEdit);

	colorLayout->addRow(r, redWidget);

	QLabel *g = new QLabel("Green");
	g->setPixmap(QIcon(":/palette/icons/green.png").pixmap(QSize(16, 16)));
	QWidget *greenWidget = new QWidget(this);
	m_greenSlider = new QSlider(Qt::Orientation::Horizontal, this);
	m_greenSpin = new QSpinBox();
	m_greenLineEdit = new QLineEdit();
	createlinkedSliderSpin(greenWidget, m_greenSlider, m_greenSpin, m_greenLineEdit);
	colorLayout->addRow(g, greenWidget);

	QLabel *b = new QLabel("Green");
	b->setPixmap(QIcon(":/palette/icons/blue.png").pixmap(QSize(16, 16)));
	QWidget *blueWidget = new QWidget(this);
	m_blueSlider = new QSlider(Qt::Orientation::Horizontal, this);
	m_blueSpin = new QSpinBox();
	m_blueLineEdit = new QLineEdit();
	createlinkedSliderSpin(blueWidget, m_blueSlider, m_blueSpin, m_blueLineEdit);
	colorLayout->addRow(b, blueWidget);

	connect(m_redSlider, SIGNAL(valueChanged(int)), this,
			SLOT(setRedIndex(int)));
	connect(m_redSpin, SIGNAL(valueChanged(int)), this, SLOT(setRedIndex(int)));

	connect(m_greenSlider, SIGNAL(valueChanged(int)), this,
			SLOT(setGreenIndex(int)));
	connect(m_greenSpin, SIGNAL(valueChanged(int)), this,
			SLOT(setGreenIndex(int)));

	connect(m_blueSlider, SIGNAL(valueChanged(int)), this,
			SLOT(setBlueIndex(int)));
	connect(m_blueSpin, SIGNAL(valueChanged(int)), this,
			SLOT(setBlueIndex(int)));

	return colorWidget;
}

void RGBLayerRGTPropPanel::updatePalette(int i) {
	m_palette->setPaletteHolder(i, m_rep->rgbLayerSlice()->image()->get(i));
}

void RGBLayerRGTPropPanel::updateSpinValue(int value, QSlider *slider, QSpinBox *spin, QLineEdit* lineEdit) {
	QSignalBlocker b1(slider);
	QSignalBlocker b2(spin);
	slider->setValue(value);
	spin->setValue(value);
	lineEdit->setText(m_rep->rgbLayerSlice()->layerSlice()->getLabelFromPosition(value+2));
	//lineEdit->setText(QString::number(m_rep->rgbLayerSlice()->layerSlice()->getFrequency(value), 'f', 0));
}

void RGBLayerRGTPropPanel::setRedIndex(int value) {
	m_oldDeltaRed = m_rep->rgbLayerSlice()->greenIndex()-2 - value;
	m_rep->rgbLayerSlice()->setRedIndex(value+2);
}
void RGBLayerRGTPropPanel::setGreenIndex(int value) {
	// red
	int deltaRed = m_oldDeltaRed;
	int valRed = value-deltaRed;
	if (value<0) {
			valRed = value;
	} else if(value-deltaRed<0) {
			valRed = 0;
	} else if(value-deltaRed>m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-2) {
			valRed = m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-3;
	}
	m_oldDeltaRed = value - valRed;

	// blue
	int deltaBlue = m_oldDeltaBlue;
	int valBlue = value + deltaBlue;
	if (value<0) {
			valBlue = value;
	} else if(value+deltaBlue>m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-3) {
			valBlue = m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-3;
	} else if(valBlue<0) {
			valBlue = 0;
	}
	m_oldDeltaBlue = valBlue-value;

	// only do that at the end to avoid m_oldDelta... modifications by called slot frequencyUpdated

	m_rep->rgbLayerSlice()->setRGBIndexes(valRed+2, value+2, valBlue+2);

}
void RGBLayerRGTPropPanel::setBlueIndex(int value) {
	m_oldDeltaBlue = value - (m_rep->rgbLayerSlice()->greenIndex()-2);
	m_rep->rgbLayerSlice()->setBlueIndex(value+2);
}

void RGBLayerRGTPropPanel::createlinkedSliderSpin(QWidget *parent, QSlider *slider,
		QSpinBox *spin, QLineEdit *lineEdit) {
	slider->setSingleStep(1);
	slider->setTickInterval(10);
	slider->setMinimum(0);
	slider->setMaximum(1);
	slider->setValue(0);

	spin->setMinimum(0);
	spin->setMaximum(10000);
	spin->setSingleStep(1);
	spin->setValue(0);
	spin->setWrapping(false);

	lineEdit->setEnabled(false);
	// set the character limit on the line edit
	lineEdit->setMaxLength(11);
	QMargins margins = lineEdit->contentsMargins();
	lineEdit->setMaximumWidth(QFontMetrics(lineEdit->font()).maxWidth() * lineEdit->maxLength() + margins.left() + margins.right());

	QHBoxLayout *hBox = new QHBoxLayout(parent);
	//hBox->setMargin(0);
	hBox->setContentsMargins(0,0,0,0);
	hBox->addWidget(slider);
	hBox->addWidget(spin);
	hBox->addWidget(lineEdit);
}

void RGBLayerRGTPropPanel::updateSliderSpin(int min, int max, QSlider *slider,
		QSpinBox *spin) {
	slider->setMaximum(max);
	slider->setMinimum(min);
	slider->setValue(min);

	spin->setMaximum(max);
	spin->setMinimum(min);
	spin->setValue(min);
}

void RGBLayerRGTPropPanel::updateSpectrum() {
	QSignalBlocker redBlocker(m_redSlider);
	QSignalBlocker red1Blocker(m_redSpin);
	QSignalBlocker greenBlocker(m_greenSlider);
	QSignalBlocker green1Blocker(m_greenSpin);
	QSignalBlocker blueBlocker(m_blueSlider);
	QSignalBlocker blue1Blocker(m_blueSpin);

	//Frequency chooser
	int freqCount = m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices() - 2;

	QStringList fList;
	fList << "iso" << "seismic";
	for (int i = 0; i < freqCount; i++)
		fList << "f" + locale().toString(i);

	updateSliderSpin(0, freqCount, m_redSlider, m_redSpin);
	updateSliderSpin(0, freqCount, m_greenSlider, m_greenSpin);
	updateSliderSpin(0, freqCount, m_blueSlider, m_blueSpin);

	int redIndex = m_rep->rgbLayerSlice()->redIndex()-2;
	m_redSlider->setValue(redIndex);
	m_redSpin->setValue(redIndex);
	m_redLineEdit->setText(m_rep->rgbLayerSlice()->layerSlice()->getLabelFromPosition(redIndex+2));
	//m_redLineEdit->setText(QString::number(m_rep->rgbLayerSlice()->layerSlice()->getFrequency(redIndex), 'f', 0));

	int greenIndex = m_rep->rgbLayerSlice()->greenIndex()-2;
	m_greenSlider->setValue(greenIndex);
	m_greenSpin->setValue(greenIndex);
	m_greenLineEdit->setText(m_rep->rgbLayerSlice()->layerSlice()->getLabelFromPosition(greenIndex+2));
	//m_greenLineEdit->setText(QString::number(m_rep->rgbLayerSlice()->layerSlice()->getFrequency(greenIndex), 'f', 0));

	int blueIndex = m_rep->rgbLayerSlice()->blueIndex()-2;
	m_blueSlider->setValue(blueIndex);
	m_blueSpin->setValue(blueIndex);
	m_blueLineEdit->setText(m_rep->rgbLayerSlice()->layerSlice()->getLabelFromPosition(blueIndex+2));
	//m_blueLineEdit->setText(QString::number(m_rep->rgbLayerSlice()->layerSlice()->getFrequency(blueIndex), 'f', 0));

	m_oldDeltaRed = greenIndex - redIndex;
	m_oldDeltaBlue = blueIndex - greenIndex;
}

void RGBLayerRGTPropPanel::updateFromDataLock() {
	QSignalBlocker b1(m_lockCheckBox);
	m_lockCheckBox->setCheckState((m_rep->rgbLayerSlice()->isLocked()) ? Qt::Checked: Qt::Unchecked);

	bool enableState = !m_rep->rgbLayerSlice()->isLocked();

	m_palette->setEnabled(enableState);
	m_redSlider->setEnabled(enableState);
	m_redSpin->setEnabled(enableState);
	m_greenSlider->setEnabled(enableState);
	m_greenSpin->setEnabled(enableState);
	m_blueSlider->setEnabled(enableState);
	m_blueSpin->setEnabled(enableState);

	m_valueMinCheckBox->setEnabled(enableState);
	m_valueMinSlider->setEnabled(enableState);
	m_valueMinSpinBox->setEnabled(enableState);
}

void RGBLayerRGTPropPanel::lockStateChange(int state) {
	RGBLayerSlice* rgb = m_rep->rgbLayerSlice();
	if (state==Qt::Checked) {
		rgb->lock();
	} else {
		rgb->unlock();
	}
}

void RGBLayerRGTPropPanel::setRange(unsigned int channel, const QVector2D& range) {
	if (m_rep->rgbLayerSlice()->isLocked()) {
		if (channel==0) {
			m_rep->rgbLayerSlice()->setLockedRedRange(range);
		} else if (channel==1) {
			m_rep->rgbLayerSlice()->setLockedGreenRange(range);
		} else if (channel==2) {
			m_rep->rgbLayerSlice()->setLockedBlueRange(range);
		}
	}
}

void RGBLayerRGTPropPanel::changeMinActivated(int state) {
	m_rep->rgbLayerSlice()->setMinimumValueActive(state==Qt::Checked);
}

void RGBLayerRGTPropPanel::changeMinSlider(int value) {
	m_rep->rgbLayerSlice()->setMinimumValue(value/100.0f);
}

void RGBLayerRGTPropPanel::changeMinSpinBox(double value) {
	m_rep->rgbLayerSlice()->setMinimumValue(value);
}

void RGBLayerRGTPropPanel::minActivated(bool activated) {
	QSignalBlocker b(m_valueMinCheckBox);
	m_valueMinCheckBox->setCheckState(activated ? Qt::Checked : Qt::Unchecked);

	if (activated) {
		m_valueMinSlider->show();
		m_valueMinSpinBox->show();
	} else {
		m_valueMinSlider->hide();
		m_valueMinSpinBox->hide();
	}
}

void RGBLayerRGTPropPanel::minValueChanged(float minValue) {
	QSignalBlocker bSlider(m_valueMinSlider);
	QSignalBlocker bSpinBox(m_valueMinSpinBox);

	m_valueMinSlider->setValue(std::floor(minValue*100));
	m_valueMinSpinBox->setValue(minValue);
}

