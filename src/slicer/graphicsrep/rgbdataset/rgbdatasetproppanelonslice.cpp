#include "rgbdatasetproppanelonslice.h"
#include "rgbdatasetreponslice.h"
#include "rgbdataset.h"

#include "seismic3dabstractdataset.h"
#include "workingsetmanager.h"
#include "folderdata.h"
#include "seismicsurvey.h"
#include "palettewidget.h"
#include "cudaimagepaletteholder.h"
#include "rgbcomputationondataset.h"

#include <QVBoxLayout>
#include <QGridLayout>
#include <QFormLayout>
#include <QComboBox>
#include <QSlider>
#include <QSpinBox>
#include <QLabel>
#include <QGroupBox>
#include <QTabWidget>
#include <QLineEdit>
#include <cmath>

RgbDatasetPropPanelOnSlice::RgbDatasetPropPanelOnSlice(RgbDatasetRepOnSlice* rep, QWidget* parent) : QWidget(parent) {
	m_rep = rep;
	initAllDatasets();

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	QGridLayout* grid = new QGridLayout;
	mainLayout->addLayout(grid);

	m_redComboBox = new QComboBox;
	m_redChannelSpinBox = new QSpinBox;
	setupChannel("Red", m_redComboBox, m_redChannelSpinBox, grid, 0);

	m_greenComboBox = new QComboBox;
	m_greenChannelSpinBox = new QSpinBox;
	setupChannel("Green", m_greenComboBox, m_greenChannelSpinBox, grid, 1);

	m_blueComboBox = new QComboBox;
	m_blueChannelSpinBox = new QSpinBox;
	setupChannel("Blue", m_blueComboBox, m_blueChannelSpinBox, grid, 2);

	m_alphaComboBox = new QComboBox;
	m_alphaChannelSpinBox = new QSpinBox;
	m_alphaComboBox->addItem("None", RgbDataset::NONE);
	m_alphaComboBox->addItem("Transparent with RGB deltas", RgbDataset::TRANSPARENT);
	m_alphaComboBox->addItem("Opaque with RGB deltas", RgbDataset::OPAQUE);
	setupChannel("Alpha", m_alphaComboBox, m_alphaChannelSpinBox, grid, 3);

	initSynchroneSliders();
	mainLayout->addWidget(m_synchroneSliderGroupBox);

	m_transparencyGroupBox = new QGroupBox("Transparency");
	QVBoxLayout* transparencyLayout = new QVBoxLayout;
	m_transparencyGroupBox->setLayout(transparencyLayout);
	mainLayout->addWidget(m_transparencyGroupBox);

	QHBoxLayout* constantTransparencyLayout = new QHBoxLayout;
	transparencyLayout->addLayout(constantTransparencyLayout);

	m_constantAlphaLabel = new QLabel("Alpha");
	constantTransparencyLayout->addWidget(m_constantAlphaLabel);

	m_constantAlphaSlider = new QSlider(Qt::Horizontal);
	m_constantAlphaSlider->setSingleStep(1);
	m_constantAlphaSlider->setTickInterval(10);
	m_constantAlphaSlider->setMinimum(0);
	m_constantAlphaSlider->setMaximum(100);
	m_constantAlphaSlider->setValue(m_rep->rgbDataset()->constantAlpha()*100);
	constantTransparencyLayout->addWidget(m_constantAlphaSlider);

	QHBoxLayout* radiusTransparencyLayout = new QHBoxLayout;
	transparencyLayout->addLayout(radiusTransparencyLayout);

	m_radiusAlphaLabel = new QLabel("Radius");
	radiusTransparencyLayout->addWidget(m_radiusAlphaLabel);

	m_radiusAlphaSlider = new QSlider(Qt::Horizontal);
	m_radiusAlphaSlider->setSingleStep(1);
	m_radiusAlphaSlider->setTickInterval(10);
	m_radiusAlphaSlider->setMinimum(0);
	m_radiusAlphaSlider->setMaximum(100);
	m_radiusAlphaSlider->setValue(m_rep->rgbDataset()->radiusAlpha()*100/std::sqrt(6));
	radiusTransparencyLayout->addWidget(m_radiusAlphaSlider);

	QTabWidget* paletteTabWidget = new QTabWidget;
	mainLayout->addWidget(paletteTabWidget);

	//palette
	m_redPalette = new PaletteWidget;
	paletteTabWidget->addTab(m_redPalette, "Red");

	m_redPalette->setLookupTable(m_rep->red()->lookupTable());
	m_redPalette->setOpacity(m_rep->red()->opacity());

	//Connect the image update
	connect(m_redPalette, SIGNAL(rangeChanged(const QVector2D &)), m_rep->red(),
			SLOT(setRange(const QVector2D &)));
	connect(m_redPalette, SIGNAL(opacityChanged(float)), m_rep->red(),
			SLOT(setOpacity(float)));
	connect(m_redPalette, SIGNAL(lookupTableChanged(const LookupTable &)),
			m_rep->red(), SLOT(setLookupTable(const LookupTable &)));

	m_redPalette->setPaletteHolder(m_rep->red());

	//palette
	m_greenPalette = new PaletteWidget;
	paletteTabWidget->addTab(m_greenPalette, "Green");

	m_greenPalette->setLookupTable(m_rep->green()->lookupTable());
	m_greenPalette->setOpacity(m_rep->green()->opacity());

	//Connect the image update
	connect(m_greenPalette, SIGNAL(rangeChanged(const QVector2D &)), m_rep->green(),
			SLOT(setRange(const QVector2D &)));
	connect(m_greenPalette, SIGNAL(opacityChanged(float)), m_rep->green(),
			SLOT(setOpacity(float)));
	connect(m_greenPalette, SIGNAL(lookupTableChanged(const LookupTable &)),
			m_rep->green(), SLOT(setLookupTable(const LookupTable &)));

	m_greenPalette->setPaletteHolder(m_rep->green());

	//palette
	m_bluePalette = new PaletteWidget;
	paletteTabWidget->addTab(m_bluePalette, "Blue");

	m_bluePalette->setLookupTable(m_rep->blue()->lookupTable());
	m_bluePalette->setOpacity(m_rep->blue()->opacity());

	//Connect the image update
	connect(m_bluePalette, SIGNAL(rangeChanged(const QVector2D &)), m_rep->blue(),
			SLOT(setRange(const QVector2D &)));
	connect(m_bluePalette, SIGNAL(opacityChanged(float)), m_rep->blue(),
			SLOT(setOpacity(float)));
	connect(m_bluePalette, SIGNAL(lookupTableChanged(const LookupTable &)),
			m_rep->blue(), SLOT(setLookupTable(const LookupTable &)));

	m_bluePalette->setPaletteHolder(m_rep->blue());


	switch (m_rep->rgbDataset()->alphaMode()) {
	case RgbDataset::NONE:
		m_alphaComboBox->setCurrentIndex(0);
		m_radiusAlphaLabel->hide();
		m_radiusAlphaSlider->hide();
		break;
	case RgbDataset::TRANSPARENT:
		m_alphaComboBox->setCurrentIndex(1);
		m_constantAlphaLabel->hide();
		m_constantAlphaSlider->hide();
		break;
	case RgbDataset::OPAQUE:
		m_alphaComboBox->setCurrentIndex(2);
		m_constantAlphaLabel->hide();
		m_constantAlphaSlider->hide();
		break;
	default:
		if (m_alphaComboBox->count()>3) {
			m_alphaComboBox->setCurrentIndex(3);
			m_transparencyGroupBox->hide();
		} else {
			m_alphaComboBox->setCurrentIndex(0);
			m_radiusAlphaLabel->hide();
			m_radiusAlphaSlider->hide();
		}
		break;
	}

	connect(m_redComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &RgbDatasetPropPanelOnSlice::redChangedInternal);
	connect(m_redChannelSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &RgbDatasetPropPanelOnSlice::redChannelChangedInternal);

	connect(m_greenComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &RgbDatasetPropPanelOnSlice::greenChangedInternal);
	connect(m_greenChannelSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &RgbDatasetPropPanelOnSlice::greenChannelChangedInternal);

	connect(m_blueComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &RgbDatasetPropPanelOnSlice::blueChangedInternal);
	connect(m_blueChannelSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &RgbDatasetPropPanelOnSlice::blueChannelChangedInternal);

	connect(m_alphaComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &RgbDatasetPropPanelOnSlice::alphaChangedInternal);
	connect(m_alphaChannelSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &RgbDatasetPropPanelOnSlice::alphaChannelChangedInternal);

	connect(m_constantAlphaSlider, &QSlider::valueChanged, this, &RgbDatasetPropPanelOnSlice::constantAlphaChangedInternal);
	connect(m_radiusAlphaSlider, &QSlider::valueChanged, this, &RgbDatasetPropPanelOnSlice::radiusAlphaChangedInternal);

	connect(m_seismicSurvey, &SeismicSurvey::datasetAdded, this, &RgbDatasetPropPanelOnSlice::dataAdded);
	connect(m_seismicSurvey, &SeismicSurvey::datasetRemoved, this, &RgbDatasetPropPanelOnSlice::dataRemoved);

	updateSynchroneSliders();
}

RgbDatasetPropPanelOnSlice::~RgbDatasetPropPanelOnSlice() {

}

void RgbDatasetPropPanelOnSlice::updatePalette() {
	m_redPalette->setPaletteHolder(m_rep->red());
	m_greenPalette->setPaletteHolder(m_rep->green());
	m_bluePalette->setPaletteHolder(m_rep->blue());
}

void RgbDatasetPropPanelOnSlice::redChanged() {
}

void RgbDatasetPropPanelOnSlice::greenChanged() {

}

void RgbDatasetPropPanelOnSlice::blueChanged() {

}

void RgbDatasetPropPanelOnSlice::alphaChanged() {

}

void RgbDatasetPropPanelOnSlice::constantAlphaChanged() {

}

void RgbDatasetPropPanelOnSlice::radiusAlphaChanged() {

}

void RgbDatasetPropPanelOnSlice::redChangedInternal(int index) {
	bool ok;
	int listIndex = m_redComboBox->itemData(index).toInt(&ok);
	if (ok && listIndex>=0) {
		//m_blue = m_allDatasets[listIndex];
		m_redChannelSpinBox->setMaximum(m_allDatasets[listIndex]->dimV()-1);
	} else {
		//m_blue = nullptr;
		m_redChannelSpinBox->setMaximum(0);
	}

	if (m_redChannelSpinBox->maximum()>0) {
		m_redChannelSpinBox->show();
	} else {
		m_redChannelSpinBox->hide();
	}
}

void RgbDatasetPropPanelOnSlice::greenChangedInternal(int index) {
	bool ok;
	int listIndex = m_greenComboBox->itemData(index).toInt(&ok);
	if (ok && listIndex>=0) {
		//m_blue = m_allDatasets[listIndex];
		m_greenChannelSpinBox->setMaximum(m_allDatasets[listIndex]->dimV()-1);
	} else {
		//m_blue = nullptr;
		m_greenChannelSpinBox->setMaximum(0);
	}

	if (m_greenChannelSpinBox->maximum()>0) {
		m_greenChannelSpinBox->show();
	} else {
		m_greenChannelSpinBox->hide();
	}
}

void RgbDatasetPropPanelOnSlice::blueChangedInternal(int index) {
	bool ok;
	int listIndex = m_blueComboBox->itemData(index).toInt(&ok);
	if (ok && listIndex>=0) {
		//m_blue = m_allDatasets[listIndex];
		m_blueChannelSpinBox->setMaximum(m_allDatasets[listIndex]->dimV()-1);
	} else {
		//m_blue = nullptr;
		m_blueChannelSpinBox->setMaximum(0);
	}

	if (m_blueChannelSpinBox->maximum()>0) {
		m_blueChannelSpinBox->show();
	} else {
		m_blueChannelSpinBox->hide();
	}
}

void RgbDatasetPropPanelOnSlice::alphaChangedInternal(int index) {
	bool ok;
	int listIndex = m_alphaComboBox->itemData(index).toInt(&ok);
	if (ok && listIndex>=0) {
		//m_alpha = m_allDatasets[listIndex];
		m_alphaChannelSpinBox->setMaximum(m_allDatasets[listIndex]->dimV()-1);

		m_constantAlphaLabel->hide();
		m_constantAlphaSlider->hide();
		m_radiusAlphaLabel->hide();
		m_radiusAlphaSlider->hide();
		m_transparencyGroupBox->hide();
		m_rep->rgbDataset()->setAlphaMode(RgbDataset::OTHER);
	} else {
		//m_alpha = nullptr;
		m_alphaChannelSpinBox->setMaximum(0);

		if (listIndex==RgbDataset::NONE) {
			m_constantAlphaLabel->show();
			m_constantAlphaSlider->show();
			m_radiusAlphaLabel->hide();
			m_radiusAlphaSlider->hide();
			m_transparencyGroupBox->show();
			m_rep->rgbDataset()->setAlphaMode(RgbDataset::NONE);
		} else if (listIndex==RgbDataset::TRANSPARENT || listIndex==RgbDataset::OPAQUE) {
			m_constantAlphaLabel->hide();
			m_constantAlphaSlider->hide();
			m_radiusAlphaLabel->show();
			m_radiusAlphaSlider->show();
			m_transparencyGroupBox->show();
			if (listIndex==RgbDataset::TRANSPARENT) {
				m_rep->rgbDataset()->setAlphaMode(RgbDataset::TRANSPARENT);
			} else {
				m_rep->rgbDataset()->setAlphaMode(RgbDataset::OPAQUE);
			}
		}
	}

	if (m_alphaChannelSpinBox->maximum()>0) {
		m_alphaChannelSpinBox->show();
	} else {
		m_alphaChannelSpinBox->hide();
	}
}

void RgbDatasetPropPanelOnSlice::redChannelChangedInternal(int index) {

}

void RgbDatasetPropPanelOnSlice::greenChannelChangedInternal(int index) {

}

void RgbDatasetPropPanelOnSlice::blueChannelChangedInternal(int index) {

}

void RgbDatasetPropPanelOnSlice::alphaChannelChangedInternal(int index) {
	QVariant variant = m_alphaComboBox->itemData(index);
	bool ok;

	int val = variant.toInt(&ok);
	if (ok) {
		if (val==RgbDataset::NONE) {
			m_rep->rgbDataset()->setAlphaMode(RgbDataset::NONE);
		} else if (val==RgbDataset::TRANSPARENT) {
			m_rep->rgbDataset()->setAlphaMode(RgbDataset::TRANSPARENT);
		} else if (val==RgbDataset::OPAQUE) {
			m_rep->rgbDataset()->setAlphaMode(RgbDataset::OPAQUE);
		}
	}
}

void RgbDatasetPropPanelOnSlice::constantAlphaChangedInternal(int value) {
	m_rep->rgbDataset()->setConstantAlpha(value/100.0);
}

void RgbDatasetPropPanelOnSlice::radiusAlphaChangedInternal(int value) {
	m_rep->rgbDataset()->setRadiusAlpha(value/100.0 * std::sqrt(6));
}

void RgbDatasetPropPanelOnSlice::setupChannel(const QString& name, QComboBox* comboBox, QSpinBox* spinBox, QGridLayout* gridLayout, int row) {
	QLabel* labelRed = new QLabel(name);
	gridLayout->addWidget(labelRed, row, 0);

	fillComboBox(comboBox);
	gridLayout->addWidget(comboBox, row, 1);

	spinBox->setMinimum(0);
	if (m_allDatasets.size()>0) {
		spinBox->setMaximum(m_allDatasets[0]->dimV()-1);
	} else {
		spinBox->setMaximum(0);
	}
	spinBox->setValue(0);
	gridLayout->addWidget(spinBox, row, 2);

	if (spinBox->maximum()==0) {
		spinBox->hide();
	}
}

void RgbDatasetPropPanelOnSlice::fillComboBox(QComboBox* comboBox) {
	for (std::pair<std::size_t, Seismic3DAbstractDataset*> pair : m_allDatasets) {
		Seismic3DAbstractDataset* dataset = pair.second;
		comboBox->addItem(dataset->name(), QVariant((int)pair.first));
	}
}

void RgbDatasetPropPanelOnSlice::initAllDatasets() {
	const QList<IData*>& surveyDatas = m_rep->data()->workingSetManager()->folders().seismics->data();

	for (IData* surveyData : surveyDatas) {
		if (SeismicSurvey* survey = dynamic_cast<SeismicSurvey*>(surveyData)) {
			m_seismicSurvey = survey;
			for (Seismic3DAbstractDataset* dataset : survey->datasets()) {
				m_allDatasets[nextIndex()] = dataset;
			}
		}
	}
}

void RgbDatasetPropPanelOnSlice::dataAdded(Seismic3DAbstractDataset* data) {
	int nextIndex = this->nextIndex();
	m_allDatasets[nextIndex] = data;
	m_redComboBox->addItem(data->name(), QVariant(nextIndex));
	m_greenComboBox->addItem(data->name(), QVariant(nextIndex));
	m_blueComboBox->addItem(data->name(), QVariant(nextIndex));
	m_alphaComboBox->addItem(data->name(), QVariant(nextIndex));
}

void RgbDatasetPropPanelOnSlice::dataRemoved(Seismic3DAbstractDataset* data) {
	std::map<std::size_t, Seismic3DAbstractDataset*>::iterator it = std::find_if(m_allDatasets.begin(), m_allDatasets.end(), [data](const std::pair<std::size_t, Seismic3DAbstractDataset*>& v) {
		return data==v.second;
	});
	if (it!=m_allDatasets.end()) {
		int index = it->first;

		removeDataFromComboBox(m_redComboBox, m_redChannelSpinBox, index);
		removeDataFromComboBox(m_greenComboBox, m_greenChannelSpinBox, index);
		removeDataFromComboBox(m_blueComboBox, m_blueChannelSpinBox, index);
		removeDataFromComboBox(m_alphaComboBox, m_alphaChannelSpinBox, index);
		m_allDatasets.erase(it);
	}
}

bool RgbDatasetPropPanelOnSlice::removeDataFromComboBox(QComboBox* comboBox, QSpinBox* spinBox, int key) {
	int index = 0;
	bool notFound = true;
	while (notFound && index<comboBox->count()) {
		QVariant variant = comboBox->itemData(index);
		bool ok;
		int keyFromData = variant.toInt(&ok);
		if (ok) {
			notFound = key!=keyFromData;
		}
		if (notFound) {
			index++;
		}
	}
	if (!notFound) {
		if (comboBox->currentIndex()==index && comboBox->count()>1) {
			comboBox->setCurrentIndex(0);
			// risk that index may change in the process or that it loops back and creates a crash
			// may need to do a signal blocker
			// but do not forget to update spinbox
			comboBox->removeItem(index);
		} else if (comboBox->currentIndex()==index) {
			spinBox->hide();
			comboBox->removeItem(index);
			// TODO set nullptr but red, green and blue are not supposed to be nullptr update objects if mandatory
		} else {
			comboBox->removeItem(index);
		}
	}
	return !notFound;
}

std::size_t RgbDatasetPropPanelOnSlice::nextIndex() const {
	return m_nextIndex++;
}

void RgbDatasetPropPanelOnSlice::createlinkedSliderSpin(QWidget *parent, QSlider *slider,
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
	lineEdit->setMaxLength(4);
	QMargins margins = lineEdit->contentsMargins();
	lineEdit->setMaximumWidth(QFontMetrics(lineEdit->font()).maxWidth() * lineEdit->maxLength() + margins.left() + margins.right());

	QHBoxLayout *hBox = new QHBoxLayout(parent);
	//hBox->setMargin(0);
	hBox->setContentsMargins(0,0,0,0);
	hBox->addWidget(slider);
	hBox->addWidget(spin);
	hBox->addWidget(lineEdit);
}

void RgbDatasetPropPanelOnSlice::updateSynchroneSliders() {
	if (m_rep->rgbDataset()->red()==m_rep->rgbDataset()->green() &&
			m_rep->rgbDataset()->red()==m_rep->rgbDataset()->blue() &&
			dynamic_cast<RgbComputationOnDataset*>(m_rep->rgbDataset()->red())!=nullptr) {
		updateRGBSynchroneSliders();
		m_synchroneSliderGroupBox->show();
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
		connect(m_rep->rgbDataset(), SIGNAL(redChannelChanged()), this,
				SLOT(frequencyChanged()));
		connect(m_rep->rgbDataset(), SIGNAL(greenChannelChanged()), this,
				SLOT(frequencyChanged()));
		connect(m_rep->rgbDataset(), SIGNAL(blueChannelChanged()), this,
				SLOT(frequencyChanged()));
	} else {
		m_synchroneSliderGroupBox->hide();
		disconnect(m_redSlider, SIGNAL(valueChanged(int)), this,
				SLOT(setRedIndex(int)));
		disconnect(m_redSpin, SIGNAL(valueChanged(int)), this, SLOT(setRedIndex(int)));

		disconnect(m_greenSlider, SIGNAL(valueChanged(int)), this,
				SLOT(setGreenIndex(int)));
		disconnect(m_greenSpin, SIGNAL(valueChanged(int)), this,
				SLOT(setGreenIndex(int)));

		disconnect(m_blueSlider, SIGNAL(valueChanged(int)), this,
				SLOT(setBlueIndex(int)));
		disconnect(m_blueSpin, SIGNAL(valueChanged(int)), this,
				SLOT(setBlueIndex(int)));
		disconnect(m_rep->rgbDataset(), SIGNAL(redChannelChanged()), this,
				SLOT(frequencyChanged()));
		disconnect(m_rep->rgbDataset(), SIGNAL(greenChannelChanged()), this,
				SLOT(frequencyChanged()));
		disconnect(m_rep->rgbDataset(), SIGNAL(blueChannelChanged()), this,
				SLOT(frequencyChanged()));
	}
}

void RgbDatasetPropPanelOnSlice::initSynchroneSliders() {
	m_synchroneSliderGroupBox = new QGroupBox;
	QFormLayout *colorLayout = new QFormLayout(m_synchroneSliderGroupBox);

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
}

void RgbDatasetPropPanelOnSlice::setRedIndex(int value) {
	m_oldDeltaRed = m_rep->rgbDataset()->channelGreen() - value;
	m_rep->rgbDataset()->setChannelRed(value);
}

void RgbDatasetPropPanelOnSlice::setGreenIndex(int value) {
	// red
	int deltaRed = m_oldDeltaRed;
	int valRed = value-deltaRed;
	if (value<0) {
			valRed = value;
	} else if(value-deltaRed<0) {
			valRed = 0;
	} else if(value-deltaRed>m_rep->rgbDataset()->red()->dimV()-1) {
			valRed = m_rep->rgbDataset()->red()->dimV()-1;
	}
	m_oldDeltaRed = value - valRed;

	// blue
	int deltaBlue = m_oldDeltaBlue;
	int valBlue = value + deltaBlue;
	if (value<0) {
			valBlue = value;
	} else if(value+deltaBlue>m_rep->rgbDataset()->red()->dimV()-1) {
			valBlue = m_rep->rgbDataset()->red()->dimV()-1;
	} else if(valBlue<0) {
			valBlue = 0;
	}
	m_oldDeltaBlue = valBlue-value;

	// only do that at the end to avoid m_oldDelta... modifications by called slot frequencyUpdated
	m_rep->rgbDataset()->setChannelRed(valRed);
	m_rep->rgbDataset()->setChannelGreen(value);
	m_rep->rgbDataset()->setChannelBlue(valBlue);
}

void RgbDatasetPropPanelOnSlice::setBlueIndex(int value) {
	m_oldDeltaBlue = value - (m_rep->rgbDataset()->channelGreen());
	m_rep->rgbDataset()->setChannelBlue(value);
}

void RgbDatasetPropPanelOnSlice::updateRGBSynchroneSliders() {
	QSignalBlocker redBlocker(m_redSlider);
	QSignalBlocker red1Blocker(m_redSpin);
	QSignalBlocker greenBlocker(m_greenSlider);
	QSignalBlocker green1Blocker(m_greenSpin);
	QSignalBlocker blueBlocker(m_blueSlider);
	QSignalBlocker blue1Blocker(m_blueSpin);

	//Frequency chooser
	int freqCount = m_rep->rgbDataset()->red()->dimV();

	QStringList fList;
	fList << "iso" << "seismic";
	for (int i = 0; i < freqCount; i++)
		fList << "f" + locale().toString(i);

	updateSliderSpin(0, freqCount, m_redSlider, m_redSpin);
	updateSliderSpin(0, freqCount, m_greenSlider, m_greenSpin);
	updateSliderSpin(0, freqCount, m_blueSlider, m_blueSpin);

	RgbComputationOnDataset* dataset = dynamic_cast<RgbComputationOnDataset*>(m_rep->rgbDataset()->red());

	int redIndex = m_rep->rgbDataset()->channelRed();
	m_redSlider->setValue(redIndex);
	m_redSpin->setValue(redIndex);
	if (dataset) {
		m_redLineEdit->setText(QString::number(dataset->getFrequency(redIndex), 'f', 0));
	}

	int greenIndex = m_rep->rgbDataset()->channelGreen();
	m_greenSlider->setValue(greenIndex);
	m_greenSpin->setValue(greenIndex);
	if (dataset) {
		m_greenLineEdit->setText(QString::number(dataset->getFrequency(greenIndex), 'f', 0));
	}

	int blueIndex = m_rep->rgbDataset()->channelBlue();
	m_blueSlider->setValue(blueIndex);
	m_blueSpin->setValue(blueIndex);
	if (dataset) {
		m_blueLineEdit->setText(QString::number(dataset->getFrequency(blueIndex), 'f', 0));
	}

	m_oldDeltaRed = greenIndex - redIndex;
	m_oldDeltaBlue = blueIndex - greenIndex;
}

void RgbDatasetPropPanelOnSlice::frequencyChanged() {
	m_oldDeltaRed = m_rep->rgbDataset()->channelGreen() - m_rep->rgbDataset()->channelRed();
	m_oldDeltaBlue = m_rep->rgbDataset()->channelBlue() - m_rep->rgbDataset()->channelGreen();
	updateSpinValue(m_rep->rgbDataset()->channelRed(), m_redSlider, m_redSpin, m_redLineEdit);
	updateSpinValue(m_rep->rgbDataset()->channelGreen(), m_greenSlider,
			m_greenSpin, m_greenLineEdit);
	updateSpinValue(m_rep->rgbDataset()->channelBlue(), m_blueSlider,
			m_blueSpin, m_blueLineEdit);
}

void RgbDatasetPropPanelOnSlice::updateSpinValue(int value, QSlider *slider, QSpinBox *spin, QLineEdit* lineEdit) {
	QSignalBlocker b1(slider);
	QSignalBlocker b2(spin);
	slider->setValue(value);
	spin->setValue(value);
	RgbComputationOnDataset* dataset = dynamic_cast<RgbComputationOnDataset*>(m_rep->rgbDataset()->red());
	if (dataset!=nullptr) {
		lineEdit->setText(QString::number(dataset->getFrequency(value), 'f', 0));
	}
}

void RgbDatasetPropPanelOnSlice::updateSliderSpin(int min, int max, QSlider *slider,
		QSpinBox *spin) {
	slider->setMaximum(max);
	slider->setMinimum(min);
	slider->setValue(min);

	spin->setMaximum(max);
	spin->setMinimum(min);
	spin->setValue(min);
}
