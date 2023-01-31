#include "fixedlayersfromdatasetandcubeproppanelonrandom.h"

#include <QDebug>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QSlider>
#include <QSpinBox>
#include <QSpacerItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QLineEdit>
#include <QLabel>
#include <QSpinBox>
#include <QFormLayout>
#include <QComboBox>
#include <QStringListModel>
#include <QListView>
#include <QComboBox>
#include <QToolButton>
#include <QCoreApplication>
#include <QProgressBar>

#include <iostream>
#include <sstream>

#include "cudaimagepaletteholder.h"
#include "cpuimagepaletteholder.h"
#include "fixedlayersfromdatasetandcubereponrandom.h"
#include "fixedlayersfromdatasetandcube.h"
#include "editingspinbox.h"
#include "subgridgetterdialog.h"

FixedLayersFromDatasetAndCubePropPanelOnRandom::FixedLayersFromDatasetAndCubePropPanelOnRandom(
		FixedLayersFromDatasetAndCubeRepOnRandom *rep, QWidget *parent) : QWidget(parent) {
	m_rep = rep;
	QVBoxLayout *processLayout = new QVBoxLayout(this);
	//processLayout->setMargin(0);
	processLayout->setContentsMargins(0,0,0,0);

	QWidget* modeHolder = new QWidget;
	QHBoxLayout* modeLayout = new QHBoxLayout;
	modeHolder->setLayout(modeLayout);
	processLayout->addWidget(modeHolder);
	modeLayout->addWidget(new QLabel("Mode"));

	m_modeComboBox = new QComboBox;
	modeLayout->addWidget(m_modeComboBox);
	m_modeComboBox->addItem("Read");
	m_modeComboBox->addItem("Cache");
	if (m_rep->fixedLayersFromDataset()->mode()==FixedLayersFromDatasetAndCube::READ) {
		m_modeComboBox->setCurrentIndex(0);
	} else {
		m_modeComboBox->setCurrentIndex(1);
	}

	connect(m_modeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &FixedLayersFromDatasetAndCubePropPanelOnRandom::modeChangedInternal);

	QWidget* holder = new QWidget;
	QVBoxLayout* formLayout = new QVBoxLayout(holder);

	QLabel* label = new QLabel("Layers");
	m_slider = new QSlider(Qt::Orientation::Horizontal);
	m_slider->setMinimum(0);
	m_slider->setMaximum(m_rep->fixedLayersFromDataset()->numLayers()-1);
	m_slider->setValue(m_rep->fixedLayersFromDataset()->currentImageIndex());
	m_slider->setTickInterval(1);
	m_slider->setSingleStep(1);
	m_slider->setTracking(false);

	QToolButton* lessButton = new QToolButton();
	lessButton->setArrowType(Qt::LeftArrow);
	lessButton->setAutoRepeat(true);
	lessButton->setAutoRepeatDelay(1000);
	lessButton->setAutoRepeatInterval(250);

	QToolButton* moreButton = new QToolButton();
	moreButton->setArrowType(Qt::RightArrow);
	moreButton->setAutoRepeat(true);
	moreButton->setAutoRepeatDelay(1000);
	moreButton->setAutoRepeatInterval(250);

	m_playButton = new QToolButton();
	m_playButton->setIcon(style()->standardPixmap( QStyle::SP_MediaPlay ));

	m_loopButton = new QToolButton();
	m_loopButton->setCheckable(true);
	m_loopButton->setIcon(style()->standardPixmap( QStyle::SP_BrowserReload ));

	connect(m_slider, &QSlider::valueChanged, this, &FixedLayersFromDatasetAndCubePropPanelOnRandom::changeDataKeyFromSlider);

	long val0 = m_rep->fixedLayersFromDataset()->isoOrigin();
	long val1 = val0 + (m_rep->fixedLayersFromDataset()->numLayers()-1) * m_rep->fixedLayersFromDataset()->isoStep();

	m_layerNameSpinBox = new EditingSpinBox;
	m_layerNameSpinBox->setMinimum(std::min(val0, val1));
	m_layerNameSpinBox->setMaximum(std::max(val0, val1));
	m_layerNameSpinBox->setSingleStep(std::abs(m_rep->fixedLayersFromDataset()->isoStep()));
	m_layerNameSpinBox->setValue(m_rep->fixedLayersFromDataset()->isoOrigin() +
			m_rep->fixedLayersFromDataset()->currentImageIndex()*m_rep->fixedLayersFromDataset()->isoStep());

	connect(m_layerNameSpinBox, &EditingSpinBox::contentUpdated, this, &FixedLayersFromDatasetAndCubePropPanelOnRandom::changeDataKeyFromSpinBox,
			Qt::QueuedConnection);

	m_multiplierComboBox = new QComboBox;
	m_multiplierComboBox->addItem("x1", 1);
	m_multiplierComboBox->addItem("x2", 2);
	m_multiplierComboBox->addItem("x5", 5);
	m_multiplierComboBox->addItem("x10", 10);
	m_multiplierComboBox->addItem("x15", 15);
	m_multiplierComboBox->addItem("x20", 20);

	connect(m_multiplierComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &FixedLayersFromDatasetAndCubePropPanelOnRandom::multiplierChanged);

	QWidget* holderLayer = new QWidget;
	QHBoxLayout* holderLayerLayout = new QHBoxLayout;
	holderLayer->setLayout(holderLayerLayout);

	QWidget* holderLayerLv2 = new QWidget;
	QHBoxLayout* holderLayerLayoutLv2 = new QHBoxLayout;
	holderLayerLv2->setLayout(holderLayerLayoutLv2);

	holderLayerLayout->addWidget(label);
	holderLayerLayout->addWidget(lessButton, 0);
	holderLayerLayout->addWidget(m_slider);
	holderLayerLayout->addWidget(moreButton, 0);
	holderLayerLayout->addWidget(m_playButton, 0);
	holderLayerLayout->addWidget(m_loopButton, 0);
	holderLayerLayoutLv2->addWidget(m_layerNameSpinBox, 0, Qt::AlignmentFlag::AlignLeft);
	holderLayerLayoutLv2->addWidget(m_multiplierComboBox, 0, Qt::AlignmentFlag::AlignLeft);

	formLayout->addWidget(holderLayer);
	formLayout->addWidget(holderLayerLv2);

	processLayout->addWidget(holder);
	processLayout->addWidget(modeHolder);

	m_progressBar = new QProgressBar;
	processLayout->addWidget(m_progressBar);
	m_progressBar->hide();

	connect(m_rep->fixedLayersFromDataset()->image(), &CPUImagePaletteHolder::dataChanged, this,
			[this]() {
		QSignalBlocker b(m_slider);
		QSignalBlocker b1(m_layerNameSpinBox);

		m_slider->setValue(m_rep->fixedLayersFromDataset()->currentImageIndex());
		m_layerNameSpinBox->setValue(m_rep->fixedLayersFromDataset()->isoOrigin()+
				m_rep->fixedLayersFromDataset()->currentImageIndex()*m_rep->fixedLayersFromDataset()->isoStep());
	});

	connect(m_rep->fixedLayersFromDataset(), &FixedLayersFromDatasetAndCube::modeChanged, this,
			&FixedLayersFromDatasetAndCubePropPanelOnRandom::modeChanged);

	connect(lessButton, &QToolButton::clicked, [this]() {
		if (m_rep->fixedLayersFromDataset()->getIsoStep()>0) {
			m_layerNameSpinBox->stepDown();
		} else {
			m_layerNameSpinBox->stepUp();
		}
	});

	connect(moreButton, &QToolButton::clicked, [this]() {
		if (m_rep->fixedLayersFromDataset()->getIsoStep()>0) {
			m_layerNameSpinBox->stepUp();
		} else {
			m_layerNameSpinBox->stepDown();
		}
	});

	connect(m_playButton, &QToolButton::clicked, [this]() {
		if(m_rep->fixedLayersFromDataset()->modePlay())
			m_playButton->setIcon(style()->standardPixmap( QStyle::SP_MediaPlay));
		else
			m_playButton->setIcon(style()->standardPixmap( QStyle::SP_MediaPause));

		bool looping =m_loopButton->isChecked();
		int coef = m_multiplierComboBox->currentData().toInt();
		m_rep->fixedLayersFromDataset()->play(250, coef,looping);
	});

	connect(m_rep->fixedLayersFromDataset(), &FixedLayersFromDatasetAndCube::initProgressBar, this,
			&FixedLayersFromDatasetAndCubePropPanelOnRandom::initProgressBar);
	connect(m_rep->fixedLayersFromDataset(), &FixedLayersFromDatasetAndCube::valueProgressBarChanged, this,
			&FixedLayersFromDatasetAndCubePropPanelOnRandom::valueProgressBarChanged);
	connect(m_rep->fixedLayersFromDataset(), &FixedLayersFromDatasetAndCube::endProgressBar, this,
			&FixedLayersFromDatasetAndCubePropPanelOnRandom::endProgressBar);

	modeChanged(); // finish init
}

FixedLayersFromDatasetAndCubePropPanelOnRandom::~FixedLayersFromDatasetAndCubePropPanelOnRandom() {
}

void FixedLayersFromDatasetAndCubePropPanelOnRandom::changeDataKeyFromSlider(long index) {
	FixedLayersFromDatasetAndCube* data = m_rep->fixedLayersFromDataset();
	if (data->mode()==FixedLayersFromDatasetAndCube::CACHE) {
		long min = std::min(data->cacheFirstIndex(), data->cacheLastIndex());
		long max = std::max(data->cacheFirstIndex(), data->cacheLastIndex());
		long val = index - min;
		long modifiedVal = val - (val % data->cacheStepIndex());
		index = modifiedVal + min;
	}

	data->setCurrentImageIndex(index);
	QSignalBlocker b(m_layerNameSpinBox);
	m_layerNameSpinBox->setValue(data->isoOrigin()+
			index*data->isoStep());
}

void FixedLayersFromDatasetAndCubePropPanelOnRandom::changeDataKeyFromSpinBox() {
	long indexIso = m_layerNameSpinBox->value();
	long index = (indexIso - m_rep->fixedLayersFromDataset()->isoOrigin()) / m_rep->fixedLayersFromDataset()->isoStep();
	QSignalBlocker b(m_slider);
	m_rep->fixedLayersFromDataset()->setCurrentImageIndex(index);
	m_slider->setValue(index);
}

void FixedLayersFromDatasetAndCubePropPanelOnRandom::multiplierChanged(int index) {
	bool ok;
	m_stepMultiplier = m_multiplierComboBox->itemData(index).toInt(&ok);
	if (!ok) {
		m_stepMultiplier = 1;
	}
	m_layerNameSpinBox->setSingleStep(m_stepMultiplier * std::abs(m_rep->fixedLayersFromDataset()->isoStep()));
}

void FixedLayersFromDatasetAndCubePropPanelOnRandom::modeChangedInternal(int index) {
	FixedLayersFromDatasetAndCube::Mode mode = FixedLayersFromDatasetAndCube::READ;
	if (index==1) {
		mode = FixedLayersFromDatasetAndCube::CACHE;
	}
	if (mode!=m_rep->fixedLayersFromDataset()->mode()) {
		if (mode==FixedLayersFromDatasetAndCube::READ) {
			m_rep->fixedLayersFromDataset()->moveToReadMode();
		} else {
			// get begin, end, step
			FixedLayersFromDatasetAndCube* data = m_rep->fixedLayersFromDataset();
			SubGridGetterDialog dialog(data->isoOrigin(), data->isoOrigin()+std::max((std::size_t)0, data->numLayers()-1)*data->isoStep(), data->isoStep());
			dialog.activateMemoryCost(data->cacheLayerMemoryCost()); // RGB in uchar + iso as short
			bool result = dialog.exec()==QDialog::Accepted;

			if (result) {
				long begin = dialog.outBegin();
				long end = dialog.outEnd();
				long step = dialog.outStep();
				result = m_rep->fixedLayersFromDataset()->moveToCacheMode(begin, end, step);
				if (!result) {
					m_modeComboBox->setCurrentIndex(0);// return to read mode
				}
			}
		}
	}
}

void FixedLayersFromDatasetAndCubePropPanelOnRandom::modeChanged() {
	FixedLayersFromDatasetAndCube* data = m_rep->fixedLayersFromDataset();
	if (data->mode()==FixedLayersFromDatasetAndCube::READ) {
		QSignalBlocker b1(m_slider);
		m_slider->setMinimum(0);
		m_slider->setMaximum(data->numLayers()-1);
		m_slider->setTickInterval(1);
		m_slider->setSingleStep(1);
		m_slider->setTracking(false);

		long val0 = data->isoOrigin();
		long val1 = val0 + (data->numLayers()-1) * data->isoStep();

		QSignalBlocker b2(m_layerNameSpinBox);
		m_layerNameSpinBox->setMinimum(std::min(val0, val1));
		m_layerNameSpinBox->setMaximum(std::max(val0, val1));
		m_layerNameSpinBox->setSingleStep(std::abs(data->isoStep()));

		QSignalBlocker b3(m_modeComboBox);
		m_modeComboBox->setCurrentIndex(0);

		//no need to change the value
	} else {
		QSignalBlocker b1(m_slider);
		long min = std::min(data->cacheFirstIndex(), data->cacheLastIndex());
		long max = std::max(data->cacheFirstIndex(), data->cacheLastIndex());
		m_slider->setMinimum(min);
		m_slider->setMaximum(max);
		m_slider->setTickInterval(std::abs(data->cacheStepIndex()));
		m_slider->setSingleStep(std::abs(data->cacheStepIndex()));
		m_slider->setTracking(true);

		long val0 = data->isoOrigin() + data->cacheFirstIndex() * data->isoStep();
		long val1 = data->isoOrigin() + data->cacheLastIndex() * data->isoStep();

		QSignalBlocker b2(m_layerNameSpinBox);
		m_layerNameSpinBox->setMinimum(std::min(val0, val1));
		m_layerNameSpinBox->setMaximum(std::max(val0, val1));
		m_layerNameSpinBox->setSingleStep(std::abs(data->isoStep()*data->cacheStepIndex()));

		long currentIndex = data->currentImageIndex();
		if (currentIndex<min || currentIndex>max || (currentIndex-min)%data->cacheStepIndex()!=0) {
			QCoreApplication::processEvents(); // to process all mode changed events
			if (currentIndex==data->currentImageIndex()) {//only check if it did not change to avoid a loop
				long newIndex;
				min = std::min(data->cacheFirstIndex(), data->cacheLastIndex());// redo to avoid change issues on the way
				max = std::max(data->cacheFirstIndex(), data->cacheLastIndex());// redo to avoid change issues on the way
				if (newIndex<min) {
					newIndex = min;
				} else if (newIndex>max) {
					newIndex = max;
				} else {
					newIndex = ((currentIndex - min) / std::abs(data->cacheLastIndex())) * std::abs(data->cacheLastIndex()) + min;
				}
				data->setCurrentImageIndex(newIndex);
			}
		}

		QSignalBlocker b3(m_modeComboBox);
		m_modeComboBox->setCurrentIndex(1);
	}
}

void FixedLayersFromDatasetAndCubePropPanelOnRandom::initProgressBar(int min, int max, int val) {
	m_progressBar->setRange(min, max);
	m_progressBar->setValue(val);
	m_progressBar->show();
}

void FixedLayersFromDatasetAndCubePropPanelOnRandom::valueProgressBarChanged(int val) {
	m_progressBar->setValue(val);
}

void FixedLayersFromDatasetAndCubePropPanelOnRandom::endProgressBar() {
	m_progressBar->hide();
}

