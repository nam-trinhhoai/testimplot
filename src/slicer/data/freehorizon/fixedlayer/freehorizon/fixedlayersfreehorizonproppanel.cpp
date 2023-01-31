#include "fixedlayersfreehorizonproppanel.h"
#include "fixedlayersfromdatasetandcubeproppanel.h"

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
#include <QPushButton>
#include <QToolButton>
#include <QFormLayout>
#include <QComboBox>
#include <QStringListModel>
#include <QListView>
#include <QComboBox>
#include <QCheckBox>
#include <QCoreApplication>
#include <QMutexLocker>
#include <QProgressBar>

#include <iostream>
#include <sstream>

#include "palettewidget.h"
#include "cudaimagepaletteholder.h"
#include "cpuimagepaletteholder.h"
#include "fixedlayersfromdatasetandcuberep.h"
#include "fixedlayersfromdatasetandcube.h"
#include "editingspinbox.h"
#include "subgridgetterdialog.h"

FixedLayersFreeHorizonPropPanel::FixedLayersFreeHorizonPropPanel(FixedLayersFromDatasetAndCubeRep *rep, QWidget *parent) :
FixedLayersFromDatasetAndCubePropPanel(rep, parent, false) {
	m_rep = rep;
	QVBoxLayout *processLayout = new QVBoxLayout(this);
	processLayout->setContentsMargins(0,0,0,0);

	QWidget* modeHolder = new QWidget;
	QHBoxLayout* modeLayout = new QHBoxLayout;
	modeHolder->setLayout(modeLayout);
	// processLayout->addWidget(modeHolder);
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

	connect(m_modeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &FixedLayersFreeHorizonPropPanel::modeChangedInternal);

	//Palettes
	m_palette = new PaletteWidget(this);
	processLayout->addWidget(m_palette, 0, Qt::AlignmentFlag::AlignTop);
	m_palette->setPaletteHolder(m_rep->fixedLayersFromDataset()->image());
	m_palette->setOpacity(m_rep->fixedLayersFromDataset()->image()->opacity());

	//Connect the image update
	connect(m_palette, SIGNAL(rangeChanged(const QVector2D & )),
			m_rep->fixedLayersFromDataset()->image(),
			SLOT(setRange(const QVector2D & )));
	connect(m_palette, SIGNAL(opacityChanged(float)),
			m_rep->fixedLayersFromDataset()->image(), SLOT(setOpacity(float)));
	connect(m_palette, SIGNAL(lookupTableChanged(const LookupTable &)),
			m_rep->fixedLayersFromDataset()->image(),
			SLOT(setLookupTable(const LookupTable &)));

	connect(m_rep->fixedLayersFromDataset()->image(),
			SIGNAL(rangeChanged(const QVector2D & )), m_palette,
			SLOT(setRange(const QVector2D & )));
	connect(m_rep->fixedLayersFromDataset()->image(), SIGNAL(opacityChanged(float)),
			m_palette, SLOT(setOpacity(float)));
	connect(m_rep->fixedLayersFromDataset()->image(),
			SIGNAL(lookupTableChanged(const LookupTable &)), m_palette,
			SLOT(setLookupTable(const LookupTable &)));

	m_lockPalette = new QCheckBox("Lock Palette");
	updateLockCheckBox();
	processLayout->addWidget(m_lockPalette, 0, Qt::AlignmentFlag::AlignTop);

	if ( m_rep->fixedLayersFromDataset()->enableScaleSlider() )
	{
		m_nbreScales = m_rep->fixedLayersFromDataset()->getNbreGccScales();
		QLabel *scaleLabel = new QLabel("scale:");
		m_scaleSlider = new QSlider(Qt::Orientation::Horizontal);
		m_scaleSlider->setMinimum(0);
		m_scaleSlider->setMaximum(m_nbreScales);
		m_scaleSlider->setValue(0);
		m_scaleSlider->setTickInterval(1);
		m_scaleSlider->setSingleStep(1);
		m_scaleSlider->setTracking(false);
		m_scaleSpinBox = new QSpinBox(this);
		m_scaleSpinBox->setMinimum(0);
		m_scaleSpinBox->setMaximum(m_nbreScales);
		m_scaleSpinBox->setValue(0);

		QHBoxLayout *qhbScale = new QHBoxLayout;
		qhbScale->addWidget(scaleLabel);
		qhbScale->addWidget(m_scaleSlider);
		qhbScale->addWidget(m_scaleSpinBox);
		processLayout->addLayout(qhbScale);
		connect(m_scaleSlider, SIGNAL(valueChanged(int)), this, SLOT(scaleChange(int)));
		connect(m_scaleSpinBox, SIGNAL(valueChanged(int)), this, SLOT(scaleSpinChange(int)));
	}

	//Colors
	QWidget* holder = new QWidget;
	QVBoxLayout* formLayout = new QVBoxLayout(holder);

	QLabel* label = new QLabel("Layers xxx");
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


	connect(m_slider, &QSlider::valueChanged, this, &FixedLayersFreeHorizonPropPanel::changeDataKeyFromSlider);

	long val0 = m_rep->fixedLayersFromDataset()->isoOrigin();
	long val1 = val0 + (m_rep->fixedLayersFromDataset()->numLayers()-1) * m_rep->fixedLayersFromDataset()->isoStep();

	m_layerNameSpinBox = new EditingSpinBox;
	m_layerNameSpinBox->setMinimum(std::min(val0, val1));
	m_layerNameSpinBox->setMaximum(std::max(val0, val1));
	m_layerNameSpinBox->setSingleStep(std::abs(m_rep->fixedLayersFromDataset()->isoStep()));
	m_layerNameSpinBox->setValue(m_rep->fixedLayersFromDataset()->isoOrigin() +
			m_rep->fixedLayersFromDataset()->currentImageIndex()*m_rep->fixedLayersFromDataset()->isoStep());

	connect(m_layerNameSpinBox, &EditingSpinBox::contentUpdated, this, &FixedLayersFreeHorizonPropPanel::changeDataKeyFromSpinBox,
			Qt::QueuedConnection);

	m_multiplierComboBox = new QComboBox;
	m_multiplierComboBox->setMinimumWidth(70);
		m_multiplierComboBox->setMaximumWidth(70);
	m_multiplierComboBox->addItem("x1", 1);
	m_multiplierComboBox->addItem("x2", 2);
	m_multiplierComboBox->addItem("x5", 5);
	m_multiplierComboBox->addItem("x10", 10);
	m_multiplierComboBox->addItem("x15", 15);
	m_multiplierComboBox->addItem("x20", 20);

	connect(m_multiplierComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &FixedLayersFreeHorizonPropPanel::multiplierChanged);

	QHBoxLayout* simplifyMeshLayout = new QHBoxLayout;
	QLabel* labelsimplify = new QLabel("Simplify mesh steps");

	m_simplifyStepsSpinBox = new QSpinBox;
	m_simplifyStepsSpinBox->setMinimum(1);
	m_simplifyStepsSpinBox->setMaximum(100);
	m_simplifyStepsSpinBox->setValue(m_rep->fixedLayersFromDataset()->getSimplifyMeshSteps());

	connect(m_simplifyStepsSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), m_rep->fixedLayersFromDataset(), &FixedLayersFromDatasetAndCube::setSimplifyMeshSteps);

	connect(m_rep->fixedLayersFromDataset(),  &FixedLayersFromDatasetAndCube::simplifyMeshStepsChanged, m_simplifyStepsSpinBox, &QSpinBox::setValue);

	simplifyMeshLayout->addWidget(labelsimplify);
	simplifyMeshLayout->addWidget(m_simplifyStepsSpinBox);

	QWidget* holderLayer = new QWidget;
	QHBoxLayout* holderLayerLayout = new QHBoxLayout;
	holderLayerLayout->setContentsMargins(0,0,0,0);
	holderLayer->setLayout(holderLayerLayout);

	QWidget* holderLayerLv2 = new QWidget;
	QHBoxLayout* holderLayerLayoutLv2 = new QHBoxLayout;
	holderLayerLv2->setLayout(holderLayerLayoutLv2);

	holderLayerLayout->addWidget(label);
	holderLayerLayout->addWidget(m_layerNameSpinBox, 0, Qt::AlignmentFlag::AlignLeft);
	holderLayerLayout->addWidget(m_multiplierComboBox, 0, Qt::AlignmentFlag::AlignLeft);
	holderLayerLayoutLv2->addWidget(lessButton, 0);
	holderLayerLayoutLv2->addWidget(m_slider);
	holderLayerLayoutLv2->addWidget(moreButton, 0);
	holderLayerLayoutLv2->addWidget(m_playButton, 0);
	holderLayerLayoutLv2->addWidget(m_loopButton, 0);
//	holderLayerLayoutLv2->addWidget(m_layerNameSpinBox, 0, Qt::AlignmentFlag::AlignLeft);
//	holderLayerLayoutLv2->addWidget(m_multiplierComboBox, 0, Qt::AlignmentFlag::AlignLeft);

	formLayout->addWidget(holderLayer);
	formLayout->addWidget(holderLayerLv2);

	// processLayout->addWidget(holder);
	// processLayout->addWidget(modeHolder);

	m_progressBar = new QProgressBar;
	// processLayout->addWidget(m_progressBar);
	// processLayout->addLayout(simplifyMeshLayout);
	m_progressBar->hide();

	connect(m_rep->fixedLayersFromDataset()->image(), &CUDAImagePaletteHolder::dataChanged, this,
			[this]() {
		QSignalBlocker b(m_slider);
		QSignalBlocker b2(m_layerNameSpinBox);
		m_slider->setValue(m_rep->fixedLayersFromDataset()->currentImageIndex());
		m_layerNameSpinBox->setValue(m_rep->fixedLayersFromDataset()->isoOrigin() +
				m_rep->fixedLayersFromDataset()->currentImageIndex()*m_rep->fixedLayersFromDataset()->isoStep());
	});

	connect(m_rep->fixedLayersFromDataset(), &FixedLayersFromDatasetAndCube::modeChanged, this,
			&FixedLayersFreeHorizonPropPanel::modeChanged);

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
			&FixedLayersFreeHorizonPropPanel::initProgressBar);
	connect(m_rep->fixedLayersFromDataset(), &FixedLayersFromDatasetAndCube::valueProgressBarChanged, this,
			&FixedLayersFreeHorizonPropPanel::valueProgressBarChanged);
	connect(m_rep->fixedLayersFromDataset(), &FixedLayersFromDatasetAndCube::endProgressBar, this,
			&FixedLayersFreeHorizonPropPanel::endProgressBar);

	connect(m_lockPalette, &QCheckBox::stateChanged, this,
			&FixedLayersFreeHorizonPropPanel::lockPalette);
	connect(m_palette, &PaletteWidget::rangeChanged,
				this, &FixedLayersFreeHorizonPropPanel::updateLockRange);
	connect(m_palette, &PaletteWidget::lookupTableChanged,
				this, &FixedLayersFreeHorizonPropPanel::updateLockLookupTable);

	connect(m_rep->fixedLayersFromDataset()->image(),
			QOverload<const QVector2D&>::of(&CPUImagePaletteHolder::rangeChanged), this,
			&FixedLayersFreeHorizonPropPanel::updateLockRange);
	connect(m_rep->fixedLayersFromDataset()->image(),
			QOverload<const LookupTable&>::of(&CPUImagePaletteHolder::lookupTableChanged), this,
			&FixedLayersFreeHorizonPropPanel::updateLockLookupTable);

	modeChanged(); // finish init
}

FixedLayersFreeHorizonPropPanel::~FixedLayersFreeHorizonPropPanel() {
}

void FixedLayersFreeHorizonPropPanel::changeDataKeyFromSlider(long index) {
	if (m_mutex.tryLock()) {
		{
			QMutexLocker locker(&m_mutexIndexList);
			m_sliderIndexList.clear();// to remove leftovers
		}
		bool goOn = true;
		long nextIndex = index;
		while(goOn) {
			changeDataKeyFromSliderInternal(nextIndex);
			QCoreApplication::processEvents();
			QMutexLocker locker(&m_mutexIndexList);
			goOn = m_sliderIndexList.count()>0;
			if (goOn) {
				nextIndex = m_sliderIndexList.last();
				m_sliderIndexList.clear();// to remove leftovers
			}
		}
		m_mutex.unlock();
	} else {
		QMutexLocker locker(&m_mutexIndexList);
		m_sliderIndexList.append(index);
	}
}

void FixedLayersFreeHorizonPropPanel::changeDataKeyFromSliderInternal(long index) {
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

void FixedLayersFreeHorizonPropPanel::changeDataKeyFromSpinBox() {
	long indexIso = m_layerNameSpinBox->value();
	long index = (indexIso - m_rep->fixedLayersFromDataset()->isoOrigin()) / m_rep->fixedLayersFromDataset()->isoStep();
	QSignalBlocker b(m_slider);
	m_rep->fixedLayersFromDataset()->setCurrentImageIndex(index);
	m_slider->setValue(index);
}

void FixedLayersFreeHorizonPropPanel::multiplierChanged(int index) {
	bool ok;
	m_stepMultiplier = m_multiplierComboBox->itemData(index).toInt(&ok);
	if (!ok) {
		m_stepMultiplier = 1;
	}
	m_layerNameSpinBox->setSingleStep(m_stepMultiplier * std::abs(m_rep->fixedLayersFromDataset()->isoStep()));
}

void FixedLayersFreeHorizonPropPanel::updatePalette() {
	if (m_palette) {
		m_palette->setPaletteHolder(m_rep->fixedLayersFromDataset()->image());
	}
}

void FixedLayersFreeHorizonPropPanel::modeChangedInternal(int index) {
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

void FixedLayersFreeHorizonPropPanel::modeChanged() {
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

void FixedLayersFreeHorizonPropPanel::initProgressBar(int min, int max, int val) {
	m_progressBar->setRange(min, max);
	m_progressBar->setValue(val);
	m_progressBar->show();
}

void FixedLayersFreeHorizonPropPanel::valueProgressBarChanged(int val) {
	m_progressBar->setValue(val);
}

void FixedLayersFreeHorizonPropPanel::endProgressBar() {
	m_progressBar->hide();
}

void FixedLayersFreeHorizonPropPanel::updateLockCheckBox() {
	bool isPaletteLocked = m_rep->fixedLayersFromDataset()->isPaletteLocked();
	int lockState = (isPaletteLocked) ? Qt::Checked : Qt::Unchecked;

	QSignalBlocker b1(m_lockPalette);
	m_lockPalette->setChecked(lockState);
}

void FixedLayersFreeHorizonPropPanel::lockPalette(int state) {
	FixedLayersFromDatasetAndCube* data = m_rep->fixedLayersFromDataset();
	if (state==Qt::Checked) {
		data->lockPalette(data->image()->range(),
				data->image()->lookupTable());
	} else {
		data->unlockPalette();
	}
}

void FixedLayersFreeHorizonPropPanel::updateLockRange(const QVector2D & range) {
	FixedLayersFromDatasetAndCube* data = m_rep->fixedLayersFromDataset();
	if (data->isPaletteLocked()) {
		data->lockPalette(range, data->lockedLookupTable());
	}
}

void FixedLayersFreeHorizonPropPanel::updateLockLookupTable(const LookupTable& lookupTable) {
	FixedLayersFromDatasetAndCube* data = m_rep->fixedLayersFromDataset();
	if (data->isPaletteLocked()) {
		data->lockPalette(data->lockedRange(), lookupTable);
	}
}

void FixedLayersFreeHorizonPropPanel::scaleChange(int idx)
{
	m_rep->fixedLayersFromDataset()->setGccIndex(idx);
	m_scaleSpinBox->setValue(idx);
}

void FixedLayersFreeHorizonPropPanel::scaleSpinChange(int idx)
{
	m_rep->fixedLayersFromDataset()->setGccIndex(idx);
	m_scaleSlider->setValue(idx);
}
