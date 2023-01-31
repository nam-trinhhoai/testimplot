#include "fixedrgblayersfromdatasetandcubeproppanel.h"

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
#include <QCoreApplication>
#include <QMutexLocker>
#include <QProgressBar>

#include <iostream>
#include <sstream>

#include "rgbpalettewidget.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "fixedrgblayersfromdatasetandcuberep.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "editingspinbox.h"
#include "subgridgetterdialog.h"

FixedRGBLayersFromDatasetAndCubePropPanel::FixedRGBLayersFromDatasetAndCubePropPanel(FixedRGBLayersFromDatasetAndCubeRep *rep, QWidget *parent, bool active) :
		QWidget(parent) {
	m_rep = rep;

	if ( !active ) return;

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
	if (m_rep->fixedRGBLayersFromDataset()->mode()==FixedRGBLayersFromDatasetAndCube::READ) {
		m_modeComboBox->setCurrentIndex(0);
	} else {
		m_modeComboBox->setCurrentIndex(1);
	}

	connect(m_modeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &FixedRGBLayersFromDatasetAndCubePropPanel::modeChangedInternal);

	//Palettes
	if (!m_rep->fixedRGBLayersFromDataset()->useRgb1()) {
		m_palette = new RGBPaletteWidget(this);
		processLayout->addWidget(m_palette, 0, Qt::AlignmentFlag::AlignTop);
		m_palette->setPaletteHolders(m_rep->fixedRGBLayersFromDataset()->image()->holders());
		m_palette->setOpacity(m_rep->fixedRGBLayersFromDataset()->image()->opacity());

		//Connect the image update
		connect(m_palette, SIGNAL(rangeChanged(unsigned int ,const QVector2D & )),
				m_rep->fixedRGBLayersFromDataset()->image(),
				SLOT(setRange(unsigned int ,const QVector2D & )));
		connect(m_palette, SIGNAL(opacityChanged(float)),
				m_rep->fixedRGBLayersFromDataset()->image(), SLOT(setOpacity(float)));

		connect(m_rep->fixedRGBLayersFromDataset()->image(),
				SIGNAL(rangeChanged(unsigned int ,const QVector2D & )), m_palette,
				SLOT(setRange(unsigned int ,const QVector2D & )));
		connect(m_rep->fixedRGBLayersFromDataset()->image(), SIGNAL(opacityChanged(float)),
				m_palette, SLOT(setOpacity(float)));
	} else {
		m_palette = nullptr;


		QHBoxLayout* modeLayout1 = new QHBoxLayout;
		QSlider * m_opacity=new QSlider(Qt::Orientation::Horizontal,this);
			m_opacity->setSingleStep(1);
			m_opacity->setTickInterval(10);
			m_opacity->setMinimum(0);
			m_opacity->setMaximum(100);
			m_opacity->setValue(100);

			QDoubleSpinBox* opacitySpin = new QDoubleSpinBox(this);
			opacitySpin->setMinimum(0);
			opacitySpin->setMaximum(1);
			opacitySpin->setSingleStep(0.01);
			opacitySpin->setDecimals(2);
			opacitySpin->setValue(1);
			connect(m_opacity, &QSlider::valueChanged,[=](int value){
				opacitySpin->setValue(0.01*value);
			});
			connect(opacitySpin,QOverload<double>::of(&QDoubleSpinBox::valueChanged),[=](double value){
				m_opacity->setValue(value*100);
			});

			connect(m_opacity,&QSlider::valueChanged,[=](double value){
				m_rep->fixedRGBLayersFromDataset()->image()->setOpacity(value*0.01f);
						});


			QGroupBox * opacityWidget=new QGroupBox( "Opacity",this);

			QHBoxLayout *hBox=new QHBoxLayout(opacityWidget);
			//hBox->setMargin(0);
			hBox->setContentsMargins(0,0,0,0);
			hBox->addWidget(m_opacity);
			hBox->addWidget(opacitySpin);
			processLayout->addWidget(opacityWidget,0,Qt::AlignmentFlag::AlignTop);

	}

	QLabel* useMinValueLabel = new QLabel("Value min");
	m_valueMinCheckBox = new QCheckBox();
	m_valueMinCheckBox->setCheckState((m_rep->fixedRGBLayersFromDataset()->isMinimumValueActive())? Qt::Checked : Qt::Unchecked);

	connect(m_valueMinCheckBox, &QCheckBox::stateChanged, this, &FixedRGBLayersFromDatasetAndCubePropPanel::changeMinActivated);

	QHBoxLayout* useMinLayout = new QHBoxLayout;
	useMinLayout->addWidget(useMinValueLabel);
	useMinLayout->addWidget(m_valueMinCheckBox);
	processLayout->addLayout(useMinLayout);

	m_valueMinSlider = new QSlider(Qt::Horizontal);
	m_valueMinSlider->setMinimum(0);
	m_valueMinSlider->setMaximum(100);
	m_valueMinSlider->setValue(std::floor(m_rep->fixedRGBLayersFromDataset()->minimumValue()*100));

	connect(m_valueMinSlider, &QSlider::valueChanged, this, &FixedRGBLayersFromDatasetAndCubePropPanel::changeMinSlider);

	m_valueMinSpinBox = new QDoubleSpinBox();
	m_valueMinSpinBox->setMinimum(0);
	m_valueMinSpinBox->setMaximum(1);
	m_valueMinSpinBox->setSingleStep(0.01);
	m_valueMinSpinBox->setValue(m_rep->fixedRGBLayersFromDataset()->minimumValue());

	connect(m_valueMinSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &FixedRGBLayersFromDatasetAndCubePropPanel::changeMinSpinBox);

	QHBoxLayout* minValueLayout = new QHBoxLayout;
	minValueLayout->addWidget(m_valueMinSlider);
	minValueLayout->addWidget(m_valueMinSpinBox);
	processLayout->addLayout(minValueLayout);

	if (!m_rep->fixedRGBLayersFromDataset()->isMinimumValueActive()) {
		m_valueMinSlider->hide();
		m_valueMinSpinBox->hide();
	}

	//Colors
//	processLayout->addWidget(createImageChooserWidget());
	QWidget* holder = new QWidget;
	QVBoxLayout* formLayout = new QVBoxLayout(holder);

	QLabel* label = new QLabel("Layers");
	m_slider = new QSlider(Qt::Orientation::Horizontal);
	m_slider->setMinimum(0);
	m_slider->setMaximum(m_rep->fixedRGBLayersFromDataset()->numLayers()-1);
	m_slider->setValue(m_rep->fixedRGBLayersFromDataset()->currentImageIndex());
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

	connect(m_slider, &QSlider::valueChanged, this, &FixedRGBLayersFromDatasetAndCubePropPanel::changeDataKeyFromSlider);

	long val0 = m_rep->fixedRGBLayersFromDataset()->isoOrigin();
	long val1 = val0 + (m_rep->fixedRGBLayersFromDataset()->numLayers()-1) * m_rep->fixedRGBLayersFromDataset()->isoStep();

	m_layerNameSpinBox = new EditingSpinBox;
	m_layerNameSpinBox->setMinimum(std::min(val0, val1));
	m_layerNameSpinBox->setMaximum(std::max(val0, val1));
	m_layerNameSpinBox->setSingleStep(std::abs(m_rep->fixedRGBLayersFromDataset()->isoStep()));
	m_layerNameSpinBox->setValue(m_rep->fixedRGBLayersFromDataset()->isoOrigin() +
			m_rep->fixedRGBLayersFromDataset()->currentImageIndex()*m_rep->fixedRGBLayersFromDataset()->isoStep());

	connect(m_layerNameSpinBox, &EditingSpinBox::contentUpdated, this, &FixedRGBLayersFromDatasetAndCubePropPanel::changeDataKeyFromSpinBox,
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

	connect(m_multiplierComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &FixedRGBLayersFromDatasetAndCubePropPanel::multiplierChanged);


	QHBoxLayout* simplifyMeshLayout = new QHBoxLayout;
	QLabel* labelsimplify = new QLabel("Simplify mesh steps");

	m_simplifyStepsSpinBox = new QSpinBox;
	m_simplifyStepsSpinBox->setMinimum(1);
	m_simplifyStepsSpinBox->setMaximum(100);
	m_simplifyStepsSpinBox->setValue(m_rep->fixedRGBLayersFromDataset()->getSimplifyMeshSteps());

	connect(m_simplifyStepsSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), m_rep->fixedRGBLayersFromDataset(), &FixedRGBLayersFromDatasetAndCube::setSimplifyMeshSteps);

	connect(m_rep->fixedRGBLayersFromDataset(),  &FixedRGBLayersFromDatasetAndCube::simplifyMeshStepsChanged, m_simplifyStepsSpinBox, &QSpinBox::setValue);

	simplifyMeshLayout->addWidget(labelsimplify);
	simplifyMeshLayout->addWidget(m_simplifyStepsSpinBox);

	QHBoxLayout* compressionMeshLayout = new QHBoxLayout;
	QLabel* labelcompression = new QLabel("taux compression");

	m_compressionSpinBox = new QSpinBox;
	m_compressionSpinBox->setMinimum(0);
	m_compressionSpinBox->setMaximum(99);
	m_compressionSpinBox->setSuffix("%");
	m_compressionSpinBox->setValue(m_rep->fixedRGBLayersFromDataset()->getCompressionMesh());

	connect(m_compressionSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), m_rep->fixedRGBLayersFromDataset(), &FixedRGBLayersFromDatasetAndCube::setCompressionMesh);

	connect(m_rep->fixedRGBLayersFromDataset(),  &FixedRGBLayersFromDatasetAndCube::compressionMeshChanged, m_compressionSpinBox, &QSpinBox::setValue);

	compressionMeshLayout->addWidget(labelcompression);
	compressionMeshLayout->addWidget(m_compressionSpinBox);


	QWidget* holderLayer = new QWidget;
	QHBoxLayout* holderLayerLayout = new QHBoxLayout;
	//holderLayerLayout->setMargin(0);
	holderLayerLayout->setContentsMargins(0,0,0,0);
	holderLayer->setLayout(holderLayerLayout);

	QWidget* holderLayerLv2 = new QWidget;
	QHBoxLayout* holderLayerLayoutLv2 = new QHBoxLayout;
	holderLayerLv2->setLayout(holderLayerLayoutLv2);

	/*holderLayerLayout->addWidget(label);
	holderLayerLayout->addWidget(lessButton, 0);
	holderLayerLayout->addWidget(m_slider);
	holderLayerLayout->addWidget(moreButton, 0);
	holderLayerLayout->addWidget(m_playButton, 0);
	holderLayerLayout->addWidget(m_loopButton, 0);
	holderLayerLayoutLv2->addWidget(m_layerNameSpinBox, 0, Qt::AlignmentFlag::AlignLeft);
	holderLayerLayoutLv2->addWidget(m_multiplierComboBox, 0, Qt::AlignmentFlag::AlignLeft);*/
	holderLayerLayout->addWidget(label);
	holderLayerLayout->addWidget(m_layerNameSpinBox, 0, Qt::AlignmentFlag::AlignLeft);
	holderLayerLayout->addWidget(m_multiplierComboBox, 0, Qt::AlignmentFlag::AlignLeft);
	holderLayerLayoutLv2->addWidget(lessButton, 0);
	holderLayerLayoutLv2->addWidget(m_slider);
	holderLayerLayoutLv2->addWidget(moreButton, 0);
	holderLayerLayoutLv2->addWidget(m_playButton, 0);
	holderLayerLayoutLv2->addWidget(m_loopButton, 0);

	formLayout->addWidget(holderLayer);
	formLayout->addWidget(holderLayerLv2);

	processLayout->addWidget(holder);
	processLayout->addWidget(modeHolder);

	m_progressBar = new QProgressBar;
	processLayout->addWidget(m_progressBar);
	m_progressBar->hide();

	processLayout->addLayout(simplifyMeshLayout);
	processLayout->addLayout(compressionMeshLayout);

	//processLayout->addWidget(m_simplifyStepsSpinBox);
	connect(m_rep->fixedRGBLayersFromDataset()->image(), &CUDARGBInterleavedImage::dataChanged, this,
			[this]() {
		QSignalBlocker b(m_slider);
		QSignalBlocker b2(m_layerNameSpinBox);
		m_slider->setValue(m_rep->fixedRGBLayersFromDataset()->currentImageIndex());
		m_layerNameSpinBox->setValue(m_rep->fixedRGBLayersFromDataset()->isoOrigin() +
				m_rep->fixedRGBLayersFromDataset()->currentImageIndex()*m_rep->fixedRGBLayersFromDataset()->isoStep());
	});

	connect(m_rep->fixedRGBLayersFromDataset(), &FixedRGBLayersFromDatasetAndCube::modeChanged, this,
			&FixedRGBLayersFromDatasetAndCubePropPanel::modeChanged);

	connect(lessButton, &QToolButton::clicked, [this]() {
		if (m_rep->fixedRGBLayersFromDataset()->getIsoStep()>0) {
			m_layerNameSpinBox->stepDown();
		} else {
			m_layerNameSpinBox->stepUp();
		}
	});

	connect(moreButton, &QToolButton::clicked, [this]() {
		if (m_rep->fixedRGBLayersFromDataset()->getIsoStep()>0) {
			m_layerNameSpinBox->stepUp();
		} else {
			m_layerNameSpinBox->stepDown();
		}
	});

	connect(m_playButton, &QToolButton::clicked, [this]() {
		if(m_rep->fixedRGBLayersFromDataset()->modePlay())
			m_playButton->setIcon(style()->standardPixmap( QStyle::SP_MediaPlay));
		else
			m_playButton->setIcon(style()->standardPixmap( QStyle::SP_MediaPause));

		bool looping =m_loopButton->isChecked();
		int coef = m_multiplierComboBox->currentData().toInt();
		 m_rep->fixedRGBLayersFromDataset()->play(250, coef,looping);
		});

	connect(m_rep->fixedRGBLayersFromDataset(), &FixedRGBLayersFromDatasetAndCube::initProgressBar, this,
			&FixedRGBLayersFromDatasetAndCubePropPanel::initProgressBar);
	connect(m_rep->fixedRGBLayersFromDataset(), &FixedRGBLayersFromDatasetAndCube::valueProgressBarChanged, this,
			&FixedRGBLayersFromDatasetAndCubePropPanel::valueProgressBarChanged);
	connect(m_rep->fixedRGBLayersFromDataset(), &FixedRGBLayersFromDatasetAndCube::endProgressBar, this,
			&FixedRGBLayersFromDatasetAndCubePropPanel::endProgressBar);

	modeChanged(); // finish init

	connect(m_rep->fixedRGBLayersFromDataset(), &FixedRGBLayersFromDatasetAndCube::minimumValueActivated, this,
				&FixedRGBLayersFromDatasetAndCubePropPanel::minActivated);
	connect(m_rep->fixedRGBLayersFromDataset(), &FixedRGBLayersFromDatasetAndCube::minimumValueChanged, this,
				&FixedRGBLayersFromDatasetAndCubePropPanel::minValueChanged);

//	updateSpectrum();

	//Listen to data modification
//	connect(m_rep->rgbLayerSlice(), SIGNAL(frequencyChanged()), this,
//			SLOT(frequencyChanged()));
//	connect(m_rep->rgbLayerSlice()->layerSlice(), &LayerSlice::computationFinished, [this]() {
//		if (m_rep->rgbLayerSlice()->redIndex()!=m_redSpin->value()) {
//			QSignalBlocker b1(m_redSpin);
//			QSignalBlocker b2(m_redSlider);
//			m_redSlider->setValue(m_rep->rgbLayerSlice()->redIndex());
//
//			m_redSpin->setValue(m_rep->rgbLayerSlice()->redIndex());
//		}
//		if (m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()!=m_redSpin->maximum()+1) {
//			QSignalBlocker b1(m_redSpin);
//			QSignalBlocker b2(m_redSlider);
//			m_redSlider->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-1);
//
//			m_redSpin->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-1);
//		}
//		if (m_rep->rgbLayerSlice()->blueIndex()!=m_blueSpin->value()) {
//			QSignalBlocker b1(m_blueSpin);
//			QSignalBlocker b2(m_blueSlider);
//			m_blueSlider->setValue(m_rep->rgbLayerSlice()->blueIndex());
//
//			m_blueSpin->setValue(m_rep->rgbLayerSlice()->blueIndex());
//		}
//		if (m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()!=m_blueSpin->maximum()+1) {
//			QSignalBlocker b1(m_blueSpin);
//			QSignalBlocker b2(m_blueSlider);
//			m_blueSlider->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-1);
//
//			m_blueSpin->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-1);
//		}
//		if (m_rep->rgbLayerSlice()->redIndex()!=m_greenSpin->value()) {
//			QSignalBlocker b1(m_greenSpin);
//			QSignalBlocker b2(m_greenSlider);
//			m_greenSlider->setValue(m_rep->rgbLayerSlice()->greenIndex());
//
//			m_greenSpin->setValue(m_rep->rgbLayerSlice()->greenIndex());
//		}
//		if (m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()!=m_greenSpin->maximum()+1) {
//			QSignalBlocker b1(m_greenSpin);
//			QSignalBlocker b2(m_greenSlider);
//			m_greenSlider->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-1);
//
//			m_greenSpin->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-1);
//		}
//	});
}

FixedRGBLayersFromDatasetAndCubePropPanel::~FixedRGBLayersFromDatasetAndCubePropPanel() {
}

void FixedRGBLayersFromDatasetAndCubePropPanel::changeDataKeyFromSlider(long index) {
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

void FixedRGBLayersFromDatasetAndCubePropPanel::changeDataKeyFromSliderInternal(long index) {
	FixedRGBLayersFromDatasetAndCube* data = m_rep->fixedRGBLayersFromDataset();
	if (data->mode()==FixedRGBLayersFromDatasetAndCube::CACHE) {
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

void FixedRGBLayersFromDatasetAndCubePropPanel::changeDataKeyFromSpinBox() {
	long indexIso = m_layerNameSpinBox->value();
	long index = (indexIso - m_rep->fixedRGBLayersFromDataset()->isoOrigin()) / m_rep->fixedRGBLayersFromDataset()->isoStep();
	QSignalBlocker b(m_slider);
	m_rep->fixedRGBLayersFromDataset()->setCurrentImageIndex(index);
	m_slider->setValue(index);
}

void FixedRGBLayersFromDatasetAndCubePropPanel::multiplierChanged(int index) {
	bool ok;
	m_stepMultiplier = m_multiplierComboBox->itemData(index).toInt(&ok);
	if (!ok) {
		m_stepMultiplier = 1;
	}
	m_layerNameSpinBox->setSingleStep(m_stepMultiplier * std::abs(m_rep->fixedRGBLayersFromDataset()->isoStep()));
}

QWidget* FixedRGBLayersFromDatasetAndCubePropPanel::createImageChooserWidget() {
//	//Frequency chooser
//	QWidget *colorWidget = new QWidget(this);
//	QFormLayout *colorLayout = new QFormLayout(colorWidget);
//
//	QLabel *r = new QLabel("Red");
//	r->setPixmap(QIcon(":/palette/icons/red.png").pixmap(QSize(16, 16)));
//	QWidget *redWidget = new QWidget(this);
//
//
	return nullptr; //colorWidget;
}

void FixedRGBLayersFromDatasetAndCubePropPanel::updatePalette(int i) {
	if (m_palette) {
		m_palette->setPaletteHolder(i, m_rep->fixedRGBLayersFromDataset()->image()->holder(i));
	}
}

void FixedRGBLayersFromDatasetAndCubePropPanel::modeChangedInternal(int index) {
	FixedRGBLayersFromDatasetAndCube::Mode mode = FixedRGBLayersFromDatasetAndCube::READ;
	if (index==1) {
		mode = FixedRGBLayersFromDatasetAndCube::CACHE;
	}
	if (mode!=m_rep->fixedRGBLayersFromDataset()->mode()) {
		if (mode==FixedRGBLayersFromDatasetAndCube::READ) {
			m_rep->fixedRGBLayersFromDataset()->moveToReadMode();
		} else {
			// get begin, end, step
			FixedRGBLayersFromDatasetAndCube* data = m_rep->fixedRGBLayersFromDataset();
			SubGridGetterDialog dialog(data->isoOrigin(), data->isoOrigin()+std::max((std::size_t)0, data->numLayers()-1)*data->isoStep(), data->isoStep());
			dialog.activateMemoryCost(data->cacheLayerMemoryCost()); // RGB in uchar + iso as short
			bool result = dialog.exec()==QDialog::Accepted;

			if (result) {
				long begin = dialog.outBegin();
				long end = dialog.outEnd();
				long step = dialog.outStep();
				result = m_rep->fixedRGBLayersFromDataset()->moveToCacheMode(begin, end, step);
				if (!result) {
					m_modeComboBox->setCurrentIndex(0);// return to read mode
				}
			}
		}
	}
}

void FixedRGBLayersFromDatasetAndCubePropPanel::modeChanged() {
	FixedRGBLayersFromDatasetAndCube* data = m_rep->fixedRGBLayersFromDataset();
	if (data->mode()==FixedRGBLayersFromDatasetAndCube::READ) {
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

void FixedRGBLayersFromDatasetAndCubePropPanel::initProgressBar(int min, int max, int val) {
	m_progressBar->setRange(min, max);
	m_progressBar->setValue(val);
	m_progressBar->show();
}

void FixedRGBLayersFromDatasetAndCubePropPanel::valueProgressBarChanged(int val) {
	m_progressBar->setValue(val);
}

void FixedRGBLayersFromDatasetAndCubePropPanel::endProgressBar() {
	m_progressBar->hide();
}

void FixedRGBLayersFromDatasetAndCubePropPanel::changeMinActivated(int state) {
	m_rep->fixedRGBLayersFromDataset()->setMinimumValueActive(state==Qt::Checked);
}

void FixedRGBLayersFromDatasetAndCubePropPanel::changeMinSlider(int value) {
	m_rep->fixedRGBLayersFromDataset()->setMinimumValue(value/100.0f);
}

void FixedRGBLayersFromDatasetAndCubePropPanel::changeMinSpinBox(double value) {
	m_rep->fixedRGBLayersFromDataset()->setMinimumValue(value);
}

void FixedRGBLayersFromDatasetAndCubePropPanel::minActivated(bool activated) {
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

void FixedRGBLayersFromDatasetAndCubePropPanel::minValueChanged(float minValue) {
	QSignalBlocker bSlider(m_valueMinSlider);
	QSignalBlocker bSpinBox(m_valueMinSpinBox);

	m_valueMinSlider->setValue(std::floor(minValue*100));
	m_valueMinSpinBox->setValue(minValue);
}

