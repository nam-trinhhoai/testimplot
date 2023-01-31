

#include <FixedRGBLayersFromDatasetAndCube3FreqPropPanel.h>

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
#include <QSignalBlocker>

#include <iostream>
#include <sstream>

#include "rgbpalettewidget.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "fixedrgblayersfromdatasetandcuberep.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "editingspinbox.h"
#include "subgridgetterdialog.h"

#include <FixedRGBLayersFromDatasetAndCube3FreqPropPanel.h>
#include <fixedrgblayersfromdatasetandcubeproppanel.h>

FixedRGBLayersFromDatasetAndCube3FreqPropPanel::FixedRGBLayersFromDatasetAndCube3FreqPropPanel(FixedRGBLayersFromDatasetAndCubeRep *rep, QWidget *parent)
// : QWidget(parent)
: FixedRGBLayersFromDatasetAndCubePropPanel(rep, parent, false)
{
	m_rep = rep;
	QVBoxLayout *processLayout = new QVBoxLayout(this);
	processLayout->setContentsMargins(0,0,0,0);

	/*
	QWidget* modeHolder = new QWidget;
	QHBoxLayout* modeLayout = new QHBoxLayout;
	modeHolder->setLayout(modeLayout);
	// processLayout->addWidget(modeHolder);
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

	connect(m_modeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &FixedRGBLayersFromDatasetAndCube3FreqPropPanel::modeChangedInternal);
	*/

	QHBoxLayout *displayModeLayout = new QHBoxLayout;
	displayModeLayout->addWidget(new QLabel("Mode:"));
	m_displayModeComboBox = new QComboBox;
	m_displayModeComboBox->addItem("RGB");
	m_displayModeComboBox->addItem("Frequency");
	m_displayModeComboBox->setCurrentIndex(0);
	displayModeLayout->addWidget(m_displayModeComboBox);
	processLayout->addLayout(displayModeLayout);
	connect(m_displayModeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &FixedRGBLayersFromDatasetAndCube3FreqPropPanel::displayModeChanged);

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
			hBox->setContentsMargins(0,0,0,0);
			hBox->addWidget(m_opacity);
			hBox->addWidget(opacitySpin);
			processLayout->addWidget(opacityWidget,0,Qt::AlignmentFlag::AlignTop);

	}

	QLabel* useMinValueLabel = new QLabel("Value min");
	m_valueMinCheckBox = new QCheckBox();
	m_valueMinCheckBox->setCheckState((m_rep->fixedRGBLayersFromDataset()->isMinimumValueActive())? Qt::Checked : Qt::Unchecked);

	connect(m_valueMinCheckBox, &QCheckBox::stateChanged, this, &FixedRGBLayersFromDatasetAndCube3FreqPropPanel::changeMinActivated);

	QHBoxLayout* useMinLayout = new QHBoxLayout;
	useMinLayout->addWidget(useMinValueLabel);
	useMinLayout->addWidget(m_valueMinCheckBox);
	// processLayout->addLayout(useMinLayout);

	m_valueMinSlider = new QSlider(Qt::Horizontal);
	m_valueMinSlider->setMinimum(0);
	m_valueMinSlider->setMaximum(100);
	m_valueMinSlider->setValue(std::floor(m_rep->fixedRGBLayersFromDataset()->minimumValue()*100));

	connect(m_valueMinSlider, &QSlider::valueChanged, this, &FixedRGBLayersFromDatasetAndCube3FreqPropPanel::changeMinSlider);

	m_valueMinSpinBox = new QDoubleSpinBox();
	m_valueMinSpinBox->setMinimum(0);
	m_valueMinSpinBox->setMaximum(1);
	m_valueMinSpinBox->setSingleStep(0.01);
	m_valueMinSpinBox->setValue(m_rep->fixedRGBLayersFromDataset()->minimumValue());

	connect(m_valueMinSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &FixedRGBLayersFromDatasetAndCube3FreqPropPanel::changeMinSpinBox);

	QHBoxLayout* minValueLayout = new QHBoxLayout;
	minValueLayout->addWidget(m_valueMinSlider);
	minValueLayout->addWidget(m_valueMinSpinBox);
	// processLayout->addLayout(minValueLayout);

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

	connect(m_slider, &QSlider::valueChanged, this, &FixedRGBLayersFromDatasetAndCube3FreqPropPanel::changeDataKeyFromSlider);

	long val0 = m_rep->fixedRGBLayersFromDataset()->isoOrigin();
	long val1 = val0 + (m_rep->fixedRGBLayersFromDataset()->numLayers()-1) * m_rep->fixedRGBLayersFromDataset()->isoStep();

	m_layerNameSpinBox = new EditingSpinBox;
	m_layerNameSpinBox->setMinimum(std::min(val0, val1));
	m_layerNameSpinBox->setMaximum(std::max(val0, val1));
	m_layerNameSpinBox->setSingleStep(std::abs(m_rep->fixedRGBLayersFromDataset()->isoStep()));
	m_layerNameSpinBox->setValue(m_rep->fixedRGBLayersFromDataset()->isoOrigin() +
			m_rep->fixedRGBLayersFromDataset()->currentImageIndex()*m_rep->fixedRGBLayersFromDataset()->isoStep());

	connect(m_layerNameSpinBox, &EditingSpinBox::contentUpdated, this, &FixedRGBLayersFromDatasetAndCube3FreqPropPanel::changeDataKeyFromSpinBox,
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

	connect(m_multiplierComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &FixedRGBLayersFromDatasetAndCube3FreqPropPanel::multiplierChanged);


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

	// processLayout->addWidget(holder);
	// processLayout->addWidget(modeHolder);

	m_progressBar = new QProgressBar;
	processLayout->addWidget(m_progressBar);
	m_progressBar->hide();

	// processLayout->addLayout(simplifyMeshLayout);
	// processLayout->addLayout(compressionMeshLayout);

	//processLayout->addWidget(m_simplifyStepsSpinBox);


	m_frequencyColorWidget = createFreqChooserWidget();
	processLayout->addWidget(m_frequencyColorWidget);
	m_frequencyGrayWidget = createGrayFreqChooserWidget();
	processLayout->addWidget(m_frequencyGrayWidget);
	displayModeChanged(0);

	connect(m_rep->fixedRGBLayersFromDataset()->image(), &CUDARGBInterleavedImage::dataChanged, this,
			[this]() {
		QSignalBlocker b(m_slider);
		QSignalBlocker b2(m_layerNameSpinBox);
		m_slider->setValue(m_rep->fixedRGBLayersFromDataset()->currentImageIndex());
		m_layerNameSpinBox->setValue(m_rep->fixedRGBLayersFromDataset()->isoOrigin() +
				m_rep->fixedRGBLayersFromDataset()->currentImageIndex()*m_rep->fixedRGBLayersFromDataset()->isoStep());
	});

	connect(m_rep->fixedRGBLayersFromDataset(), &FixedRGBLayersFromDatasetAndCube::modeChanged, this,
			&FixedRGBLayersFromDatasetAndCube3FreqPropPanel::modeChanged);

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
			&FixedRGBLayersFromDatasetAndCube3FreqPropPanel::initProgressBar);
	connect(m_rep->fixedRGBLayersFromDataset(), &FixedRGBLayersFromDatasetAndCube::valueProgressBarChanged, this,
			&FixedRGBLayersFromDatasetAndCube3FreqPropPanel::valueProgressBarChanged);
	connect(m_rep->fixedRGBLayersFromDataset(), &FixedRGBLayersFromDatasetAndCube::endProgressBar, this,
			&FixedRGBLayersFromDatasetAndCube3FreqPropPanel::endProgressBar);

	// jd
	// modeChanged(); // finish init

	connect(m_rep->fixedRGBLayersFromDataset(), &FixedRGBLayersFromDatasetAndCube::minimumValueActivated, this,
				&FixedRGBLayersFromDatasetAndCube3FreqPropPanel::minActivated);
	connect(m_rep->fixedRGBLayersFromDataset(), &FixedRGBLayersFromDatasetAndCube::minimumValueChanged, this,
				&FixedRGBLayersFromDatasetAndCube3FreqPropPanel::minValueChanged);

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

	connect(m_rep->fixedRGBLayersFromDataset(), SIGNAL(frequencyChanged()), this, SLOT(frequencyChanged()));
	frequencyChanged();
	m_grayLineEdit->setText(m_rep->fixedRGBLayersFromDataset()->getLabelFromPosition(m_graySlider->value()));
}

FixedRGBLayersFromDatasetAndCube3FreqPropPanel::~FixedRGBLayersFromDatasetAndCube3FreqPropPanel() {
}

QWidget* FixedRGBLayersFromDatasetAndCube3FreqPropPanel::createFreqChooserWidget() {
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

	QLabel *b = new QLabel("Blue");
	b->setPixmap(QIcon(":/palette/icons/blue.png").pixmap(QSize(16, 16)));
	QWidget *blueWidget = new QWidget(this);
	m_blueSlider = new QSlider(Qt::Orientation::Horizontal, this);
	m_blueSpin = new QSpinBox();
	m_blueLineEdit = new QLineEdit();
	createlinkedSliderSpin(blueWidget, m_blueSlider, m_blueSpin, m_blueLineEdit);
	colorLayout->addRow(b, blueWidget);

	m_nbreSpectrumFreq = m_rep->fixedRGBLayersFromDataset()->getNbreSpectrumFreq();

	m_greenSlider->setMaximum(m_nbreSpectrumFreq-1);
	m_greenSpin->setMaximum(m_nbreSpectrumFreq-1);
	m_redSlider->setMaximum(m_nbreSpectrumFreq-1);
	m_redSpin->setMaximum(m_nbreSpectrumFreq-1);
	m_blueSlider->setMaximum(m_nbreSpectrumFreq-1);
	m_blueSpin->setMaximum(m_nbreSpectrumFreq-1);

	int f1 = m_rep->fixedRGBLayersFromDataset()->getRedIndex();
	int f2 = m_rep->fixedRGBLayersFromDataset()->getGreenIndex();
	int f3 = m_rep->fixedRGBLayersFromDataset()->getBlueIndex();
	m_redSlider->setValue(f1);
	m_greenSlider->setValue(f2);
	m_blueSlider->setValue(f3);
	m_redSpin->setValue(f1);
	m_greenSpin->setValue(f2);
	m_blueSpin->setValue(f3);
	m_rep->fixedRGBLayersFromDataset()->setRGBIndexes(f1, f2, f3);
	m_freqRedSave = f1;
	m_freqGreenSave = f2;
	m_freqBlueSave = f3;
	m_freqGraySave = f2;


	setRedIndex(m_redSlider->value());
	setGreenIndex(m_greenSlider->value());
	setBlueIndex(m_blueSlider->value());

	connect(m_redSlider, SIGNAL(valueChanged(int)), this, SLOT(setRedIndex(int)));
	connect(m_redSpin, SIGNAL(valueChanged(int)), this, SLOT(setRedIndex(int)));

	connect(m_greenSlider, SIGNAL(valueChanged(int)), this, SLOT(setGreenIndex(int)));
	connect(m_greenSpin, SIGNAL(valueChanged(int)), this, SLOT(setGreenIndex(int)));

	connect(m_blueSlider, SIGNAL(valueChanged(int)), this, SLOT(setBlueIndex(int)));
	connect(m_blueSpin, SIGNAL(valueChanged(int)), this, SLOT(setBlueIndex(int)));

	return colorWidget;
}

QWidget* FixedRGBLayersFromDatasetAndCube3FreqPropPanel::createGrayFreqChooserWidget()
{
	QWidget* widget = new QWidget();
	QHBoxLayout *layout = new QHBoxLayout(widget);

	QLabel *label = new QLabel("frequency");
	m_graySlider = new QSlider(Qt::Orientation::Horizontal, this);
	m_graySpin = new QSpinBox();
	m_grayLineEdit = new QLineEdit;
	m_grayLineEdit->setEnabled(false);
	m_grayLineEdit->setMaxLength(11);
	QMargins margins = m_grayLineEdit->contentsMargins();
	m_grayLineEdit->setMaximumWidth(QFontMetrics(m_grayLineEdit->font()).maxWidth() * m_grayLineEdit->maxLength() + margins.left() + margins.right());

	// createlinkedSliderSpin(redWidget, m_redSlider, m_redSpin, m_redLineEdit);
	layout->addWidget(label);
	layout->addWidget(m_graySlider);
	layout->addWidget(m_graySpin);
	layout->addWidget(m_grayLineEdit);
	m_graySlider->setMaximum(m_nbreSpectrumFreq-1);
	m_graySpin->setMaximum(m_nbreSpectrumFreq-1);
	connect(m_graySlider, SIGNAL(valueChanged(int)), this, SLOT(setGrayFreqIndex(int)));
	connect(m_graySpin, SIGNAL(valueChanged(int)), this, SLOT(setGrayFreqSpinIndex(int)));
	return widget;
}

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::createlinkedSliderSpin(QWidget *parent, QSlider *slider,
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
	hBox->setContentsMargins(0,0,0,0);
	hBox->addWidget(slider);
	hBox->addWidget(spin);
	hBox->addWidget(lineEdit);
}




void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::changeDataKeyFromSlider(long index) {
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

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::changeDataKeyFromSliderInternal(long index) {
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

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::changeDataKeyFromSpinBox() {
	long indexIso = m_layerNameSpinBox->value();
	long index = (indexIso - m_rep->fixedRGBLayersFromDataset()->isoOrigin()) / m_rep->fixedRGBLayersFromDataset()->isoStep();
	QSignalBlocker b(m_slider);
	m_rep->fixedRGBLayersFromDataset()->setCurrentImageIndex(index);
	m_slider->setValue(index);
}

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::multiplierChanged(int index) {
	bool ok;
	m_stepMultiplier = m_multiplierComboBox->itemData(index).toInt(&ok);
	if (!ok) {
		m_stepMultiplier = 1;
	}
	m_layerNameSpinBox->setSingleStep(m_stepMultiplier * std::abs(m_rep->fixedRGBLayersFromDataset()->isoStep()));
}

QWidget* FixedRGBLayersFromDatasetAndCube3FreqPropPanel::createImageChooserWidget() {
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

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::updatePalette(int i) {
	if (m_palette) {
		m_palette->setPaletteHolder(i, m_rep->fixedRGBLayersFromDataset()->image()->holder(i));
	}
}

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::modeChangedInternal(int index) {
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

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::modeChanged() {
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

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::initProgressBar(int min, int max, int val) {
	m_progressBar->setRange(min, max);
	m_progressBar->setValue(val);
	m_progressBar->show();
}

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::valueProgressBarChanged(int val) {
	m_progressBar->setValue(val);
}

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::endProgressBar() {
	m_progressBar->hide();
}

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::changeMinActivated(int state) {
	m_rep->fixedRGBLayersFromDataset()->setMinimumValueActive(state==Qt::Checked);
}

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::changeMinSlider(int value) {
	m_rep->fixedRGBLayersFromDataset()->setMinimumValue(value/100.0f);
}

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::changeMinSpinBox(double value) {
	m_rep->fixedRGBLayersFromDataset()->setMinimumValue(value);
}

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::minActivated(bool activated) {
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

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::minValueChanged(float minValue) {
	QSignalBlocker bSlider(m_valueMinSlider);
	QSignalBlocker bSpinBox(m_valueMinSpinBox);

	m_valueMinSlider->setValue(std::floor(minValue*100));
	m_valueMinSpinBox->setValue(minValue);
}

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::frequencyChanged() {
	m_oldDeltaRed = m_rep->fixedRGBLayersFromDataset()->getGreenIndex() - m_rep->fixedRGBLayersFromDataset()->getRedIndex();
	m_oldDeltaBlue = m_rep->fixedRGBLayersFromDataset()->getBlueIndex() - m_rep->fixedRGBLayersFromDataset()->getGreenIndex();
	updateSpinValue(m_rep->fixedRGBLayersFromDataset()->getRedIndex(), m_redSlider, m_redSpin, m_redLineEdit);
	updateSpinValue(m_rep->fixedRGBLayersFromDataset()->getGreenIndex(), m_greenSlider, m_greenSpin, m_greenLineEdit);
	updateSpinValue(m_rep->fixedRGBLayersFromDataset()->getBlueIndex(), m_blueSlider, m_blueSpin, m_blueLineEdit);
}

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::updateSpinValue(int value, QSlider *slider, QSpinBox *spin, QLineEdit* lineEdit) {
	QSignalBlocker b1(slider);
	QSignalBlocker b2(spin);
	slider->setValue(value);
	spin->setValue(value);
	lineEdit->setText(m_rep->fixedRGBLayersFromDataset()->getLabelFromPosition(value));
	//lineEdit->setText(QString::number(m_rep->rgbLayerSlice()->layerSlice()->getFrequency(value), 'f', 0));
}

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::displayModeChanged(int index)
{
	if ( index == 0 )
	{
		m_freqGraySave = m_rep->fixedRGBLayersFromDataset()->getRedIndex();
		m_frequencyGrayWidget->setVisible(false);
		m_frequencyColorWidget->setVisible(true);
		int r = m_freqRedSave;
		int g = m_freqGreenSave;
		int b = m_freqBlueSave;
		m_rep->fixedRGBLayersFromDataset()->setRGBIndexes(r, g, b);
	}
	else
	{
		m_freqRedSave = m_rep->fixedRGBLayersFromDataset()->getRedIndex();
		m_freqGreenSave = m_rep->fixedRGBLayersFromDataset()->getGreenIndex();
		m_freqBlueSave = m_rep->fixedRGBLayersFromDataset()->getBlueIndex();

		m_frequencyColorWidget->setVisible(false);
		m_frequencyGrayWidget->setVisible(true);
		int val = m_freqGraySave;
		m_rep->fixedRGBLayersFromDataset()->setRGBIndexes(val, val, val);
	}
}


void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::setRedSpinIndex(int value) {

}
void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::setGreenSpinIndex(int value) {

}

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::setBlueSpinIndex(int value) {

}

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::setRedIndex(int value) {
	m_oldDeltaRed = m_rep->fixedRGBLayersFromDataset()->getGreenIndex() - value;
	m_rep->fixedRGBLayersFromDataset()->setRedIndex(value);
}
void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::setGreenIndex(int value) {
	// red
		int deltaRed = m_oldDeltaRed;
		int valRed = value - deltaRed;
		if (value < 0) {
			valRed = value;
		}
		else if (value - deltaRed < 0) {
			valRed = 0;
		}
		else if (value - deltaRed > m_nbreSpectrumFreq) {
			valRed = m_nbreSpectrumFreq - 1;
		}
		m_oldDeltaRed = value - valRed;

		// blue
		int deltaBlue = m_oldDeltaBlue;
		int valBlue = value + deltaBlue;
		if (value < 0) {
			valBlue = value;
		}
		else if (value + deltaBlue > m_nbreSpectrumFreq - 1) {
			valBlue = m_nbreSpectrumFreq - 1;
		}
		else if (valBlue < 0) {
			valBlue = 0;
		}
		m_oldDeltaBlue = valBlue - value;

		// only do that at the end to avoid m_oldDelta... modifications by called slot frequencyUpdated

		m_rep->fixedRGBLayersFromDataset()->setRGBIndexes(valRed, value, valBlue);
}

void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::setBlueIndex(int value) {
	m_oldDeltaBlue = value - (m_rep->fixedRGBLayersFromDataset()->getGreenIndex());
	m_rep->fixedRGBLayersFromDataset()->setBlueIndex(value);
}


void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::setGrayFreqIndex(int value)
{
	QSignalBlocker b1(m_graySlider);
	QSignalBlocker b2(m_graySpin);
	m_rep->fixedRGBLayersFromDataset()->setRGBIndexes(value, value, value);
	m_graySpin->setValue(value);
	m_grayLineEdit->setText(m_rep->fixedRGBLayersFromDataset()->getLabelFromPosition(value));
}


void FixedRGBLayersFromDatasetAndCube3FreqPropPanel::setGrayFreqSpinIndex(int value)
{
	QSignalBlocker b1(m_graySlider);
	QSignalBlocker b2(m_graySpin);
	m_rep->fixedRGBLayersFromDataset()->setRGBIndexes(value, value, value);
	m_graySlider->setValue(value);
	m_grayLineEdit->setText(m_rep->fixedRGBLayersFromDataset()->getLabelFromPosition(value));
}
