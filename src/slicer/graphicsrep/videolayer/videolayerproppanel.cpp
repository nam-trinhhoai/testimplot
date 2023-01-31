#include "videolayerproppanel.h"
#include "videolayerrep.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QComboBox>
#include <QLineEdit>
#include <QSlider>
#include <QMediaPlayer>

VideoLayerPropPanel::VideoLayerPropPanel(VideoLayerRep* rep, QWidget* parent) : QWidget(parent) {
	m_rep = rep;

	buildWidget();
}

VideoLayerPropPanel::~VideoLayerPropPanel() {}

void VideoLayerPropPanel::buildWidget() {
	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	QHBoxLayout* layout = new QHBoxLayout;
	mainLayout->addLayout(layout);
	QPushButton* startButton = new QPushButton("Play");
	layout->addWidget(startButton);

	m_speedComboBox = new QComboBox;
	m_speedComboBox->addItem("x1", QVariant((int) 1));
	m_speedComboBox->addItem("x10", QVariant((int) 10));
	m_speedComboBox->addItem("x-1", QVariant((int) -1));
	m_speedComboBox->addItem("x-10", QVariant((int) -10));
	layout->addWidget(m_speedComboBox);


	QPushButton* stopButton = new QPushButton("Stop");
	mainLayout->addWidget(stopButton);

	QHBoxLayout* valueLayout = new QHBoxLayout;
	m_valueSlider = new QSlider;
	m_valueSlider->setMaximum(m_rep->mediaPlayer()->duration());
	m_valueSlider->setValue(m_rep->mediaPlayer()->position());
	valueLayout->addWidget(m_valueSlider);

	m_valueLineEdit = new QLineEdit(QString::number(m_rep->mediaPlayer()->position()));
	valueLayout->addWidget(m_valueLineEdit);

	connect(startButton, &QPushButton::clicked, m_rep->mediaPlayer(), &QMediaPlayer::play);
	connect(stopButton, &QPushButton::clicked, m_rep->mediaPlayer(), &QMediaPlayer::pause);

	connect(m_speedComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &VideoLayerPropPanel::changeSpeed);

	connect(m_rep->mediaPlayer(), &QMediaPlayer::positionChanged, this, &VideoLayerPropPanel::positionChanged);
	connect(m_rep->mediaPlayer(), &QMediaPlayer::durationChanged, this, &VideoLayerPropPanel::durationChanged);
}

void VideoLayerPropPanel::changeSpeed(int index) {
	QVariant variant = m_speedComboBox->itemData(index, Qt::UserRole);
	m_rep->mediaPlayer()->setPlaybackRate(variant.toInt());
}

void VideoLayerPropPanel::positionChanged(qint64 position) {
	m_valueSlider->setValue(position);
	m_valueLineEdit->setText(QString::number(position));
}

void VideoLayerPropPanel::durationChanged(qint64 duration) {
	m_valueSlider->setMaximum(duration);
}
