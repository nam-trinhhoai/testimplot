/*
 * palettewidget.cpp
 *
 *  Created on: 4 mai 2018
 *      Author: j0334308
 */

#include "rgbpalettewidget.h"

#include <QSlider>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QSpacerItem>
#include <QTabWidget>

#include "rangehistogramwidget.h"
template<typename ... Args> struct SELECT {
	template<typename C, typename R>
	static constexpr auto OVERLOAD_OF(R (C::*pmf)(Args...)) -> decltype(pmf) {
		return pmf;
	}
};
RGBPaletteWidget::RGBPaletteWidget(QWidget *parent, Qt::WindowFlags f) :
		QWidget(parent, f) {
	QVBoxLayout *layout = new QVBoxLayout(this);
	//layout->setMargin(0);
	layout->setContentsMargins(0,0,0,0);

	QTabWidget *tab = new QTabWidget(this);
	layout->addWidget(tab);
	{
		RangeAndHistogramWidget *redWidget = new RangeAndHistogramWidget(tab);
		m_histoWidget.push_back(redWidget);
		connect(redWidget, SIGNAL(rangeChanged(const QVector2D &)), this,
				SLOT(redRangeChanged(const QVector2D &)));

		redWidget->setUseLookupTable(false);
		redWidget->setDefaultColor(QColorConstants::Red);
		tab->addTab(redWidget, QIcon(":/palette/icons/red.png"), "Red");
	}

	{
		RangeAndHistogramWidget *greenWidget = new RangeAndHistogramWidget(tab);
		m_histoWidget.push_back(greenWidget);
		connect(greenWidget, SIGNAL(rangeChanged(const QVector2D &)), this,
				SLOT(greenRangeChanged(const QVector2D &)));
		greenWidget->setUseLookupTable(false);
		greenWidget->setDefaultColor(QColorConstants::Green);
		tab->addTab(greenWidget, QIcon(":/palette/icons/green.png"), "Green");
	}
	{
		RangeAndHistogramWidget *blueWidget = new RangeAndHistogramWidget(tab);
		m_histoWidget.push_back(blueWidget);
		connect(blueWidget, SIGNAL(rangeChanged(const QVector2D &)), this,
				SLOT(blueRangeChanged(const QVector2D &)));
		blueWidget->setUseLookupTable(false);
		blueWidget->setDefaultColor(QColorConstants::Blue);
		tab->addTab(blueWidget, QIcon(":/palette/icons/blue.png"), "Blue");
	}

	m_opacity = new QSlider(Qt::Orientation::Horizontal, this);
	m_opacity->setSingleStep(1);
	m_opacity->setTickInterval(10);
	m_opacity->setMinimum(0);
	m_opacity->setMaximum(100);
	m_opacity->setValue(100);

	QDoubleSpinBox *opacitySpin = new QDoubleSpinBox();
	opacitySpin->setMinimum(0);
	opacitySpin->setMaximum(1);
	opacitySpin->setSingleStep(0.01);
	opacitySpin->setDecimals(2);
	opacitySpin->setValue(1);
	connect(m_opacity, &QSlider::valueChanged, [=](int value) {
		opacitySpin->setValue(0.01 * value);
	});
	connect(opacitySpin,
			SELECT<double>::OVERLOAD_OF(&QDoubleSpinBox::valueChanged),
			[=](double value) {
				m_opacity->setValue(value * 100);
			});

	connect(m_opacity, SIGNAL(valueChanged(int)), this,
			SLOT(opacityChanged(int)));
	{
		QGroupBox *opacityWidget = new QGroupBox("Opacity", this);

		QHBoxLayout *hBox = new QHBoxLayout(opacityWidget);
		//hBox->setMargin(0);
		hBox->setContentsMargins(0,0,0,0);
		hBox->addWidget(m_opacity);
		hBox->addWidget(opacitySpin);
		layout->addWidget(opacityWidget, 0, Qt::AlignmentFlag::AlignTop);
	}

	layout->addSpacerItem(
			new QSpacerItem(0, 0, QSizePolicy::Minimum,
					QSizePolicy::Expanding));
}

void RGBPaletteWidget::setPaletteHolders(QVector<IPaletteHolder*> image) {
	for (int i = 0; i < 3; i++) {
		m_histoWidget[i]->setPaletteHolder(image[i]);
	}

}

void RGBPaletteWidget::setRange(unsigned int i, const QVector2D &range) {
	m_histoWidget[i]->setRange(range);
}

QVector2D RGBPaletteWidget::getRange(int i)
{
	return m_histoWidget[i]->getRange();
}

void RGBPaletteWidget::setPaletteHolder(int i, IPaletteHolder *image) {
	m_histoWidget[i]->setPaletteHolder(image);
}

float RGBPaletteWidget::getOpacity() const {
	return m_opacity->value() * 0.01f;
}

void RGBPaletteWidget::setOpacity(float val) {
	QSignalBlocker b(m_opacity);
	m_opacity->setValue((int) (val * 100));
}

void RGBPaletteWidget::opacityChanged(int value) {
	emit opacityChanged(value * 0.01f);
}

void RGBPaletteWidget::redRangeChanged(const QVector2D &range) {
	emit rangeChanged(0, range);
}
void RGBPaletteWidget::greenRangeChanged(const QVector2D &range) {
	emit rangeChanged(1, range);
}
void RGBPaletteWidget::blueRangeChanged(const QVector2D &range) {
	emit rangeChanged(2, range);
}

RGBPaletteWidget::~RGBPaletteWidget() {
}

