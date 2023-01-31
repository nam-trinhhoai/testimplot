
#include "rangehistogramwidget.h"

#include "histogramwidget.h"
#include "colortableregistry.h"
#include "ipaletteholder.h"

#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QHBoxLayout>
#include <QVBoxLayout>

#include <QLocale>
#include <QGroupBox>
#include <iostream>

RangeAndHistogramWidget::RangeAndHistogramWidget(QWidget* parent , Qt::WindowFlags f) :QWidget(parent,f) {
	QVBoxLayout *layout=new QVBoxLayout(this);
	//layout->setMargin(0);
	layout->setContentsMargins(0,0,0,0);
	m_histoWidget=new HistogramWidget(this);
	connect(m_histoWidget, SIGNAL(rangeChanged(const QVector2D &)), this, SLOT(histogramRangeChanged(const QVector2D &)));
	{
		QWidget * hWidget=new QWidget(this);
		QHBoxLayout *hBox=new QHBoxLayout(hWidget);
		//hBox->setMargin(0);
		hBox->setContentsMargins(0,0,0,0);
		hBox->addWidget(m_histoWidget);

		QWidget * buttonWidget=new QWidget(this);
		QVBoxLayout *vBox=new QVBoxLayout(buttonWidget);
		//vBox->setMargin(0);
		vBox->setContentsMargins(0,0,0,0);
		m_reset=new QPushButton();
		m_reset->setFixedSize(24,24);
		m_reset->setStyleSheet("min-width: 24px;");
		m_reset->setDefault(false);
		m_reset->setAutoDefault(false);
		m_reset->setIcon(QIcon(":/palette/icons/undo.png"));
		vBox->addWidget(m_reset);
		connect(m_reset, SIGNAL(clicked()), this, SLOT(resetRange()));

		m_recompute=new QPushButton();
		m_recompute->setFixedSize(24,24);
		m_recompute->setStyleSheet("min-width: 24px;");
		m_recompute->setDefault(false);
		m_recompute->setAutoDefault(false);
		m_recompute->setIcon(QIcon(":/palette/icons/shrink.png"));
		vBox->addWidget(m_recompute);
		connect(m_recompute, SIGNAL(clicked()), this, SLOT(recompute()));

		m_wand=new QPushButton();
		m_wand->setFixedSize(24,24);
		m_wand->setStyleSheet("min-width: 24px;");
		m_wand->setDefault(false);
		m_wand->setAutoDefault(false);
		m_wand->setIcon(QIcon(":/palette/icons/wand.png"));
		vBox->addWidget(m_wand);
		connect(m_wand, SIGNAL(clicked()), this, SLOT(smartAdjust()));

		hBox->addWidget(buttonWidget,0,Qt::AlignmentFlag::AlignTop);

		layout->addWidget(hWidget);
	}
	{
		QWidget * rangeWidget=new QWidget(this);
		QHBoxLayout *hBox=new QHBoxLayout(rangeWidget);
		//hBox->setMargin(0);
		hBox->setContentsMargins(0,0,0,0);
		m_min=new QLineEdit();
		m_min->setLocale(QLocale::C);
		connect(m_min, SIGNAL(editingFinished()), this, SLOT(valueChanged()));
		hBox->addWidget(new QLabel("Min:"));
		hBox->addWidget(m_min);

		m_max=new QLineEdit();
		m_max->setLocale(QLocale::C);
		connect(m_max, SIGNAL(editingFinished()), this, SLOT(valueChanged()));

		hBox->addSpacerItem(new QSpacerItem(0,0,QSizePolicy::Expanding, QSizePolicy::Minimum));
		hBox->addWidget(new QLabel("Max:"));
		hBox->addWidget(m_max);

		layout->addWidget(rangeWidget,0,Qt::AlignmentFlag::AlignTop);
	}
}
void RangeAndHistogramWidget::setUseLookupTable(bool useColorTable)
{
	m_histoWidget->setUseLookupTable(useColorTable);
}
void RangeAndHistogramWidget::setDefaultColor(QColor color)
{
	m_histoWidget->setDefaultColor(color);
}
void RangeAndHistogramWidget::setPaletteHolder(IPaletteHolder *image)
{
	m_image=image;
	m_originalRange=m_image->dataRange();
	m_histo=m_image->computeHistogram(m_originalRange,QHistogram::HISTOGRAM_SIZE);
	updateHistogram();
	setRangeField(m_image->range());
}

void RangeAndHistogramWidget::smartAdjust()
{
	QVector2D adpatedRange=IPaletteHolder::smartAdjust(m_originalRange,m_histo);
	setRangeField(adpatedRange);

	emit rangeChanged(adpatedRange);
}

void RangeAndHistogramWidget::setRangeField(const QVector2D &r)
{
	m_min->setText(locale().toString(r.x(), 'f', 2));
	m_max->setText(locale().toString(r.y(), 'f', 2));
	m_histoWidget->setRange(r);
}


void RangeAndHistogramWidget::resetRange()
{
	if(!m_image)return;

	setRangeField(m_originalRange);
	m_histoWidget->setRange(m_originalRange);

	m_histo = m_image->computeHistogram(m_originalRange,QHistogram::HISTOGRAM_SIZE);
	updateHistogram();
	emit rangeChanged(m_originalRange);
}


QVector2D RangeAndHistogramWidget::getRange() const
{
	QVector2D range;
	range.setX(locale().toDouble(m_min->text()));
	range.setY(locale().toDouble(m_max->text()));

	return range;
}

void RangeAndHistogramWidget::recompute()
{
	if(!m_image)return;

	m_histo = m_image->computeHistogram(getRange(),QHistogram::HISTOGRAM_SIZE);
	updateHistogram();
}

void RangeAndHistogramWidget::valueChanged()
{
	if(!m_image)return;

	QVector2D r=getRange();
	if ( r.x() > r.y()) {
		r.setX( r.y());
		m_min->setText(locale().toString(r.x()));
	}
	else if ( r.x() > r.y()) {
		r.setY(r.x());
		m_max->setText(locale().toString(r.y()));
	}
	m_histoWidget->setRange(r);
	emit rangeChanged(r);
}


void RangeAndHistogramWidget::updateHistogram()
{
	m_histoWidget->setHistogram(m_histo, getRange());
}


void RangeAndHistogramWidget::histogramRangeChanged(const QVector2D & range)
{
	setRangeField(range);
	emit rangeChanged(range);
}

RangeAndHistogramWidget::~RangeAndHistogramWidget() {
}



void RangeAndHistogramWidget::setRange(QVector2D range)
{
    m_min->setText(locale().toString(range.x()));
    m_max->setText(locale().toString(range.y()));

    m_histoWidget->setRange(range);
}

