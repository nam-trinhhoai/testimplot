#include "histogramwidget.h"

#include <cmath>
#include <QImage>
#include <QPainter>
#include <QVector>
#include <QMouseEvent>
#include <QDebug>

#include <iostream>
#include "colortableregistry.h"
#include "lutrenderutil.h"

HistogramWidget::HistogramWidget(QWidget *parent) :
		QWidget(parent) {
	setMinimumSize(LUTRenderUtil::INITIAL_WIDTH, LUTRenderUtil::INITIAL_HEIGHT);
	m_histoSet = false;

	m_currentTable = ColorTableRegistry::PALETTE_REGISTRY().DEFAULT();
	m_useLookupTable = true;
	m_defaultColor = QColorConstants::LightGray;

	QPalette pal = palette();
	pal.setColor(QPalette::Base, Qt::gray);
	setAutoFillBackground(true);
}

void HistogramWidget::setHistogramAndLookupTable(const QHistogram &histo,
		const QVector2D &restrictedRange, const LookupTable &table) {
	m_currentTable = table;
	setHistogram(histo, restrictedRange);
}

void HistogramWidget::setHistogram(const QHistogram &histo,
		const QVector2D &restrictedRange) {
	m_histo = histo;

	double maxVal = 0;
	double minVal = 0;
	for (int i = 0; i < QHistogram::HISTOGRAM_SIZE; i++) {
		double val = histo[i];
		minVal = std::min(val, minVal);
		maxVal = std::max(val, maxVal);
	}

	hranges[0] = minVal;
	hranges[1] = maxVal;

	setRange(restrictedRange);

	m_histoSet = true;
	update();
}

void HistogramWidget::setRange(const QVector2D &range) {
	m_min = (int) (QHistogram::HISTOGRAM_SIZE
			* (range.x() - m_histo.range().x())
			/ (m_histo.range().y() - m_histo.range().x()));
	m_min = qMax(0, qMin(m_min, (int) (QHistogram::HISTOGRAM_SIZE - 1)));

	m_max = (int) (QHistogram::HISTOGRAM_SIZE
			* (range.y() - m_histo.range().x())
			/ (m_histo.range().y() - m_histo.range().x()));
	m_max = qMax(0, qMin(m_max, (int) (QHistogram::HISTOGRAM_SIZE - 1)));

	update();
}

/**
 * paintEvent(QPaintEvent *)
 * draw the histogram image corresponing to the image graph
 */
void HistogramWidget::paintEvent(QPaintEvent*) {
	if( m_currentTable.size()==0)
		throw std::runtime_error("Color table is void!");

	int iMaxY = height() - LUTRenderUtil::Y_OFFSET;

	QPainter p(this);

	p.setBrush(QColor(0xFF, 0xFF, 0xFF));

	p.drawLine(LUTRenderUtil::X_MIN, iMaxY + 2, width() - LUTRenderUtil::X_MIN,
			iMaxY + 2);
	p.drawLine(LUTRenderUtil::X_MIN, iMaxY + 2, LUTRenderUtil::X_MIN, 10);

	LUTRenderUtil::drawVArrow(&p);
	LUTRenderUtil::drawHArrow(&p, size());

	if (!m_histoSet)
		return;

	int iH = iMaxY - 10;

	if (m_useLookupTable) {

		double ratioLutHisto = m_currentTable.size()
				/ (double) QHistogram::HISTOGRAM_SIZE;
		double rationHistoRange = QHistogram::HISTOGRAM_SIZE
				/ (m_max - m_min + 1.);

		for (int i = 0; i < QHistogram::HISTOGRAM_SIZE; i++) {
			uint pos = (uint) (m_histo[i] * iH / (hranges[1]) + 0.5);
			double lutInd1 = i * ratioLutHisto;
			// Georges: apply range min max
			std::array<int, 4> c;
			if (lutInd1 <= m_min) {
				c = m_currentTable.getColors(0);
			} else if (lutInd1 >= m_max) {
				c = m_currentTable.getColors(m_currentTable.size() - 1);
			} else {
				int lutInd = round((lutInd1 - m_min) * rationHistoRange);
				if (lutInd >= m_currentTable.size())
					c = m_currentTable.getColors(m_currentTable.size() - 1);
				else
					c = m_currentTable.getColors(lutInd);
			}

			p.setPen(QColor(c[0], c[1], c[2], c[3]));
			p.setBrush(QColor(c[0], c[1], c[2], c[3]));

			int xMin = LUTRenderUtil::convertPositionFromHistogramToCanvas(i,
					QHistogram::HISTOGRAM_SIZE, width());
			int xMax = LUTRenderUtil::convertPositionFromHistogramToCanvas(
					i + 1, QHistogram::HISTOGRAM_SIZE, width());

			p.drawRect(xMin, iMaxY - pos, xMax - xMin, pos);
		}
	} else {
		for (int i = 0; i < QHistogram::HISTOGRAM_SIZE; i++) {
			uint pos = (uint) (m_histo[i] * iH / (hranges[1]) + 0.5);
			p.setPen(m_defaultColor);
			p.setBrush(m_defaultColor);

			int xMin = LUTRenderUtil::convertPositionFromHistogramToCanvas(i,
					QHistogram::HISTOGRAM_SIZE, width());
			int xMax = LUTRenderUtil::convertPositionFromHistogramToCanvas(
					i + 1, QHistogram::HISTOGRAM_SIZE, width());

			p.drawRect(xMin, iMaxY - pos, xMax - xMin, pos);
		}
	}

	p.setPen(QColor("black"));
	LUTRenderUtil::drawButton(m_min, &p, size(), QHistogram::HISTOGRAM_SIZE);
	LUTRenderUtil::drawButton(m_max, &p, size(), QHistogram::HISTOGRAM_SIZE);
}

void HistogramWidget::setUseLookupTable(bool useColorTable) {
	m_useLookupTable = useColorTable;
	update();
}
void HistogramWidget::setDefaultColor(QColor color) {
	m_defaultColor = color;
	update();
}

void HistogramWidget::notifyRangeChanged() {
	QVector2D r;
	r.setX((float) (m_min
			* ((double) m_histo.range().y() - m_histo.range().x())
			/ QHistogram::HISTOGRAM_SIZE + m_histo.range().x()));
	r.setY( (float) (m_max
			* ((double) m_histo.range().y() - m_histo.range().x())
			/ QHistogram::HISTOGRAM_SIZE + m_histo.range().x()));
	emit rangeChanged(r);
}

void HistogramWidget::mouseReleaseEvent(QMouseEvent *event) {
	if (m_minMoving || m_maxMoving) {
		updateIndividualRange(event->x());

		m_minMoving = false;
		m_maxMoving = false;
	}
	if (m_bothMoving) {
		updateBothRange(event->x());
		m_bothMoving = false;
	}

}

void HistogramWidget::mouseMoveEvent(QMouseEvent *event) {
	if (m_minMoving || m_maxMoving) {
		updateIndividualRange(event->x());
	}
	if (m_bothMoving) {
		updateBothRange(event->x());
	}
}

void HistogramWidget::updateIndividualRange(int x) {
	int val = LUTRenderUtil::convertPositionFromCanvasToHistogram(x,
			QHistogram::HISTOGRAM_SIZE, width());
	if (val < 0)
		val = 0;
	if (val > QHistogram::HISTOGRAM_SIZE)
		val = QHistogram::HISTOGRAM_SIZE;

	if (m_minMoving) {
		if (val > m_max)
			return;
		m_min = val;
		update();
	}
	if (m_maxMoving) {
		if (val < m_min)
			return;
		m_max = val;
		update();
	}
	notifyRangeChanged();
}

void HistogramWidget::updateBothRange(int x) {
	int delta = x - m_bothInitialPosition;
	m_bothInitialPosition = x;

	int newMin = m_min + delta;
	int newMax = m_max + delta;
	if (newMin < 0)
		return;
	if (newMax > QHistogram::HISTOGRAM_SIZE)
		return;

	m_min = newMin;
	m_max = newMax;
	update();
	notifyRangeChanged();
}

void HistogramWidget::mousePressEvent(QMouseEvent *event) {
	int minCanvasPosition = LUTRenderUtil::convertPositionFromHistogramToCanvas(
			m_min, QHistogram::HISTOGRAM_SIZE, width());
	int maxCanvasPosition = LUTRenderUtil::convertPositionFromHistogramToCanvas(
			m_max, QHistogram::HISTOGRAM_SIZE, width());

	//Move individually
	m_minMoving = event->x() < (minCanvasPosition + LUTRenderUtil::ARROW_SIZE)
			&& event->x() > (minCanvasPosition - LUTRenderUtil::ARROW_SIZE);
	m_maxMoving = event->x() < (maxCanvasPosition + LUTRenderUtil::ARROW_SIZE)
			&& event->x() > (maxCanvasPosition - LUTRenderUtil::ARROW_SIZE);

	//Move both if we are in between
	if (!m_minMoving && !m_maxMoving) {
		if (event->x() > minCanvasPosition && event->x() < maxCanvasPosition) {
			m_bothInitialPosition = event->x();
			m_bothMoving = true;
		}
	}
}

HistogramWidget::~HistogramWidget() {

}
