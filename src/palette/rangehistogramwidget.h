#ifndef QTCUDAIMAGEVIEWER_SRC_RGB_RANGEHISTOGRAMWIDGET_H_
#define QTCUDAIMAGEVIEWER_SRC_RGB_RANGEHISTOGRAMWIDGET_H_

#include <QWidget>

#include "lookuptable.h"
#include "qhistogram.h"

class QLineEdit;
class QPushButton;

class HistogramWidget;
class IPaletteHolder;

class RangeAndHistogramWidget : public QWidget{
	Q_OBJECT

	Q_PROPERTY(QVector2D range READ getRange CONSTANT WRITE setRange NOTIFY rangeChanged)
public:
	RangeAndHistogramWidget(QWidget* parent = 0, Qt::WindowFlags f = Qt::Widget );
	virtual ~RangeAndHistogramWidget();

    void setUseLookupTable(bool useColorTable);
    void setDefaultColor(QColor color);

	void setPaletteHolder(IPaletteHolder *image);

	QVector2D getRange() const;

    void setRange(QVector2D range);

signals:
	void rangeChanged(const QVector2D & range);

public slots:
	void resetRange();
	void recompute();
	void smartAdjust();


	void valueChanged();

	void histogramRangeChanged(const QVector2D &);
	void setRangeField(const QVector2D &r);
private:
	void updateHistogram();
private:
	HistogramWidget *m_histoWidget;

	QLineEdit * m_min;
	QLineEdit * m_max;

	QPushButton *m_reset;
	QPushButton *m_recompute;
	QPushButton *m_wand;

	IPaletteHolder *m_image;

	QVector2D m_originalRange;
	QHistogram m_histo;
};


#endif /* QTCUDAIMAGEVIEWER_SRC_RGB_RANGEHISTOGRAMWIDGET_H_ */
