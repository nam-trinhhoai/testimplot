/*
 * palettewidget.h
 *
 *  Created on: 4 mai 2018
 *      Author: j0334308
 */

#ifndef PALETTEWIDGET_H_
#define PALETTEWIDGET_H_

#include <QWidget>

#include "lookuptable.h"
#include "qhistogram.h"

class QLineEdit;
class QPushButton;
class QSlider;
class QComboBox;
class ColorTableSelector;
class LUTEditDialog;


class HistogramWidget;

class IPaletteHolder;

class PaletteWidget : public QWidget{
	Q_OBJECT

	Q_PROPERTY(LookupTable lookupTable READ getLookupTable WRITE setLookupTable NOTIFY lookupTableChanged)
	Q_PROPERTY(QVector2D range READ getRange WRITE setRange NOTIFY rangeChanged)
	Q_PROPERTY(float opacity READ getOpacity WRITE setOpacity NOTIFY opacityChanged)
public:
	PaletteWidget(QWidget* parent = 0, Qt::WindowFlags f = Qt::Widget );
	virtual ~PaletteWidget();

	void setPaletteHolder(IPaletteHolder *image);

	LookupTable getLookupTable();
	float getOpacity() const;
	QVector2D getRange() const;

public slots:
	void setLookupTable(const LookupTable &table);
	void setOpacity(float val);
    void setRange(const QVector2D &range);

signals:
	void rangeChanged(const QVector2D & range);
	void opacityChanged(float opacity);
	void lookupTableChanged(const LookupTable & colorTable);
private slots:
	void resetRange();
	void recompute();
	void smartAdjust();

	void valueChanged();

	void histogramRangeChanged(const QVector2D &);
	void opacityChanged(int value);
	void setRangeField(const QVector2D &r);

	void colorTableIndexChanged();
	void colorTableEdited();

	void lookupTableChangedInternal(const LookupTable& colorTable);
private:
	QWidget * createLUTWidget();
	void updateHistogramAndColorTable();
	void updateColorTable();
private:
	LUTEditDialog * m_editDialog;
	ColorTableSelector* m_paletteSelector= nullptr;
	QPushButton *m_edit;
#if 0
	QPushButton *m_paletteChooser;
#endif
	QComboBox *m_list;

	HistogramWidget *m_histoWidget;
	QSlider *m_opacity;

	QLineEdit * m_min;
	QLineEdit * m_max;

	QPushButton *m_reset;
	QPushButton *m_recompute;
	QPushButton *m_wand;

	IPaletteHolder *m_image;

	QVector2D m_originalRange;

	LookupTable m_lookupTable;
	QHistogram m_histo;
};

#endif
