#ifndef StratiSliceFrequencyPropPanel_H
#define StratiSliceFrequencyPropPanel_H

#include <QWidget>

class QLineEdit;
class StratiSliceFrequencyRep;
class PaletteWidget;
class QSlider;
class QSpinBox;
class QGroupBox;
class QCheckBox;
class AbstractView;
class QToolButton;
class PointPickingTask;

class StratiSliceFrequencyPropPanel: public QWidget {
	Q_OBJECT
public:
	StratiSliceFrequencyPropPanel(StratiSliceFrequencyRep *rep, bool for3D,QWidget *parent);
	virtual ~StratiSliceFrequencyPropPanel();
	void updatePalette();
private slots:
	void valueChanged();
	void sliceChanged(int val);
	void freqSliceChanged(int val);

	void showCrossHair(int value);

	void extractionWindowChanged(unsigned int size);
	void RGTIsoValueChanged(int pos);
	void frequencyIndexChanged();
protected:
	void updateFrequency();
	uint getExtactionWindow();
	QVector2D initSliceRange();
	QWidget* createSliceBox();

	QWidget* createFreqSliceBox();

	void updateSpinValue(int value,QSlider * slider, QSpinBox * spin);
private:
	QLineEdit *m_window;
	StratiSliceFrequencyRep *m_rep;

	QSlider *m_sliceSlider;
	QSpinBox *m_sliceSpin;
	QGroupBox *m_sliderBox;


	QSlider *m_freqSliceSlider;
	QSpinBox *m_freqSliceSpin;
	QGroupBox *m_freqSliderBox;


	PaletteWidget *m_palette;

	QCheckBox * m_showCrossHair;
};

#endif /* QTCUDAIMAGEVIEWER_SRC_APP_RGTHORIZONTALSLICEVIEW_H_ */
