#ifndef StratiSliceAmplitudePropPanel_H
#define StratiSliceAmplitudePropPanel_H

#include <QWidget>

class QLineEdit;
class StratiSliceAmplitudeRep;
class PaletteWidget;
class QSlider;
class QSpinBox;
class QGroupBox;
class QCheckBox;
class AbstractView;
class QToolButton;
class PointPickingTask;

class StratiSliceAmplitudePropPanel: public QWidget {
	Q_OBJECT
public:
	StratiSliceAmplitudePropPanel(StratiSliceAmplitudeRep *rep, bool for3D,QWidget *parent);
	virtual ~StratiSliceAmplitudePropPanel();
	void updatePalette();
private slots:
	void valueChanged();
	void sliceChanged(int val);
	void showCrossHair(int value);

	void extractionWindowChanged(unsigned int size);
	void RGTIsoValueChanged(int pos);
protected:
	uint getExtactionWindow();
	QVector2D initSliceRange();
	QWidget* createSliceBox();

	void updateSpinValue(int value,QSlider * slider, QSpinBox * spin);
private:
	QLineEdit *m_window;
	StratiSliceAmplitudeRep *m_rep;
	QSlider *m_sliceSlider;
	QSpinBox *m_sliceSpin;
	QGroupBox *m_sliderBox;
	PaletteWidget *m_palette;

	QCheckBox * m_showCrossHair;
};

#endif /* QTCUDAIMAGEVIEWER_SRC_APP_RGTHORIZONTALSLICEVIEW_H_ */
