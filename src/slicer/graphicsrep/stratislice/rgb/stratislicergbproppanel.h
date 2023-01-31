#ifndef StratiSliceRGBPropPanel_H
#define StratiSliceRGBPropPanel_H

#include <QWidget>

class StratiSliceRGBAttributeRep;
class RGBPaletteWidget;
class QSlider;
class QSpinBox;
class QGroupBox;
class QSpacerItem;
class QVBoxLayout;
class QToolButton;
class QSplitter;
class QLineEdit;

class StratiSliceRGBPropPanel : public QWidget{
	  Q_OBJECT
public:
	  StratiSliceRGBPropPanel(StratiSliceRGBAttributeRep *rep,QWidget *parent);
	virtual ~StratiSliceRGBPropPanel();

	void setSlicePosition(int pos);

	void updatePalette(int i);

private slots:
	void valueChanged();
	void setRedIndex(int value);
	void setBlueIndex(int value);
	void setGreenIndex(int value);

	void sliceChanged(int val);

	void extractionWindowChanged(unsigned int size);
	void RGTIsoValueChanged(int pos);
	void frequencyChanged();
private:
	uint getExtactionWindow();
	void updateSpinValue(int value,QSlider * slider, QSpinBox * spin);

	void createlinkedSliderSpin(QWidget * parent,QSlider *slider,QSpinBox * spin );
	QWidget * createSlideSpinBox(QString title,QSlider *slider,QSpinBox * spin );
	QWidget * createWindowParameterWidget();
	QWidget * createFreqChooserWidget();

	void updateSpectrum(unsigned int w);
	void updateSliderSpin(int min, int max,QSlider  *slider,QSpinBox * spin);
protected:;
	StratiSliceRGBAttributeRep *m_rep;
	RGBPaletteWidget * m_palette;

	//Extraction window
	QLineEdit *m_window;

	QSlider *m_redSlider;
	QSpinBox *m_redSpin;

	QSlider *m_greenSlider;
	QSpinBox *m_greenSpin;

	QSlider *m_blueSlider;
	QSpinBox *m_blueSpin;

	//RGT Slicing
	QSlider *m_sliceSlider;
	QSpinBox *m_sliceSpin;
};

#endif
