#ifndef RGBLayerRGTPropPanel_H
#define RGBLayerRGTPropPanel_H

#include <QWidget>

class RGBLayerRGTRep;
class RGBPaletteWidget;
class QSlider;
class QSpinBox;
class QDoubleSpinBox;
class QGroupBox;
class QSpacerItem;
class QVBoxLayout;
class QToolButton;
class QSplitter;
class QLineEdit;
class QCheckBox;

class RGBLayerRGTPropPanel : public QWidget{
	  Q_OBJECT
public:
	  RGBLayerRGTPropPanel(RGBLayerRGTRep *rep,QWidget *parent);
	virtual ~RGBLayerRGTPropPanel();

	void updatePalette(int i);

private slots:
	void setRedIndex(int value);
	void setBlueIndex(int value);
	void setGreenIndex(int value);

	void frequencyChanged();
	void updateSpectrum();
	void updateFromComputation();
	void updateFromDataLock();
	void lockStateChange(int state);
	void setRange(unsigned int channel, const QVector2D& range);
private:
	void updateSpinValue(int value,QSlider * slider, QSpinBox * spin, QLineEdit* lineEdit);

	void createlinkedSliderSpin(QWidget * parent,QSlider *slider,QSpinBox * spin, QLineEdit* lineEdit);
	QWidget * createSlideSpinBox(QString title,QSlider *slider,QSpinBox * spin , QLineEdit* lineEdit);
	QWidget * createFreqChooserWidget();

	void changeMinActivated(int state);
	void changeMinSlider(int value);
	void changeMinSpinBox(double value);
	void minActivated(bool activated);
	void minValueChanged(float minValue);

	void updateSliderSpin(int min, int max,QSlider  *slider,QSpinBox * spin);
protected:;
	RGBLayerRGTRep *m_rep;
	RGBPaletteWidget * m_palette;

	QCheckBox* m_lockCheckBox;

	QSlider *m_redSlider;
	QSpinBox *m_redSpin;
	QLineEdit *m_redLineEdit;

	QSlider *m_greenSlider;
	QSpinBox *m_greenSpin;
	QLineEdit *m_greenLineEdit;

	QSlider *m_blueSlider;
	QSpinBox *m_blueSpin;
	QLineEdit *m_blueLineEdit;

	QCheckBox* m_valueMinCheckBox;
	QSlider* m_valueMinSlider;
	QDoubleSpinBox* m_valueMinSpinBox;

	long m_oldDeltaRed = 0;
	long m_oldDeltaBlue = 0;
};

#endif
