

#ifndef __FixedRGBLayersFromDatasetAndCube3FreqPropPanel__
#define __FixedRGBLayersFromDatasetAndCube3FreqPropPanel__


#include <QWidget>
#include <QMutex>

class FixedRGBLayersFromDatasetAndCubeRep;
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
class QComboBox;
class QCheckBox;
class EditingSpinBox;
class QProgressBar;
class QHBoxLayout;

#include <fixedrgblayersfromdatasetandcubeproppanel.h>

class FixedRGBLayersFromDatasetAndCube3FreqPropPanel : public FixedRGBLayersFromDatasetAndCubePropPanel {
	  Q_OBJECT
public:
	  FixedRGBLayersFromDatasetAndCube3FreqPropPanel(FixedRGBLayersFromDatasetAndCubeRep *rep,QWidget *parent);
	virtual ~FixedRGBLayersFromDatasetAndCube3FreqPropPanel();

	void updatePalette(int i);

private:
	void changeDataKeyFromSlider(long index);
	void changeDataKeyFromSliderInternal(long index);
	void changeDataKeyFromSpinBox();
	void multiplierChanged(int index);
	QWidget* createImageChooserWidget();
	void modeChangedInternal(int index);
	void modeChanged();

	void initProgressBar(int min, int max, int val);
	void valueProgressBarChanged(int val);
	void endProgressBar();

	void changeMinActivated(int state);
	void changeMinSlider(int value);
	void changeMinSpinBox(double value);
	void minActivated(bool activated);
	void minValueChanged(float minValue);
	QWidget* createFreqChooserWidget();
	QWidget *createGrayFreqChooserWidget();
	void createlinkedSliderSpin(QWidget *parent, QSlider *slider, QSpinBox *spin, QLineEdit *lineEdit);

private slots:
	void setRedIndex(int value);
	void setGreenIndex(int value);
	void setBlueIndex(int value);
	void setRedSpinIndex(int value);
	void setGreenSpinIndex(int value);
	void setBlueSpinIndex(int value);
	void frequencyChanged();
	void displayModeChanged(int index);
	void setGrayFreqIndex(int value);
	void setGrayFreqSpinIndex(int value);



private:
	FixedRGBLayersFromDatasetAndCubeRep *m_rep;
	RGBPaletteWidget * m_palette;
	EditingSpinBox* m_layerNameSpinBox;
	QSlider* m_slider;
	QComboBox* m_multiplierComboBox;
	QComboBox* m_modeComboBox;
	QProgressBar* m_progressBar;

	QSpinBox* m_simplifyStepsSpinBox;
	QSpinBox* m_compressionSpinBox;

	QCheckBox* m_valueMinCheckBox;
	QSlider* m_valueMinSlider;
	QDoubleSpinBox* m_valueMinSpinBox;

	QToolButton* m_playButton;
	QToolButton* m_loopButton;


	int m_stepMultiplier = 1;
	QMutex m_mutex;
	QMutex m_mutexIndexList;
	QList<long> m_sliderIndexList;
	QSlider *m_redSlider;
	QSpinBox *m_redSpin;
	QLineEdit *m_redLineEdit;

	QSlider *m_greenSlider;
	QSpinBox *m_greenSpin;
	QLineEdit *m_greenLineEdit;

	QSlider *m_blueSlider;
	QSpinBox *m_blueSpin;
	QLineEdit *m_blueLineEdit;

	QComboBox *m_displayModeComboBox = nullptr;
	QWidget *m_frequencyColorWidget = nullptr;
	QWidget *m_frequencyGrayWidget = nullptr;
	QSlider *m_graySlider = nullptr;
	QSpinBox *m_graySpin = nullptr;
	QLineEdit *m_grayLineEdit;

	int m_freqRedSave = 0;
	int m_freqGreenSave = 0;
	int m_freqBlueSave = 0;
	int m_freqGraySave = 0;

	int m_nbreSpectrumFreq = -1;
	int m_oldDeltaRed = 0;
	int m_oldDeltaBlue = 0;
	void updateSpinValue(int value, QSlider *slider, QSpinBox *spin, QLineEdit* lineEdit);



};

#endif



