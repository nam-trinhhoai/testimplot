#ifndef FixedRGBLayersFromDatasetAndCubePropPanel_H
#define FixedRGBLayersFromDatasetAndCubePropPanel_H

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

class FixedRGBLayersFromDatasetAndCubePropPanel : public QWidget{
	  Q_OBJECT
public:
	  FixedRGBLayersFromDatasetAndCubePropPanel(FixedRGBLayersFromDatasetAndCubeRep *rep,QWidget *parent, bool active = true );
	virtual ~FixedRGBLayersFromDatasetAndCubePropPanel();

	virtual void updatePalette(int i);

private:
	virtual void changeDataKeyFromSlider(long index);
	virtual void changeDataKeyFromSliderInternal(long index);
	virtual void changeDataKeyFromSpinBox();
	virtual void multiplierChanged(int index);
	virtual QWidget* createImageChooserWidget();
	virtual void modeChangedInternal(int index);
	virtual void modeChanged();

	virtual void initProgressBar(int min, int max, int val);
	virtual void valueProgressBarChanged(int val);
	virtual void endProgressBar();

	virtual void changeMinActivated(int state);
	virtual void changeMinSlider(int value);
	virtual void changeMinSpinBox(double value);
	virtual void minActivated(bool activated);
	virtual void minValueChanged(float minValue);

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
};

#endif
